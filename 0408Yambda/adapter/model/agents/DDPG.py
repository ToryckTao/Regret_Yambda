import numpy as np
import torch
import torch.nn.functional as F

import utils
from _loader import load_hsrl_module

_orig = load_hsrl_module("model/agents/DDPG.py", "_hsrl_original_ddpg")


class DDPG(_orig.DDPG):
    """
    Yambda SID 训练用 DDPG 覆盖层。

    核心变化：
    - 修正 feedback 为 numpy 时 `.to()` 报错
    - SID policy 同时有 sid_logits/action_prob 时优先走 token-level PG 分支
    """

    def step_train(self):
        observation, policy_output, reward, done_mask, next_observation, feedback = self.facade.sample_buffer(self.batch_size)
        if isinstance(reward, np.ndarray):
            reward = torch.FloatTensor(reward).to(self.device)
        elif isinstance(reward, torch.Tensor):
            reward = reward.float().to(self.device)
        if isinstance(done_mask, np.ndarray):
            done_mask = torch.FloatTensor(done_mask).to(self.device)
        elif isinstance(done_mask, torch.Tensor):
            done_mask = done_mask.float().to(self.device)
        if isinstance(feedback, np.ndarray):
            feedback = torch.FloatTensor(feedback).to(self.device)
        elif isinstance(feedback, torch.Tensor):
            feedback = feedback.float().to(self.device)

        critic_loss, actor_loss = self.get_ddpg_loss(observation, policy_output, reward, done_mask, next_observation, feedback)
        self.training_history["actor_loss"].append(actor_loss.item())
        self.training_history["critic_loss"].append(critic_loss.item())
        if hasattr(self, "_last_entropy_loss") and self._last_entropy_loss is not None:
            self.training_history["entropy_loss"].append(self._last_entropy_loss.item())
            self.training_history["bc_loss"].append(self._last_bc_loss.item())
            self.training_history["advantage"].append(self._last_advantage.item())

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {"step_loss": (self.training_history["actor_loss"][-1], self.training_history["critic_loss"][-1])}

    def get_ddpg_loss(
        self,
        observation,
        policy_output,
        reward,
        done_mask,
        next_observation,
        feedback=None,
        do_actor_update=True,
        do_critic_update=True,
    ):
        current_critic_output = self.facade.apply_critic(
            observation,
            utils.wrap_batch(policy_output, device=self.device),
            self.critic,
        )
        current_Q = current_critic_output["q"]

        next_policy_output = self.facade.apply_policy(next_observation, self.actor_target)
        target_critic_output = self.facade.apply_critic(next_observation, next_policy_output, self.critic_target)
        target_Q = target_critic_output["q"]
        target_Q = reward + self.gamma * (done_mask * target_Q).detach()

        critic_loss = F.mse_loss(current_Q, target_Q).mean()
        if do_critic_update and self.critic_lr > 0:
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        cur_policy_output = self.facade.apply_policy(observation, self.actor)
        with torch.no_grad():
            v_s = current_Q.detach().mean()
            advantage = torch.clamp(target_Q - v_s, -1, 1).mean()

        if "sid_logits" in cur_policy_output:
            sid_tokens = policy_output["sid_tokens"]
            sid_logits_list = cur_policy_output["sid_logits"]
            batch_size, slate_size, n_level = sid_tokens.shape

            sid_tokens_flat = sid_tokens.view(batch_size * slate_size, n_level)
            level_probs = [torch.softmax(logits_l, dim=-1) for logits_l in sid_logits_list]
            level_probs_flat = [p.repeat_interleave(slate_size, dim=0) for p in level_probs]

            nll_slot = 0.0
            for level in range(n_level):
                probs_l_flat = level_probs_flat[level]
                z_l = sid_tokens_flat[:, level].view(-1, 1)
                logp_l = torch.log(torch.gather(probs_l_flat, 1, z_l) + 1e-12).squeeze(1)
                nll_slot = nll_slot + (-logp_l)

            nll_per_sample = nll_slot.view(batch_size, slate_size).mean(dim=1)
            pg_loss = (nll_per_sample * advantage).mean()
        elif "action_prob" in cur_policy_output:
            action_indices = policy_output.get("action", None)
            if action_indices is not None:
                action_prob = cur_policy_output["action_prob"]
                n_action = action_prob.shape[1]
                action_indices = action_indices.long()
                if action_indices.min() < 0 or action_indices.max() >= n_action:
                    critic_output = self.facade.apply_critic(observation, cur_policy_output, self.critic)
                    pg_loss = -critic_output["q"].mean()
                else:
                    batch_idx = torch.arange(action_prob.shape[0], device=action_prob.device).unsqueeze(1)
                    selected_prob = action_prob[batch_idx, action_indices].detach()
                    action_log_prob = torch.log(selected_prob + 1e-12).mean()
                    pg_loss = -advantage * action_log_prob
            else:
                critic_output = self.facade.apply_critic(observation, cur_policy_output, self.critic)
                pg_loss = -critic_output["q"].mean()
        else:
            critic_output = self.facade.apply_critic(observation, cur_policy_output, self.critic)
            pg_loss = -critic_output["q"].mean()

        entropy_loss = torch.tensor(0.0, device=self.device)
        if "sid_logits" in cur_policy_output:
            for logits_l in cur_policy_output["sid_logits"]:
                probs = torch.softmax(logits_l, dim=-1)
                entropy_loss = entropy_loss + -(probs * torch.log(probs + 1e-12)).sum(dim=-1).mean()
            entropy_loss = entropy_loss / len(cur_policy_output["sid_logits"])
        elif "action_prob" in cur_policy_output:
            probs = cur_policy_output["action_prob"]
            entropy_loss = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).mean()

        bc_loss = torch.tensor(0.0, device=self.device)
        if feedback is not None:
            if isinstance(feedback, np.ndarray):
                feedback = torch.FloatTensor(feedback).to(self.device)
            else:
                feedback = feedback.float().to(self.device)

            if "action_prob" in cur_policy_output:
                action_prob = cur_policy_output["action_prob"]
                action_indices = policy_output.get("action", None)
                n_action = action_prob.shape[1]
                if action_indices is not None:
                    action_indices = action_indices.long()
                    if action_indices.min() >= 0 and action_indices.max() < n_action:
                        batch_idx = torch.arange(action_prob.shape[0], device=action_prob.device).unsqueeze(1)
                        selected_prob = action_prob[batch_idx, action_indices].detach()
                        epsilon = 1e-6
                        bc_loss = -(feedback * torch.log(selected_prob + epsilon)).sum(dim=-1).mean()
                        pos_ratio = (feedback > 0).float().mean() + epsilon
                        bc_loss = bc_loss / pos_ratio

        actor_loss = pg_loss - self.entropy_coef * entropy_loss + self.bc_coef * bc_loss
        if do_actor_update and self.actor_lr > 0:
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        self._last_pg_loss = pg_loss.detach()
        self._last_entropy_loss = entropy_loss.detach()
        self._last_bc_loss = bc_loss.detach()
        self._last_advantage = advantage.detach()
        return critic_loss, actor_loss
