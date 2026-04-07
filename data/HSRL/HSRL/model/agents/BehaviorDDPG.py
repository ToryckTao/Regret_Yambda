import time
import copy
import torch
import torch.nn.functional as F
import numpy as np

import utils
from model.agents.BaseRLAgent import BaseRLAgent
from model.agents.DDPG import DDPG
    
class BehaviorDDPG(DDPG):
    '''
    DDPG with behavior feedback signal + SARA Delayed Regret Queue
    '''
    
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - behavior_lr
        - behavior_decay
        - advantage_bias
        - entropy_coef
        - from DDPG:
            ...
        '''
        parser = DDPG.parse_model_args(parser)
        parser.add_argument('--behavior_lr', type=float, default=0.0001,
                            help='behaviorvise loss coefficient')
        parser.add_argument('--behavior_decay', type=float, default=0.00003,
                            help='behaviorvise loss coefficient')
        parser.add_argument('--advantage_bias', type=float, default=0,
                            help='advantage bias term')
        parser.add_argument('--entropy_coef', type=float, default=0.1,
                            help='entropy regularization coefficient')
        return parser
    
    
    def __init__(self, args, facade):
        super().__init__(args, facade)
        self.behavior_lr = args.behavior_lr
        self.behavior_decay = args.behavior_decay
        self.actor_behavior_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.behavior_lr,
                                                         weight_decay=args.behavior_decay)

        # Advantage and entropy parameters
        self.advantage_bias = getattr(args, 'advantage_bias', 0)
        self.entropy_coef = getattr(args, 'entropy_coef', 0.1)

        # ========== SARA: 延迟后悔事件队列 ==========
        # 队列中存储字典: {'trigger_step': int, 'buffer_index': int, 'phi': float, 'sid_tokens': tensor}
        self.regret_event_queue = []
        # ============================================
        
    def action_before_train(self):
        super().action_before_train()
        self.training_history["behavior_loss"] = []
        self.training_history["entropy_loss"] = []
        self.training_history["advantage"] = []
        self.regret_event_queue = [] # 每次训练开始前清空队列
    
    def run_episode_step(self, step, epsilon, observation, do_explore):
        """
        覆写父类方法，植入 SARA 埋雷和排雷逻辑
        """
        # ================= SARA: 1. 结算(引爆)到期的后悔炸弹 =================
        # 找出所有到期的炸弹
        ready_events = [e for e in self.regret_event_queue if e['trigger_step'] <= step]
        
        # 将未到期的炸弹保留在队列中 (防止无限累积导致 OOM)
        self.regret_event_queue = [e for e in self.regret_event_queue if e['trigger_step'] > step]
        
        # 依次引爆到期炸弹，修改 Facade 里的历史账本
        for event in ready_events:
            if hasattr(self.facade, 'override_reward'):
                self.facade.override_reward(
                    buffer_idx=event['buffer_index'],
                    new_reward=event['phi'],       # 用 phi (-lambda) 覆盖原始 reward
                    sid_tokens=event['sid_tokens'],
                    phi=event['phi']
                )
        # ====================================================================

        # === 以下为原版交互逻辑 ===
        policy_output = self.facade.apply_policy(observation, self.actor, epsilon, do_explore=do_explore)
        next_observation, reward, done_mask, info = self.facade.env_step(policy_output)
        
        # 存入 Buffer，并获取这些数据在 Buffer 中的物理索引 [B]
        stored_indices = self.facade.update_buffer(observation, policy_output, reward, done_mask, next_observation, info)
        # =========================

        # ================= SARA: 2. 预埋新生成的后悔炸弹 =================
        # 如果环境生成了炸弹 (sara_delta_t > 0)，就把它压入队列
        if 'sara_delta_t' in info:
            delta_t_matrix = info['sara_delta_t'] # shape: (B, slate_size)
            B, slate_size = delta_t_matrix.shape
            
            for b in range(B):
                for s in range(slate_size):
                    dt = delta_t_matrix[b, s].item()
                    if dt > 0:
                        # 触发了后悔，计算引爆时间
                        trigger_step = step + dt
                        
                        # 惩罚值 phi = -lambda_unlike * gamma^dt
                        # 假设环境传来了 lambda_unlike，或者你可以从 args 里读取。
                        # 为了简洁，我们这里用固定的 -1.0 惩罚值乘上衰减
                        gamma = getattr(self.facade.env, 'regret_gamma', 0.9)
                        lambda_unlike = getattr(self.facade.env, 'lambda_unlike', 1.0)
                        phi_val = -lambda_unlike * (gamma ** dt)
                        phi_tensor = torch.tensor(phi_val, dtype=torch.float32)
                        
                        # 获取引发后悔的具体物品 SID (用于前端拦截)
                        # policy_output['sid_tokens'] shape: (B, slate_size, L)
                        sid_tokens = policy_output['sid_tokens'][b, s].clone()
                        
                        # 压入队列
                        self.regret_event_queue.append({
                            'trigger_step': trigger_step,
                            'buffer_index': stored_indices[b], # 指向 Facade Buffer 中的对应行
                            'phi': phi_tensor,
                            'sid_tokens': sid_tokens
                        })
        # =================================================================
        
        return next_observation

    def get_behavior_loss(self, observation, policy_output, next_observation, do_update = True):
        observation, exposure, feedback = self.facade.extract_behavior_data(observation, policy_output, next_observation)
        observation['candidate_ids'] = exposure['ids']
        observation['candidate_features'] = exposure['features']
        policy_output = self.facade.apply_policy(observation, self.actor, do_softmax = False)
        action_prob = torch.sigmoid(policy_output['candidate_prob'])
        behavior_loss = F.binary_cross_entropy(action_prob, feedback)
        
        if do_update and self.behavior_lr > 0:
            self.actor_behavior_optimizer.zero_grad()
            behavior_loss.backward()
            self.actor_behavior_optimizer.step()
        return behavior_loss

    def step_train(self):
        if 'behavior_loss' not in self.training_history:
            self.training_history['behavior_loss'] = []
            self.training_history['entropy_loss'] = []
            self.training_history['advantage'] = []

        observation, policy_output, reward, done_mask, next_observation = self.facade.sample_buffer(self.batch_size)

        # 使用新的 A2C-style loss（包含 Advantage 计算和逐token loss）
        critic_loss, actor_loss, entropy_loss, advantage = self.get_a2c_loss(
            observation, policy_output, reward, done_mask, next_observation
        )
        behavior_loss = self.get_behavior_loss(observation, policy_output, next_observation)

        self.training_history['actor_loss'].append(actor_loss.item())
        self.training_history['critic_loss'].append(critic_loss.item())
        self.training_history['behavior_loss'].append(behavior_loss.item())
        self.training_history['entropy_loss'].append(entropy_loss.item())
        self.training_history['advantage'].append(advantage.item())

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {"step_loss": (self.training_history['actor_loss'][-1],
                              self.training_history['critic_loss'][-1],
                              self.training_history['behavior_loss'][-1],
                              self.training_history['entropy_loss'][-1])}

    def get_a2c_loss(self, observation, policy_output, reward, done_mask, next_observation,
                     do_actor_update=True, do_critic_update=True):
        """
        基于 A2C_SID_rl4rs.py 实现的 Advantage + 逐token loss + 熵正则化
        符合原文 Eq. 18-19, Eq. 49
        """
        # ---- Critic Target ----
        with torch.no_grad():
            next_po = self.facade.apply_policy(
                next_observation, self.actor_target,
                epsilon=0.0, do_explore=False
            )
            V_sp_out = self.critic_target({'context_list': next_po['context_list']})
            V_sp_seq = V_sp_out['v_seq']  # (B, L+1)

            # 使用 Critic 中的可学习权重加权
            if hasattr(self.critic, 'token_weight'):
                weights = F.softmax(self.critic.token_weight, dim=0).unsqueeze(0).expand(V_sp_seq.size(0), -1)
            else:
                weights = torch.ones_like(V_sp_seq) / V_sp_seq.size(1)

            V_sp_weighted = (V_sp_seq * weights).sum(dim=1)  # (B,)
            Q_s = reward + self.gamma * (done_mask * V_sp_weighted)

        # ---- 当前状态 Critic ----
        cur_po = self.facade.apply_policy(
            observation, self.actor,
            epsilon=0.0, do_explore=False
        )

        V_s_out = self.critic({'context_list': cur_po['context_list']})
        V_s_seq = V_s_out['v_seq']  # (B, L+1)

        if hasattr(self.critic, 'token_weight'):
            weights = F.softmax(self.critic.token_weight, dim=0).unsqueeze(0).expand(V_s_seq.size(0), -1)
        else:
            weights = torch.ones_like(V_s_seq) / V_s_seq.size(1)

        V_s_weighted = (V_s_seq * weights).sum(dim=1)  # (B,)

        # Critic Loss (Eq. 46)
        value_loss = F.mse_loss(V_s_weighted, Q_s)

        if do_critic_update and self.critic_lr > 0:
            self.critic_optimizer.zero_grad()
            value_loss.backward(retain_graph=True)
            self.critic_optimizer.step()

        # ---- Actor 部分: 逐token NLL + 熵正则化 ----
        sid_tokens = policy_output['sid_tokens']  # (B, K, L)
        sid_logits_list = cur_po['sid_logits']    # list[(B, V_l)]
        B, K, L = sid_tokens.shape

        # flatten (B,K,*) -> (B*K,*)
        sid_tokens_flat = sid_tokens.view(B * K, L)
        level_probs = [torch.softmax(logits_l, dim=-1) for logits_l in sid_logits_list]
        level_probs_flat = [p.repeat_interleave(K, dim=0) for p in level_probs]

        # === NLL 计算 (Eq. 49 第一项) ===
        nll_slot = 0.0
        entropy_raw = 0.0
        for l in range(L):
            probs_l_flat = level_probs_flat[l]  # (B*K, V_l)
            z_l = sid_tokens_flat[:, l].view(-1, 1)  # (B*K, 1)
            logp_l = torch.log(torch.gather(probs_l_flat, 1, z_l) + 1e-12).squeeze(1)
            nll_slot = nll_slot + (-logp_l)

            # 熵项
            entropy_raw = entropy_raw + \
                (level_probs[l] * torch.log(level_probs[l] + 1e-12)).sum(dim=-1).mean()

        # (B*K,) -> (B,K) -> (B,)
        nll_per_sample = nll_slot.view(B, K).mean(dim=1)

        # ---- Advantage (Eq. 18-19) ----
        with torch.no_grad():
            advantage = torch.clamp(Q_s - V_s_weighted, -1, 1).view(-1)  # (B,)

        # Actor Loss = NLL * Advantage + entropy_coef * entropy (Eq. 49)
        actor_loss = (nll_per_sample * (advantage + self.advantage_bias)).mean()
        total_actor = actor_loss + self.entropy_coef * entropy_raw

        if do_actor_update and self.actor_lr > 0:
            self.actor_optimizer.zero_grad()
            total_actor.backward()
            self.actor_optimizer.step()

        return value_loss.detach(), actor_loss.detach(), entropy_raw.detach(), advantage.mean().detach()

    def test(self):
        """
        评测（复用训练时的交互逻辑）：
        """
        import torch
        from time import time
        from tqdm import tqdm

        if not hasattr(self.facade, "current_buffer_size"):
            self.facade.current_buffer_size = 0
        if not hasattr(self.facade, "update_buffer"):
            def _no_buffer(*args, **kwargs): return None
            self.facade.update_buffer = _no_buffer

        if hasattr(self.facade, "initialize_train"):
            try:
                self.facade.initialize_train()
            except Exception as e:
                print(f"[TEST] WARN: initialize_train() failed: {e}")

        self.training_history = {"critic_loss": [], "actor_loss": [], "behavior_loss": [],
                                  "entropy_loss": [], "advantage": []}

        try:
            self.load()
            print("[TEST] Loaded trained weights from save_path prefix.")
        except Exception as e:
            print(f"[TEST] WARN: load() failed or weights not found, continue. Detail: {e}")

        self.actor.eval()
        if hasattr(self, "critic") and self.critic is not None:
            self.critic.eval()
        torch.set_grad_enabled(False)

        print("[TEST] Reset environment for evaluation.")
        observation = self.facade.reset_env({"batch_size": self.episode_batch_size})

        total_steps = self.n_iter[-1] if isinstance(self.n_iter, (list, tuple)) else int(self.n_iter)
        start_time = time()
        last_t = start_time
        print("[TEST] Start rollout (epsilon=0, buffer on, no optimization).")
        for i in tqdm(range(total_steps)):
            observation = self.run_episode_step(i, 0.0, observation, True)

            if i % self.check_episode == 0:
                now = time()
                print(f"[TEST] step {i} | dt={now - last_t:.2f}s | total={now - start_time:.2f}s")
                print(self.log_iteration(i))
                last_t = now

        if hasattr(self.facade, "stop_env"):
            try: self.facade.stop_env()
            except: pass
        torch.set_grad_enabled(True)
        print("[TEST] Evaluation finished.")