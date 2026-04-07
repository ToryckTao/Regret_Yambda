import time
import copy
import torch
import torch.nn.functional as F
import numpy as np

import utils
from model.agents.BaseRLAgent import BaseRLAgent
    
class DDPG(BaseRLAgent):
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - episode_batch_size
        - batch_size
        - actor_lr
        - critic_lr
        - actor_decay
        - critic_decay
        - target_mitigate_coef
        - args from BaseRLAgent:
            - gamma
            - n_iter
            - train_every_n_step
            - initial_greedy_epsilon
            - final_greedy_epsilon
            - elbow_greedy
            - check_episode
            - with_eval
            - save_path
        '''
        parser = BaseRLAgent.parse_model_args(parser)
        parser.add_argument('--episode_batch_size', type=int, default=8, 
                            help='episode sample batch size')
        parser.add_argument('--batch_size', type=int, default=32, 
                            help='training batch size')
        parser.add_argument('--actor_lr', type=float, default=1e-4, 
                            help='learning rate for actor')
        parser.add_argument('--critic_lr', type=float, default=1e-4, 
                            help='decay rate for critic')
        parser.add_argument('--actor_decay', type=float, default=1e-4, 
                            help='learning rate for actor')
        parser.add_argument('--critic_decay', type=float, default=1e-4, 
                            help='decay rate for critic')
        parser.add_argument('--target_mitigate_coef', type=float, default=0.01, 
                            help='mitigation factor')
        parser.add_argument('--entropy_coef', type=float, default=0.01, 
                            help='entropy regularization coefficient (lambda_en)')
        parser.add_argument('--bc_coef', type=float, default=0.1, 
                            help='behavior cloning coefficient (lambda_BC)')
        return parser
    
    
    def __init__(self, args, facade):
        '''
        self.gamma
        self.n_iter
        self.check_episode
        self.with_eval
        self.save_path
        self.facade
        self.exploration_scheduler
        '''
        super().__init__(args, facade)
        self.episode_batch_size = args.episode_batch_size
        self.batch_size = args.batch_size
        
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.actor_decay = args.actor_decay
        self.critic_decay = args.critic_decay
        
        self.actor = facade.actor
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr, 
                                                weight_decay=args.actor_decay)

        self.critic = facade.critic
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr, 
                                                 weight_decay=args.critic_decay)

        self.tau = args.target_mitigate_coef
        
        # Entropy and behavior cloning coefficients
        self.entropy_coef = getattr(args, 'entropy_coef', 0.01)
        self.bc_coef = getattr(args, 'bc_coef', 0.1)
        
        if len(self.n_iter) == 1:
            with open(self.save_path + ".report", 'w') as outfile:
                outfile.write(f"{args}\n")
        
    def action_before_train(self):
        '''
        Action before training:
        - facade setup:
            - buffer setup
        - run random episodes to build-up the initial buffer
        '''
        self.facade.initialize_train() # buffer setup
        prepare_step = 0
        # random explore before training
        initial_epsilon = 1.0
        observation = self.facade.reset_env({"batch_size": self.episode_batch_size})
        while not self.facade.is_training_available:
            observation = self.run_episode_step(0, initial_epsilon, observation, True)
            prepare_step += 1
        # training records
        self.training_history = {"critic_loss": [], "actor_loss": [], "entropy_loss": [], "bc_loss": [], "advantage": []}
        
        print(f"Total {prepare_step} prepare steps")
        
        
    def run_episode_step(self, *episode_args):
        '''
        One step of interaction
        '''
        episode_iter, epsilon, observation, do_buffer_update = episode_args
        with torch.no_grad():
            # sample action
            policy_output = self.facade.apply_policy(observation, self.actor, epsilon, do_explore = True)
            # apply action on environment and update replay buffer
            next_observation, reward, done, info = self.facade.env_step(policy_output)
            # update replay buffer
            if do_buffer_update:
                self.facade.update_buffer(observation, policy_output, reward, done, next_observation, info)
        return next_observation
            
#     def run_an_episode(self, epsilon, initial_observation = None, with_train = False, pick_rows = None):
#         '''
#         Run episode for a batch of user
#         @input:
#         - epsilon: greedy epsilon for random exploration
#         - initial_observation
#         - with_train: apply batch training for each step of the episode
#         - pick_rows: pick certain rows of the data when reseting the environment
#         '''
#         # observation --> state, action
#         if initial_observation:
#             observation = initial_observation
#         elif pick_rows:
#             observation = self.facade.reset_env({"batch_size": len(pick_rows), 'pick_rows': pick_rows})
#         else:
#             observation = self.facade.reset_env({"batch_size": self.episode_batch_size})
#         step = 0
#         done = [False] * self.batch_size
#         train_report = None
#         while sum(done) < len(done):
#             step += 1
#             with torch.no_grad():
#                 # sample action
#                 policy_output = self.facade.apply_policy(observation, self.actor, epsilon, do_explore = True)
#                 # apply action on environment and update replay buffer
#                 next_observation, reward, done, info = self.facade.env_step(policy_output)
#                 # update replay buffer
#                 if not pick_rows:
#                     self.facade.update_buffer(observation, policy_output, reward, done, next_observation, info)
#                 # observate for the next step
#                 observation = next_observation
#             if with_train:
#                 train_report = self.step_train()
#         episode_reward = self.facade.get_total_reward()
#         return {'average_total_reward': np.mean(episode_reward['total_rewards']),
#                 'reward_variance': np.var(episode_reward['total_rewards']),
#                 'max_total_reward': np.max(episode_reward['total_rewards']),
#                 'min_total_reward': np.min(episode_reward['total_rewards']),
#                 'average_n_step': np.mean(episode_reward['n_step']),
#                 'step': step, 
#                 'buffer_size': self.facade.current_buffer_size}, train_report

    def step_train(self):
        observation, policy_output, reward, done_mask, next_observation, feedback = self.facade.sample_buffer(self.batch_size)
        # Handle both numpy arrays and tensors (could be on cpu or cuda)
        if isinstance(reward, np.ndarray):
            reward = torch.FloatTensor(reward).to(self.device)
        elif isinstance(reward, torch.Tensor):
            reward = reward.float().to(self.device)
        if isinstance(done_mask, np.ndarray):
            done_mask = torch.FloatTensor(done_mask).to(self.device)
        elif isinstance(done_mask, torch.Tensor):
            done_mask = done_mask.float().to(self.device)
        if isinstance(feedback, np.ndarray):
            feedback = feedback.to(self.device)
        
        critic_loss, actor_loss = self.get_ddpg_loss(observation, policy_output, reward, done_mask, next_observation, feedback)
        
        # Record all losses
        self.training_history['actor_loss'].append(actor_loss.item())
        self.training_history['critic_loss'].append(critic_loss.item())
        if hasattr(self, '_last_entropy_loss') and self._last_entropy_loss is not None:
            self.training_history['entropy_loss'].append(self._last_entropy_loss.item())
            self.training_history['bc_loss'].append(self._last_bc_loss.item())
            self.training_history['advantage'].append(self._last_advantage.item())

        # Update the frozen target models
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {"step_loss": (self.training_history['actor_loss'][-1], 
                              self.training_history['critic_loss'][-1])}
    
    def get_ddpg_loss(self, observation, policy_output, reward, done_mask, next_observation, feedback=None, 
                      do_actor_update = True, do_critic_update = True):
        
        # Get current Q estimate
        current_critic_output = self.facade.apply_critic(observation, 
                                                         utils.wrap_batch(policy_output, device = self.device), 
                                                         self.critic)
        current_Q = current_critic_output['q']
        
        # Compute the target Q value
        next_policy_output = self.facade.apply_policy(next_observation, self.actor_target)
        target_critic_output = self.facade.apply_critic(next_observation, next_policy_output, self.critic_target)
        target_Q = target_critic_output['q']
        target_Q = reward + self.gamma * (done_mask * target_Q).detach()

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q, target_Q).mean()
        
        if do_critic_update and self.critic_lr > 0:
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        # ==================== Actor Loss with PG + Entropy + BC ====================
        # Get fresh policy output for current observation
        cur_policy_output = self.facade.apply_policy(observation, self.actor)
        
        # ---- Advantage (Eq. 21) ----
        with torch.no_grad():
            V_s = current_Q.detach().mean()
            advantage = torch.clamp(target_Q - V_s, -1, 1).mean()
        
        # ---- Policy Gradient Loss (Eq. 21-22) ----
        # For SASRec/OneStagePolicy: use action_prob over candidates
        if 'action_prob' in cur_policy_output:
            # Simple policy: log π(a|s) - use the sampled actions
            action_indices = policy_output.get('action', None)
            if action_indices is not None:
                # Clamp action indices to valid range [0, L)
                action_prob = cur_policy_output['action_prob']  # (B, L)
                L = action_prob.shape[1]
                
                # Handle action indices: convert to 0-indexed and clamp
                action_indices = action_indices.long()
                if action_indices.min() < 0 or action_indices.max() >= L:
                    # Invalid indices: fallback to simple negative Q
                    critic_output = self.facade.apply_critic(observation, cur_policy_output, self.critic)
                    pg_loss = -critic_output['q'].mean()
                else:
                    # Get log prob of the selected actions - detach to save memory
                    batch_idx = torch.arange(action_prob.shape[0], device=action_prob.device).unsqueeze(1)
                    slate_idx = action_indices.long()  # (B, K)
                    selected_prob = action_prob[batch_idx, slate_idx].detach()  # (B, K) - detach!
                    action_log_prob = torch.log(selected_prob + 1e-12).mean()
                    pg_loss = -advantage * action_log_prob
            else:
                # Fallback: use negative Q
                critic_output = self.facade.apply_critic(observation, cur_policy_output, self.critic)
                pg_loss = -critic_output['q'].mean()
        elif 'sid_logits' in cur_policy_output:
            # SID policy: token-level NLL (Eq. 22)
            sid_tokens = policy_output['sid_tokens']  # (B, K, L) from stored output
            sid_logits_list = cur_policy_output['sid_logits']  # list[(B, V_l)]
            B, K, L = sid_tokens.shape
            
            # Flatten for computation
            sid_tokens_flat = sid_tokens.view(B * K, L)
            level_probs = [torch.softmax(logits_l, dim=-1) for logits_l in sid_logits_list]
            level_probs_flat = [p.repeat_interleave(K, dim=0) for p in level_probs]
            
            # NLL across tokens (Eq. 22)
            nll_slot = 0.0
            for l in range(L):
                probs_l_flat = level_probs_flat[l]
                z_l = sid_tokens_flat[:, l].view(-1, 1)
                logp_l = torch.log(torch.gather(probs_l_flat, 1, z_l) + 1e-12).squeeze(1)
                nll_slot = nll_slot + (-logp_l)
            
            nll_per_sample = nll_slot.view(B, K).mean(dim=1)
            pg_loss = (nll_per_sample * advantage).mean()
        else:
            # Fallback: use negative Q as before
            critic_output = self.facade.apply_critic(observation, cur_policy_output, self.critic)
            pg_loss = -critic_output['q'].mean()
        
        # ---- Entropy Regularization (Eq. 23) ----
        entropy_loss = torch.tensor(0.0, device=self.device)
        if 'sid_logits' in cur_policy_output:
            # Token-level entropy
            for logits_l in cur_policy_output['sid_logits']:
                probs = torch.softmax(logits_l, dim=-1)
                entropy_loss = entropy_loss + -(probs * torch.log(probs + 1e-12)).sum(dim=-1).mean()
            entropy_loss = entropy_loss / len(cur_policy_output['sid_logits'])
        elif 'action_prob' in cur_policy_output:
            # Simple entropy: -Σ p log p over candidate items
            probs = cur_policy_output['action_prob']  # (B, L)
            entropy_loss = -(probs * torch.log(probs + 1e-12)).sum(dim=-1).mean()
        
        # ---- Behavioral Cloning Loss (Eq. 24) ----
        bc_loss = torch.tensor(0.0, device=self.device)
        if feedback is not None:
            # feedback: (B, K) binary feedback for selected items
            if isinstance(feedback, np.ndarray):
                feedback = torch.FloatTensor(feedback).to(self.device)
            else:
                feedback = feedback.float().to(self.device)
            
            if 'action_prob' in cur_policy_output:
                action_prob = cur_policy_output['action_prob']  # (B, L)
                action_indices = policy_output.get('action', None)
                L = action_prob.shape[1]
                
                if action_indices is not None:
                    action_indices = action_indices.long()
                    # Only compute BC if indices are valid
                    if action_indices.min() >= 0 and action_indices.max() < L:
                        # Get prob of selected actions - detach to save memory
                        batch_idx = torch.arange(action_prob.shape[0], device=action_prob.device).unsqueeze(1)
                        slate_idx = action_indices.long()  # (B, K)
                        selected_prob = action_prob[batch_idx, slate_idx].detach()  # (B, K) - detach!
                        
                        # BC loss: -Σ y * log π(a) (Eq. 24)
                        epsilon = 1e-6
                        bc_loss = -(feedback * torch.log(selected_prob + epsilon)).sum(dim=-1).mean()
                        # Normalize by positive feedback ratio
                        pos_ratio = (feedback > 0).float().mean() + epsilon
                        bc_loss = bc_loss / pos_ratio
        
        # Total Actor Loss (Eq. 25)
        actor_loss = pg_loss - self.entropy_coef * entropy_loss + self.bc_coef * bc_loss
        
        if do_actor_update and self.actor_lr > 0:
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        
        # Store for logging
        self._last_pg_loss = pg_loss.detach()
        self._last_entropy_loss = entropy_loss.detach()
        self._last_bc_loss = bc_loss.detach()
        self._last_advantage = advantage.detach()
        
        return critic_loss, actor_loss


    def save(self):
        torch.save(self.critic.state_dict(), self.save_path + "_critic")
        torch.save(self.critic_optimizer.state_dict(), self.save_path + "_critic_optimizer")

        torch.save(self.actor.state_dict(), self.save_path + "_actor")
        torch.save(self.actor_optimizer.state_dict(), self.save_path + "_actor_optimizer")


    def load(self):
        self.critic.load_state_dict(torch.load(self.save_path + "_critic", map_location=self.device))
        self.critic_optimizer.load_state_dict(torch.load(self.save_path + "_critic_optimizer", map_location=self.device))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(self.save_path + "_actor", map_location=self.device))
        self.actor_optimizer.load_state_dict(torch.load(self.save_path + "_actor_optimizer", map_location=self.device))
        self.actor_target = copy.deepcopy(self.actor)

#     def log_iteration(self, step):
#         episode_report, train_report = self.get_report()
#         log_str = f"step: {step} @ episode report: {episode_report} @ step loss: {train_report['step_loss']}\n"
# #         if train_report:
# #             log_str = f"step: {step} @ episode report: {episode_report} @ step loss (actor,critic): {train_report['step_loss']}\n"
# #         else:
# #             log_str = f"step: {step} @ episode report: {episode_report}\n"
#         with open(self.save_path + ".report", 'a') as outfile:
#             outfile.write(log_str)
#         return log_str

#     def test(self):
#         t = time.time()
#         # Testing
#         self.facade.initialize_train()
#         self.load()
#         print("Testing:")
#         # for i in tqdm(range(self.n_iter)):
#         with torch.no_grad():
#             for i in tqdm(range(len(self.facade.env.reader) // self.batch_size)):
#                 pick_rows = [row for row in range(i * self.batch_size, (i + 1) * self.batch_size)]
#                 episode_report, _ = self.run_an_episode(self.exploration_scheduler.value(i), pick_rows = pick_rows)

#                 if (i+1) % 1 == 0:
#                     t_ = time.time()
#                     print(f"Episode {i+1}, time diff {t_ - t})")
#                     print(self.log_iteration(i, episode_report))
#                     t = t_