"""
SARA Facade - 终极融合版
核心改动：
1. 完美保留显式存储 history_features 的安全设计（防 Padding 越界）
2. 彻底移除引发内存爆炸的 RegretPoolManager 和 user_ids
3. 启用全局黑名单 (global_sara_pool) 配合张量推断拦截 (get_sara_logits)
4. 实现时光倒流更新 advantage
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import utils
import pickle

class SARAFacade_credit():
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--slate_size', type=int, default=6, help='slate size for actions')
        parser.add_argument('--buffer_size', type=int, default=10000, help='replay buffer size')
        parser.add_argument('--start_timestamp', type=int, default=1000, help='start timestamp for buffer sampling')
        parser.add_argument('--noise_var', type=float, default=0, help='noise magnitude for action embedding sampling')
        parser.add_argument('--q_laplace_smoothness', type=float, default=0.5, help='critic smoothness scalar for actors')
        parser.add_argument('--topk_rate', type=float, default=1., help='rate choosing topk rather than categorical sampling for items')
        parser.add_argument('--empty_start_rate', type=float, default=0, help='probability of starting an episode from empty history')
        parser.add_argument('--item2sid', type=str, default='dataset/sid_index_item2sid.pkl', help='probability of starting an episode from empty history')
        parser.add_argument('--regret_pool_size', type=int, default=1000, help='Max size of global SARA pool')
        return parser

    def __init__(self, args, environment, actor, critic):
        super().__init__()
        self.device = args.device
        self.env = environment
        self.actor = actor
        self.critic = critic

        self.slate_size = args.slate_size
        self.noise_var = args.noise_var
        self.noise_decay = args.noise_var / args.n_iter[-1] if hasattr(args, 'n_iter') else 0
        self.q_laplace_smoothness = args.q_laplace_smoothness
        self.topk_rate = args.topk_rate
        self.empty_start_rate = args.empty_start_rate
        self.regret_pool_size = getattr(args, 'regret_pool_size', 1000)

        print(f"Note!! item2sid path is {args.item2sid}")
        if args.item2sid is not None and os.path.exists(args.item2sid):
            with open(args.item2sid, "rb") as f:
                self.item2sid = pickle.load(f)
        else:
            self.item2sid = None

        self.n_item = self.env.action_space['item_id'][1]
        self.candidate_iids = np.arange(1, self.n_item + 1)
        self.candidate_features = torch.FloatTensor(self.env.reader.get_item_list_meta(self.candidate_iids)).to(self.device)
        self.candidate_iids = torch.tensor(self.candidate_iids).to(self.device)

        # ========== SARA: 启用极简全局黑名单 ==========
        self.global_sara_pool = {'tokens': [], 'phis': []}

        self.buffer_size = args.buffer_size
        self.start_timestamp = args.start_timestamp

    def initialize_train(self):
        # 获取 item_dim 用于显式分配 history_features 的空间
        stats = self.env.reader.get_statistics()
        item_dim = stats.get('item_vec_size', 16)
        max_seq_len = stats.get('max_seq_len', 50)
        
        self.buffer = {
            "user_profile": torch.zeros(self.buffer_size, self.env.reader.portrait_len),
            "history": torch.zeros(self.buffer_size, max_seq_len).to(torch.long),
            "history_features": torch.zeros(self.buffer_size, max_seq_len, item_dim),
            "next_history": torch.zeros(self.buffer_size, max_seq_len).to(torch.long),
            "next_history_features": torch.zeros(self.buffer_size, max_seq_len, item_dim),
            "context_list": torch.zeros(
                self.buffer_size,
                self.actor.sid_levels + 1,
                self.actor.state_dim,
                dtype=torch.float32
            ),
            "action": torch.zeros(self.buffer_size, self.slate_size, dtype=torch.long),
            "reward": torch.zeros(self.buffer_size),
            "feedback": torch.zeros(self.buffer_size, self.slate_size),
            "done": torch.zeros(self.buffer_size, dtype=torch.bool),
            "sid_tokens": torch.zeros(
                self.buffer_size,
                self.slate_size,
                self.actor.sid_levels,
                dtype=torch.long
            )
        }
        for k, v in self.buffer.items():
            self.buffer[k] = v.to(self.device)
        self.buffer_head = 0
        self.current_buffer_size = 0
        self.n_stream_record = 0
        self.is_training_available = False

    def reset_env(self, initial_params={"batch_size": 1}):
        initial_params['empty_history'] = True if np.random.rand() < self.empty_start_rate else False
        return self.env.reset(initial_params)

    def env_step(self, policy_output):
        action_dict = {'action': policy_output['action'], 'action_features': policy_output['action_features']}
        observation, reward, done, info = self.env.step(action_dict)
        return observation, reward, done, info

    def update_buffer(self, observation, policy_output, reward, done_mask, next_observation, info):
        """更新 replay buffer，返回存储的索引供 SARA 埋雷使用"""
        stored_indices = self.store_transition(
            observation, policy_output, reward, done_mask, info, next_observation
        )
        return stored_indices

    def stop_env(self):
        self.env.stop()

    def get_episode_report(self, n_recent=10):
        recent_rewards = self.env.reward_history[-n_recent:]
        recent_steps = self.env.step_history[-n_recent:]
        if not recent_rewards:
            recent_rewards = [0.0]
        if not recent_steps:
            recent_steps = [0.0]
        episode_report = {
            'average_total_reward': np.mean(recent_rewards),
            'reward_variance': np.var(recent_rewards),
            'max_total_reward': np.max(recent_rewards),
            'min_total_reward': np.min(recent_rewards),
            'average_n_step': np.mean(recent_steps),
            'max_n_step': np.max(recent_steps),
            'min_n_step': np.min(recent_steps),
            'buffer_size': self.current_buffer_size,
        }
        return episode_report

    def apply_policy(self, observation, policy_model, epsilon=0.0, do_explore=False, do_softmax=True):
        """
        SARA 版策略执行：直接将全局后悔池传入 Policy，进行极速张量拦截
        """
        feed_dict = observation
        
        # 1. 提取全局后悔池转为 Tensor
        pool_tokens_tensor = None
        pool_phis_tensor = None
        if len(self.global_sara_pool['tokens']) > 0:
            pool_tokens_tensor = torch.stack(self.global_sara_pool['tokens']).to(self.device) # [N, L]
            pool_phis_tensor = torch.stack(self.global_sara_pool['phis']).to(self.device)    # [N]

        # 2. 调用 SARA 专用的带拦截前向推断
        if hasattr(policy_model, 'get_sara_logits'):
            out_dict = policy_model.get_sara_logits(feed_dict, pool_tokens_tensor, pool_phis_tensor)
        else:
            out_dict = policy_model(feed_dict)

        assert 'sid_logits' in out_dict, "SIDPolicy 必须输出 'sid_logits'"
        sid_logits_list = out_dict['sid_logits']
        B = sid_logits_list[0].size(0)
        L = len(sid_logits_list)

        if 'candidate_ids' in feed_dict:
            cand_ids = feed_dict['candidate_ids']
            if isinstance(cand_ids, torch.Tensor) and cand_ids.dim() == 1:
                cand_ids = cand_ids.unsqueeze(0).repeat(B, 1)
        else:
            cand_ids = self.candidate_iids.unsqueeze(0).repeat(B, 1)

        sid_table = getattr(self, "_sid_table", None)
        if (sid_table is None) or (sid_table.size(1) != L):
            table = torch.zeros(self.n_item + 1, L, dtype=torch.long)
            if self.item2sid is not None:
                for iid, sid in self.item2sid.items():
                    tt = tuple(sid)
                    if len(tt) >= L:
                        table[int(iid), :] = torch.tensor(tt[:L], dtype=torch.long)
            self._sid_table = table.to(self.device)
            sid_table = self._sid_table

        cand_sid = sid_table[cand_ids]
        level_probs = [torch.softmax(logits_l, dim=-1) for logits_l in sid_logits_list]
        candidate_prob = torch.ones_like(cand_ids, dtype=torch.float32)

        for l in range(L):
            idx_l = cand_sid[..., l]
            pl = level_probs[l].gather(1, idx_l)
            candidate_prob = candidate_prob * pl

        candidate_prob = candidate_prob / (candidate_prob.sum(dim=1, keepdim=True) + 1e-12)

        if do_explore and epsilon > 0:
            candidate_prob = (1 - epsilon) * candidate_prob + epsilon * (1.0 / candidate_prob.size(1))

        if np.random.rand() >= self.topk_rate:
            action, indices = utils.sample_categorical_action(
                candidate_prob, cand_ids, self.slate_size,
                with_replacement=False, batch_wise=True, return_idx=True
            )
        else:
            _, indices = torch.topk(candidate_prob, k=self.slate_size, dim=1)
            action = torch.gather(cand_ids, 1, indices).detach()

        out_dict['action'] = action
        out_dict['action_features'] = self.candidate_features[action - 1]
        out_dict['action_prob'] = torch.gather(candidate_prob, 1, indices)
        out_dict['candidate_prob'] = candidate_prob
        out_dict['sid_tokens'] = sid_table[action]

        # 提取 state_emb 和 action_emb 供 Critic 计算
        if 'context_list' in out_dict:
            out_dict['state_emb'] = out_dict['context_list'][:, 0, :]
            out_dict['action_emb'] = out_dict['context_list'][:, -1, :]

        return out_dict

    def apply_critic(self, observation, policy_output, critic_model):
        """为 Critic 计算 Q 值"""
        feed_dict = {
            "state_emb": policy_output["state_emb"],
            "action_emb": policy_output["action_emb"],
            "context_list": policy_output.get("context_list", None)
        }
        critic_output = critic_model(feed_dict)
        
        if 'v_seq' in critic_output and 'q' not in critic_output:
            critic_output['q'] = critic_output['v_seq'][:, -1]
            
        return critic_output

    def store_transition(self, observation, policy_output, reward, done, info, next_observation):
        """将安全抽取的 Features 一并存入 Buffer"""
        sid_tokens = policy_output['sid_tokens']
        context_list = policy_output['context_list']

        B = observation['history'].size(0)
        stored_indices = []
        
        for b in range(B):
            idx = self.buffer_head
            stored_indices.append(idx)
            
            self.buffer['user_profile'][idx] = observation['user_profile'][b]
            self.buffer['history'][idx] = observation['history'][b]
            # 安全直接存储特征
            self.buffer['history_features'][idx] = observation['history_features'][b].detach()
            self.buffer['next_history'][idx] = next_observation['history'][b]
            self.buffer['next_history_features'][idx] = next_observation['history_features'][b].detach()
            
            self.buffer['context_list'][idx] = context_list[b].detach()
            self.buffer['action'][idx] = policy_output['action'][b]
            self.buffer['reward'][idx] = reward[b]
            self.buffer['done'][idx] = done[b]
            self.buffer['sid_tokens'][idx] = sid_tokens[b].detach()

            if 'response' in info:
                self.buffer['feedback'][idx] = info['response'][b]
            else:
                self.buffer['feedback'][idx] = torch.zeros(self.slate_size)

            self.buffer_head = (self.buffer_head + 1) % self.buffer_size
            self.n_stream_record += 1

        self.current_buffer_size = min(self.n_stream_record, self.buffer_size)
        if self.n_stream_record >= self.start_timestamp:
            self.is_training_available = True
        
        return stored_indices

    def override_reward(self, buffer_idx, new_reward, sid_tokens=None, phi=None):
        """时光倒流：覆写历史奖励，并把失败路径打入全局冷宫"""
        self.buffer['reward'][buffer_idx] = new_reward
        
        if sid_tokens is not None and phi is not None:
            self.global_sara_pool['tokens'].append(sid_tokens.detach().cpu())
            self.global_sara_pool['phis'].append(phi.detach().cpu())
            
            # 维持滑动窗口大小
            if len(self.global_sara_pool['tokens']) > self.regret_pool_size:
                self.global_sara_pool['tokens'].pop(0)
                self.global_sara_pool['phis'].pop(0)

    def read_buffer(self, indices):
        U = self.buffer['user_profile'][indices]
        H = self.buffer['history'][indices]
        HF = self.buffer['history_features'][indices]
        N = self.buffer['next_history'][indices]
        NF = self.buffer['next_history_features'][indices]
        CON = self.buffer['context_list'][indices]
        SID = self.buffer['sid_tokens'][indices]
        A = self.buffer['action'][indices]
        R = self.buffer['reward'][indices]
        F = self.buffer['feedback'][indices]
        D = self.buffer['done'][indices]
        return U, H, HF, N, NF, CON, SID, A, R, F, D

    def sample_buffer(self, batch_size):
        indices = np.random.randint(0, self.current_buffer_size, size=batch_size)
        U, H, HF, N, NF, CON, SID, A, R, F, D = self.read_buffer(indices)
        
        observation = {"user_profile": U, "history": H, "history_features": HF}
        policy_output = {
            "context_list": CON, 
            "action": A, 
            "sid_tokens": SID,
            "state_emb": CON[:, 0, :], 
            "action_emb": CON[:, -1, :]
        }
        next_observation = {"user_profile": U, "history": N, "history_features": NF, "previous_feedback": F}

        return observation, policy_output, R, D, next_observation

    def extract_behavior_data(self, observation, policy_output, next_observation):
        observation_dict = {
            "user_profile": observation["user_profile"], 
            "history_features": observation["history_features"]
        }
        exposed_items = policy_output["action"]
        exposure = {
            "ids": exposed_items, 
            "features": self.candidate_features[exposed_items-1]
        }
        user_feedback = next_observation["previous_feedback"]
        return observation_dict, exposure, user_feedback