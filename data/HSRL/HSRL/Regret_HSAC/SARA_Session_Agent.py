"""
SARA Session Agent - 阶段1：Session 级别训练
核心改动：
1. 不走 Replay Buffer，直接收集完整 Session 轨迹
2. 保留 A2C Loss 计算逻辑
3. 为后续 RRCA（回溯修正 Advantage）做准备
"""

import copy
import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from model.agents.BaseRLAgent import BaseRLAgent


class SARA_Session_Agent(BaseRLAgent):
    """
    Session 级别的 SARA Agent
    
    核心区别于 A2C_SID_credit：
    - 不使用 Replay Buffer
    - 每次 step_train() 收集完整 Session（Episode）
    - 在完整轨迹上计算 Advantage
    """
    
    @staticmethod
    def parse_model_args(parser):
        parser = BaseRLAgent.parse_model_args(parser)
        parser.add_argument('--episode_batch_size', type=int, default=8, 
                            help='episode sample batch size')
        parser.add_argument('--session_max_steps', type=int, default=50, 
                            help='max steps per session (L)')
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
        parser.add_argument('--advantage_bias', type=float, default=0, 
                            help='mitigation factor')
        parser.add_argument('--entropy_coef', type=float, default=0.1, 
                            help='mitigation factor')
        parser.add_argument('--token_lr', type=float, default=0, 
                            help='mitigation factor')
        parser.add_argument('--behavior_lr', type=float, default=0.0001, 
                            help='behaviorvise loss coefficient')
        parser.add_argument('--behavior_decay', type=float, default=0.00003, 
                            help='behaviorvise loss coefficient')
        
        # RRCA 参数 (使用 rrca_ 前缀避免冲突)
        parser.add_argument('--rrca_lambda_unlike', type=float, default=1.0, 
                            help='RRCA unlike 惩罚权重')
        parser.add_argument('--rrca_lambda_undislike', type=float, default=0.5, 
                            help='RRCA unlike 奖励权重')
        parser.add_argument('--rrca_regret_gamma', type=float, default=0.9, 
                            help='RRCA 后悔折扣因子')
        
        # RAPI 参数
        parser.add_argument('--regret_pool_size', type=int, default=10, 
                            help='后悔池容量 N')
        parser.add_argument('--regret_eta', type=float, default=1.0, 
                            help='RAPI 惩罚系数 η')
        parser.add_argument('--regret_W', type=str, default='0.05,0.25,0.7', 
                            help='层参数 W (逗号分隔)')
        
        return parser

    def __init__(self, args, facade):
        super().__init__(args, facade)
        self.episode_batch_size = args.episode_batch_size
        self.session_max_steps = args.session_max_steps
        self.batch_size = args.episode_batch_size  # 训练时 batch = episode batch

        self.actor = facade.actor
        self.critic = facade.critic
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(),  lr=args.actor_lr,  weight_decay=args.actor_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr, weight_decay=args.critic_decay)

        self.tau = args.target_mitigate_coef
        self.advantage_bias = args.advantage_bias
        self.entropy_coef = args.entropy_coef

        # token weight for token-level value
        self.token_weight = nn.Parameter(torch.ones(3+1, device=self.device) / (3+1))
        self.token_optimizer = torch.optim.Adam([self.token_weight], lr=args.token_lr)
        
        if len(self.n_iter) == 1:
            with open(self.save_path + ".report", 'w') as f:
                f.write(f"{args}\n")
        
        self.behavior_lr = args.behavior_lr
        self.behavior_decay = args.behavior_decay
        self.actor_behavior_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.behavior_lr, 
                                                         weight_decay=args.behavior_decay)
        
        # RRCA 参数
        self.lambda_unlike = args.rrca_lambda_unlike
        self.lambda_undislike = args.rrca_lambda_undislike
        self.regret_gamma = args.rrca_regret_gamma
        
        # RAPI 参数
        self.regret_pool_size = args.regret_pool_size  # N
        self.regret_eta = args.regret_eta  # 惩罚系数 η
        # 解析 W_layer 参数，如 "0.05,0.25,0.7" → [0.05, 0.25, 0.7]
        self.regret_W = torch.tensor([float(x) for x in args.regret_W.split(',')], 
                                      device=self.device)
        
        # 每个用户（batch中的每个索引）的独立后悔池
        # shape: (episode_batch_size, regret_pool_size, L+1)
        # 前 L 列是 sid_tokens (L,)，最后一列是 phi 值
        # -1 表示空槽
        self.L = facade.actor.l  # HAC 层数
        self.user_regret_pool = torch.full(
            (self.episode_batch_size, self.regret_pool_size, self.L + 1), 
            -1, dtype=torch.long, device=self.device
        )
        
        # 检查 actor 是否支持 get_sara_logits（用于 RAPI 拦截）
        self.use_rapi = hasattr(facade.actor, 'get_sara_logits')

    def action_before_train(self):
        super().action_before_train()
        self.training_history['entropy_loss'] = []
        self.training_history['advantage'] = []
        self.training_history["behavior_loss"] = []
        self.training_history['regret_events'] = []  # RRCA: 记录每次 regret 事件数量

    @torch.no_grad()
    def run_episode_step(self, *episode_args):
        """
        单步交互（保留接口兼容，但本 Agent 主要用 collect_session）
        """
        episode_iter, epsilon, observation, do_buffer_update = episode_args
        policy_output = self.facade.apply_policy(observation, self.actor, epsilon, do_explore=True)
        next_observation, reward, done, info = self.facade.env_step(policy_output)
        return next_observation

    def collect_session(self, max_steps=None):
        """
        收集一个完整 Session（Episode）的轨迹
        
        Returns:
            trajectory: list of dict, 每个 dict 包含:
                - observation
                - policy_output  
                - reward
                - done
                - next_observation
                - info (包含 sara_delta_t, sara_unlike_prob)
        """
        if max_steps is None:
            max_steps = self.session_max_steps
            
        # 1. 重置环境，得到 batch_size 个用户的初始状态
        observation = self.facade.reset_env({"batch_size": self.episode_batch_size})
        
        # 2. 手动收集完整 Session（不走 buffer）
        trajectory = []
        for step in range(max_steps):
            # 在推理前，同步 Agent 的后悔池到 Facade（用于 RAPI 拦截）
            # 只有当 actor 支持 get_sara_logits 时才同步
            if hasattr(self, 'use_rapi') and self.use_rapi:
                self._sync_regret_pool_to_facade()
            
            # Policy forward
            policy_output = self.facade.apply_policy(
                observation, self.actor, 
                epsilon=0.0,  # 训练时不需要 exploration
                do_explore=False
            )
            
            # Env step
            next_observation, reward, done, info = self.facade.env_step(policy_output)
            
            # 存储 transition - 缓存 policy_output 避免后续重复计算
            trajectory.append({
                'observation': observation,
                'policy_output': policy_output,  # 直接缓存
                'reward': reward,
                'done': done,
                'next_observation': next_observation,
                'info': info  # 包含 sara_delta_t, sara_unlike_prob
            })
            
            # 更新观察
            observation = next_observation
            
            # 如果所有用户都 done，提前结束
            if isinstance(done, torch.Tensor):
                if done.all().item():
                    break
            elif done.all():
                break
        
        return trajectory
    
    def _simulate_regret_explosion(self, trajectory):
        """
        模拟延迟后悔的引爆（RRCA 核心）
        
        对于每个时刻 t 埋下的 "定时炸弹" sara_delta_t[t]:
        - 如果 t + delta_t 时刻仍在 session 内，则检查是否引爆
        - 实际引爆概率 = sara_unlike_prob
        
        Returns:
            regret_events: list of dict, 每个 dict 包含:
                - t: 埋雷时刻
                - t_explode: 引爆时刻  
                - batch_idx: batch 中的用户索引
                - item_idx: 推荐物品索引
                - psi: 后悔强度（固定值）
                - sid_tokens: 推荐的语义 token（用于后悔池）
        """
        T = len(trajectory)
        B = self.episode_batch_size
        
        regret_events = []
        
        for t in range(T):
            info = trajectory[t]['info']
            
            # 获取 sara_delta_t 和 sara_unlike_prob
            sara_delta_t = info.get('sara_delta_t', None)
            sara_unlike_prob = info.get('sara_unlike_prob', None)
            sid_tokens = info.get('sid_tokens', None)  # 推荐物品的语义 token
            
            if sara_delta_t is None or sara_unlike_prob is None:
                continue
            
            # sara_delta_t shape: (B, K), K 是 slate size
            B_cur, K = sara_delta_t.shape
            
            for b in range(B_cur):
                for k in range(K):
                    delta_t = sara_delta_t[b, k].item()
                    unlike_prob = sara_unlike_prob[b, k].item()
                    
                    if delta_t == 0:
                        continue
                    
                    t_explode = t + delta_t
                    
                    # 检查是否在 session 内
                    if t_explode >= T:
                        continue
                    
                    # 模拟引爆（固定概率）
                    if torch.rand(1).item() < unlike_prob:
                        # 获取该次推荐对应的 sid_tokens
                        if sid_tokens is not None:
                            item_sid_tokens = sid_tokens[b, k, :]  # (L,)
                        else:
                            item_sid_tokens = None
                        
                        # 引爆！记录后悔事件
                        regret_events.append({
                            't': t,
                            't_explode': t_explode,
                            'batch_idx': b,
                            'item_idx': k,
                            'psi': -self.lambda_unlike,  # 固定值：unlike 惩罚
                            'sid_tokens': item_sid_tokens,
                            'phi': -self.lambda_unlike * (self.regret_gamma ** delta_t)  # 折扣后的惩罚值
                        })
        
        return regret_events
    
    def _apply_rrca_advantage(self, advantage, regret_events, T, B):
        """
        应用回溯遗憾修正到 Advantage
        
        A_t^RRCA = A_t + Σ[psi * γ^(t_explode - t)]
        
        Args:
            advantage: (T, B) 原始 advantage
            regret_events: list of regret events
            T: trajectory 长度
            B: batch size
            
        Returns:
            advantage_rrca: (T, B) 修正后的 advantage
        """
        advantage_rrca = advantage.clone()
        
        for event in regret_events:
            t = event['t']
            t_explode = event['t_explode']
            b = event['batch_idx']
            psi = event['psi']
            
            # 计算折扣因子
            gamma_pow = self.regret_gamma ** (t_explode - t)
            
            # 回溯修正
            advantage_rrca[t, b] += psi * gamma_pow
        
        return advantage_rrca
    
    def _update_user_regret_pool(self, regret_events):
        """
        将引爆的后悔事件加入对应用户的独立后悔池
        
        Args:
            regret_events: list of regret events (from _simulate_regret_explosion)
        """
        for event in regret_events:
            sid_tokens = event.get('sid_tokens', None)
            phi = event.get('phi', None)
            batch_idx = event.get('batch_idx', 0)
            
            if sid_tokens is None or phi is None:
                continue
            
            # 找到该用户的空槽（值为 -1）
            pool_b = self.user_regret_pool[batch_idx]  # (N, L+1)
            empty_slots = (pool_b[:, -1] == -1).nonzero(as_tuple=True)[0]
            
            if len(empty_slots) > 0:
                # 写入第一个空槽
                slot_idx = empty_slots[0]
                # 前 L 列是 sid_tokens，最后一列是 phi
                self.user_regret_pool[batch_idx, slot_idx, :self.L] = sid_tokens.detach()
                self.user_regret_pool[batch_idx, slot_idx, self.L] = int(phi * 100)  # 存为整数
            else:
                # 池已满，FIFO 替换
                slot_idx = 0  # 简单替换第一个
                self.user_regret_pool[batch_idx, slot_idx, :self.L] = sid_tokens.detach()
                self.user_regret_pool[batch_idx, slot_idx, self.L] = int(phi * 100)
    
    def _sync_regret_pool_to_facade(self):
        """
        将每个用户的独立后悔池同步到 Facade（用于 RAPI 推理拦截）
        
        Facade 的 apply_policy 会调用 policy.get_sara_logits，后者会根据这个池计算惩罚
        """
        if not hasattr(self, 'use_rapi') or not self.use_rapi:
            return
        
        # user_regret_pool: (B, N, L+1)
        # 需要转为: {'tokens': [tensor(N, L), ...], 'phis': [tensor(N,), ...]}
        # 但这是每个用户独立的，所以需要按用户分别处理
        
        # 取所有非空槽
        B = self.episode_batch_size
        pool_list = []
        phi_list = []
        
        for b in range(B):
            pool_b = self.user_regret_pool[b]  # (N, L+1)
            # 找到非空槽（最后一列 != -1）
            valid_mask = pool_b[:, -1] != -1
            if valid_mask.sum() > 0:
                tokens = pool_b[valid_mask, :self.L]  # (M, L)
                phis = pool_b[valid_mask, self.L].float() / 100  # 转回 float
                pool_list.append(tokens)
                phi_list.append(phis)
        
        if len(pool_list) > 0:
            # 合并所有用户的池（跨用户也可以共享，因为是全局黑名单）
            all_tokens = torch.cat(pool_list, dim=0)
            all_phis = torch.cat(phi_list, dim=0)
            
            self.facade.global_sara_pool = {
                'tokens': [all_tokens],
                'phis': [all_phis]
            }
        else:
            self.facade.global_sara_pool = {'tokens': [], 'phis': []}

    def step_train(self):
        """
        Session 级别的训练：
        1. 收集完整 Session 轨迹
        2. 在轨迹上计算 A2C Loss（含 RRCA 回溯修正）
        """
        # 收集完整 Session
        trajectory = self.collect_session()
        
        # 计算 Loss（包含 RRCA 修正）
        critic_loss, actor_loss, entropy_loss, advantage, n_regret = self.get_session_a2c_loss(trajectory)
        
        self.training_history['actor_loss'].append(actor_loss.item())
        self.training_history['critic_loss'].append(critic_loss.item())
        self.training_history['entropy_loss'].append(entropy_loss.item())
        self.training_history['advantage'].append(advantage.item())
        self.training_history['regret_events'].append(n_regret)

        # soft update targets
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
            for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return {
            "step_loss": (
                self.training_history['actor_loss'][-1],
                self.training_history['critic_loss'][-1],
                self.training_history['entropy_loss'][-1],
                self.training_history['advantage'][-1],
                0.0  # behavior_loss placeholder
            )
        }

    def get_session_a2c_loss(self, trajectory):
        """
        在完整 Session 轨迹上计算 A2C Loss（含 RRCA 回溯修正）
        
        Args:
            trajectory: list of dict, 每个 dict 包含:
                - observation
                - policy_output  
                - reward
                - done
                - next_observation
                - info (包含 sara_delta_t, sara_unlike_prob)
        """
        T = len(trajectory)
        B = self.episode_batch_size
        
        all_rewards = torch.stack([t['reward'] for t in trajectory])  # (T, B)
        all_dones = torch.stack([t['done'] for t in trajectory]).float()  # (T, B)
        
        # ---- 直接从 trajectory 提取 context，不重新 forward ----
        all_context = torch.stack([t['policy_output']['context_list'] for t in trajectory])  # (T, B, D)
        
        # Critic: 计算 V(s) for each timestep
        V_s_list = []
        for t in range(T):
            V_out = self.critic({'context_list': all_context[t]})
            V_s_seq = V_out['v_seq']  # (B, L+1)
            token_weight = torch.softmax(self.token_weight, dim=0).unsqueeze(0).expand(B, -1)
            V_s_weighted = (V_s_seq * token_weight).sum(dim=1)  # (B,)
            V_s_list.append(V_s_weighted)
        
        V_s_all = torch.stack(V_s_list)  # (T, B)
        
        # ---- TD Target: Q_s = r + gamma * V(s') ----
        with torch.no_grad():
            V_sp = V_s_all[-1]  # (B,)
            
            Q_s_list = []
            for t in reversed(range(T)):
                if t == T - 1:
                    target = all_rewards[t] + self.gamma * (1 - all_dones[t]) * V_sp
                else:
                    target = all_rewards[t] + self.gamma * (1 - all_dones[t]) * V_s_all[t + 1]
                Q_s_list.append(target)
            
            Q_s_all = torch.stack(list(reversed(Q_s_list)))  # (T, B)
            
            # 原始 Advantage
            advantage = Q_s_all - V_s_all  # (T, B)
            advantage = torch.clamp(advantage, -1, 1)
        
        # ---- RRCA: 回溯修正 Advantage ----
        with torch.no_grad():
            # 1. 模拟引爆延迟后悔
            regret_events = self._simulate_regret_explosion(trajectory)
            
            # 2. 应用修正
            advantage_rrca = self._apply_rrca_advantage(advantage, regret_events, T, B)
            
            # 3. 断开计算图
            advantage_rrca = advantage_rrca.detach()
        
        # ---- RAPI: 更新用户独立后悔池 ----
        self._update_user_regret_pool(regret_events)
        
        # ---- Critic Loss ----
        value_loss = F.mse_loss(V_s_all, Q_s_all)  # (T, B)
        value_loss = value_loss.mean()
        
        if self.critic_optimizer is not None:
            self.critic_optimizer.zero_grad()
            value_loss.backward()  # 安全：计算图在这里终结
            self.critic_optimizer.step()
        
        # ---- Actor Loss ----
        actor_loss = 0.0
        entropy_raw = 0.0
        valid_steps = 0.0
        
        for t in range(T):
            # 添加 mask，过滤掉已经 done 的 batch item
            mask_t = 1.0 - all_dones[t]
            if mask_t.sum() == 0:
                continue
            
            po = trajectory[t]['policy_output']
            sid_tokens = po['sid_tokens']  # (B, K, L)
            sid_logits_list = po['sid_logits']
            B_t, K, L = sid_tokens.shape
            
            # flatten
            sid_tokens_flat = sid_tokens.view(B_t * K, L)
            level_probs = [torch.softmax(logits_l, dim=-1) for logits_l in sid_logits_list]
            level_probs_flat = [p.repeat_interleave(K, dim=0) for p in level_probs]
            
            # NLL
            nll_slot = 0.0
            for l in range(L):
                probs_l_flat = level_probs_flat[l]
                z_l = sid_tokens_flat[:, l].view(-1, 1)
                logp_l = torch.log(torch.gather(probs_l_flat, 1, z_l) + 1e-12).squeeze(1)
                nll_slot = nll_slot + (-logp_l)
            
            nll_per_sample = nll_slot.view(B_t, K).mean(dim=1)  # (B,)
            
            # 使用 RRCA 修正后的 advantage
            actor_loss_t = nll_per_sample * (advantage_rrca[t] + self.advantage_bias)
            actor_loss_t = (actor_loss_t * mask_t).sum() / (mask_t.sum() + 1e-8)
            actor_loss += actor_loss_t
            valid_steps += 1.0
            
            # Entropy
            for l in range(L):
                entropy_t = (level_probs[l] * torch.log(level_probs[l] + 1e-12)).sum(dim=-1).mean()
                entropy_raw += entropy_t
        
        # 平均
        actor_loss = actor_loss / (valid_steps + 1e-8)
        entropy_raw = entropy_raw / (T + 1e-8)
        
        total_actor = actor_loss + self.entropy_coef * entropy_raw
        
        if self.actor_optimizer is not None:
            self.actor_optimizer.zero_grad()
            total_actor.backward()  # 安全：使用了提前缓存的 logits，无冲突
            self.actor_optimizer.step()
        
        # 记录 regret 事件数量用于监控
        n_regret_events = len(regret_events) if 'regret_events' in dir() else 0
        
        return value_loss.detach(), actor_loss.detach(), entropy_raw.detach(), advantage_rrca.mean().detach(), n_regret_events

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

    def test(self):
        """
        评测模式
        """
        import torch
        from time import time
        from tqdm import tqdm

        # 加载权重
        try:
            self.load()
            print("[TEST] Loaded trained weights from save_path prefix.")
        except Exception as e:
            print(f"[TEST] WARN: load() failed or weights not found, continue. Detail: {e}")

        # eval + 关梯度
        self.actor.eval()
        if hasattr(self, "critic") and self.critic is not None:
            self.critic.eval()
        torch.set_grad_enabled(False)

        # 环境 reset
        print("[TEST] Reset environment for evaluation.")
        observation = self.facade.reset_env({"batch_size": self.episode_batch_size})

        total_steps = self.n_iter[-1] if isinstance(self.n_iter, (list, tuple)) else int(self.n_iter)
        start_time = time()
        
        print("[TEST] Start evaluation (collect sessions without training).")
        for i in tqdm(range(total_steps)):
            # 只收集 session，不训练
            trajectory = self.collect_session()
            
            if i % self.check_episode == 0:
                # 统计 reward
                total_reward = sum([t['reward'].mean().item() for t in trajectory])
                print(f"[TEST] step {i} | avg_reward={total_reward/len(trajectory):.4f}")

        torch.set_grad_enabled(True)
        print("[TEST] Evaluation finished.")
