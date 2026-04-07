"""
RRCA + RAPI 最小实现测试
不依赖外部环境，完全模拟数据流程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

# ==========================================
# 1. 最小数据集模拟
# ==========================================
class MiniDataset:
    """模拟 Yambda 数据集"""
    def __init__(self, n_users=100, n_items=1000, seq_len=20):
        self.n_users = n_users
        self.n_items = n_items
        self.seq_len = seq_len
        
        # 模拟用户历史
        self.user_history = {}
        for u in range(n_users):
            self.user_history[u] = list(np.random.randint(0, n_items, seq_len))
    
    def sample_users(self, batch_size):
        """采样 batch_size 个用户"""
        user_ids = np.random.randint(0, self.n_users, batch_size)
        histories = [self.user_history[u] for u in user_ids]
        return user_ids, histories

# ==========================================
# 2. 最小环境模拟
# ==========================================
class MiniEnvironment:
    """模拟 SARAEnvironment - 输出 sara_delta_t 和 sara_unlike_prob"""
    def __init__(self, dataset):
        self.dataset = dataset
        self.current_users = None
        self.step_count = 0
        
    def reset(self, batch_size):
        """重置环境，返回用户历史"""
        self.current_users, histories = self.dataset.sample_users(batch_size)
        self.step_count = 0
        return histories
    
    def step(self, actions):
        """
        模拟环境 step
        actions: (B, slate_size) 物品ID
        返回: reward, done, info
        """
        B, slate_size = actions.shape
        self.step_count += 1
        
        # 模拟用户反馈
        # 随机生成点击/不喜欢
        response = torch.rand(B, slate_size) > 0.7  # 30% 点击率
        
        # 计算奖励
        reward = response.float().mean(dim=-1)
        
        # 模拟 done (随机结束)
        done = torch.rand(B) < (self.step_count / 20.0)
        if self.step_count >= 20:
            done = torch.ones(B, dtype=torch.bool)
        
        # 核心：生成 sara_delta_t 和 sara_unlike_prob
        # sara_delta_t: 延迟步数 (0-5)
        sara_delta_t = torch.randint(0, 6, (B, slate_size))
        # sara_unlike_prob: 触发概率
        sara_unlike_prob = torch.rand(B, slate_size) * 0.3
        
        info = {
            'response': response,
            'sara_delta_t': sara_delta_t,
            'sara_unlike_prob': sara_unlike_prob,
            'actions': actions  # 记录推荐的物品
        }
        
        return reward, done, info

# ==========================================
# 3. 最小 Actor (简化版 SIDPolicy)
# ==========================================
class MiniActor(nn.Module):
    def __init__(self, n_items=1000, embed_dim=32, vocab_size=16, n_levels=3, slate_size=3):
        super().__init__()
        self.n_items = n_items
        self.n_levels = n_levels
        self.slate_size = slate_size
        self.vocab_size = vocab_size
        
        # 简化：直接输出 token 概率
        self.embed = nn.Embedding(n_items, embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size)
        
        # RAPI 相关参数
        self.sara_layer_weights = [0.05, 0.25, 0.7]
        self.sara_eta = 1.0
        
    def forward(self, history_items, pool_tokens=None, pool_phis=None):
        """
        history_items: (B, H) 物品ID
        返回: sid_tokens (B, slate_size, n_levels)
        """
        B = history_items.shape[0]
        
        # Embedding
        history_embs = self.embed(history_items)  # (B, H, embed_dim)
        
        # 取最后一个作为 context
        context = history_embs[:, -1, :]  # (B, embed_dim)
        
        # 每层生成一个 token
        sid_tokens = []
        for level in range(self.n_levels):
            logits = self.fc(context)  # (B, vocab_size)
            
            # RAPI: 如果有后悔池，应用惩罚
            if pool_tokens is not None and pool_tokens.shape[0] > 0:
                logits = self._apply_regret_penalty(logits, pool_tokens, pool_phis, level)
            
            probs = F.softmax(logits, dim=-1)
            tokens = torch.multinomial(probs, self.slate_size, replacement=True)
            sid_tokens.append(tokens)
        
        # (B, slate_size, n_levels)
        sid_tokens = torch.stack(sid_tokens, dim=-1)
        return sid_tokens
    
    def _apply_regret_penalty(self, logits, pool_tokens, pool_phis, level):
        """应用 RAPI 惩罚"""
        V = logits.shape[-1]
        
        # pool_tokens: (N, n_levels) -> one-hot: (N, n_levels, V)
        B_pool = F.one_hot(pool_tokens, num_classes=V).float()
        
        # 计算相似度
        if level == 0:
            similarity = torch.ones(logits.shape[0], pool_tokens.shape[0], device=logits.device)
        else:
            # 简化：只检查当前层
            B_curr = B_pool[:, level, :]  # (N, V)
            similarity = torch.matmul(logits, B_curr.T)  # (B, N)
        
        # 惩罚 = 相似度 * phi
        penalty = similarity * pool_phis.unsqueeze(0)  # (B, N)
        penalty = penalty.sum(dim=-1, keepdim=True)  # (B, 1)
        
        # 应用惩罚
        return logits - self.sara_eta * penalty

# ==========================================
# 4. 最小 Critic
# ==========================================
class MiniCritic(nn.Module):
    def __init__(self, embed_dim=32, n_levels=3):
        super().__init__()
        self.fc = nn.Linear(embed_dim, n_levels + 1)
        
    def forward(self, history_embs):
        """返回每层的 value"""
        context = history_embs[:, -1, :]
        return self.fc(context)  # (B, n_levels+1)

# ==========================================
# 5. RRCA + RAPI Agent
# ==========================================
class RRCARAPIAgent:
    def __init__(self, args, env):
        self.env = env
        self.n_items = args['n_items']
        self.n_levels = args['n_levels']
        self.slate_size = args['slate_size']
        self.vocab_size = args['vocab_size']
        
        # 模型
        self.actor = MiniActor(
            n_items=self.n_items,
            embed_dim=args['embed_dim'],
            vocab_size=self.vocab_size,
            n_levels=self.n_levels,
            slate_size=self.slate_size
        )
        self.critic = MiniCritic(
            embed_dim=args['embed_dim'],
            n_levels=self.n_levels
        )
        
        # 优化器
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=args['actor_lr'])
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=args['critic_lr'])
        
        # RRCA 参数
        self.lambda_unlike = args['lambda_unlike']
        self.regret_gamma = args['regret_gamma']
        
        # RAPI 参数
        self.regret_pool_size = args['regret_pool_size']
        
        # 用户独立后悔池: (B, N, L+1)
        B = args['batch_size']
        self.user_regret_pool = torch.full((B, self.regret_pool_size, self.n_levels + 1), 
                                          -1, dtype=torch.long)
        
        # 历史记录
        self.history = {
            'actor_loss': [],
            'critic_loss': [],
            'reward': [],
            'regret_events': [],
            'pool_size': []
        }
        
    def _sync_pool_to_actor(self):
        """将用户后悔池同步到 Actor"""
        B = self.user_regret_pool.shape[0]
        pool_list = []
        phi_list = []
        
        for b in range(B):
            pool_b = self.user_regret_pool[b]
            valid_mask = pool_b[:, -1] != -1
            if valid_mask.sum() > 0:
                tokens = pool_b[valid_mask, :self.n_levels]
                phis = pool_b[valid_mask, self.n_levels].float() / 100
                pool_list.append(tokens)
                phi_list.append(phis)
        
        if len(pool_list) > 0:
            return torch.cat(pool_list, dim=0), torch.cat(phi_list, dim=0)
        return None, None
    
    def _collect_session(self):
        """收集一个完整 session"""
        histories = self.env.reset(self.user_regret_pool.shape[0])
        
        # 转换为 tensor
        history_items = torch.tensor(histories, dtype=torch.long)  # (B, H)
        
        trajectory = []
        
        for step in range(20):  # max steps
            # 同步后悔池
            pool_tokens, pool_phis = self._sync_pool_to_actor()
            
            # Actor 推理
            sid_tokens = self.actor(history_items, pool_tokens, pool_phis)
            
            # 转换为物品 ID (简化: 直接用 token 作为 item id)
            actions = sid_tokens[:, :, 0] % self.n_items  # (B, slate_size)
            
            # Env step
            reward, done, info = self.env.step(actions)
            
            trajectory.append({
                'history_items': history_items.clone(),
                'sid_tokens': sid_tokens,
                'actions': actions,
                'reward': reward,
                'done': done,
                'info': info
            })
            
            # 更新历史 (简化: 滑动窗口)
            history_items = torch.cat([history_items[:, 1:], actions], dim=1)
            
            if done.all():
                break
        
        return trajectory
    
    def _simulate_regret_explosion(self, trajectory):
        """模拟延迟后悔引爆"""
        T = len(trajectory)
        B = trajectory[0]['actions'].shape[0]
        
        regret_events = []
        
        for t in range(T):
            info = trajectory[t]['info']
            sara_delta_t = info['sara_delta_t']
            sara_unlike_prob = info['sara_unlike_prob']
            actions = info['actions']
            
            B_cur, K = sara_delta_t.shape
            
            for b in range(B_cur):
                for k in range(K):
                    delta_t = sara_delta_t[b, k].item()
                    if delta_t == 0:
                        continue
                    
                    t_explode = t + delta_t
                    if t_explode >= T:
                        continue
                    
                    # 模拟引爆
                    if torch.rand(1).item() < sara_unlike_prob[b, k].item():
                        # 记录后悔事件
                        regret_events.append({
                            't': t,
                            't_explode': t_explode,
                            'batch_idx': b,
                            'sid_tokens': trajectory[t]['sid_tokens'][b, k, :],
                            'phi': -self.lambda_unlike * (self.regret_gamma ** delta_t)
                        })
        
        return regret_events
    
    def _update_user_pool(self, regret_events):
        """更新用户后悔池"""
        for event in regret_events:
            sid_tokens = event['sid_tokens']
            phi = event['phi']
            b = event['batch_idx']
            
            pool_b = self.user_regret_pool[b]
            empty_slots = (pool_b[:, -1] == -1).nonzero(as_tuple=True)[0]
            
            if len(empty_slots) > 0:
                slot = empty_slots[0]
                self.user_regret_pool[b, slot, :self.n_levels] = sid_tokens.cpu()
                self.user_regret_pool[b, slot, self.n_levels] = int(phi * 100)
            else:
                slot = 0  # 简化: 替换第一个
                self.user_regret_pool[b, slot, :self.n_levels] = sid_tokens.cpu()
                self.user_regret_pool[b, slot, self.n_levels] = int(phi * 100)
    
    def train_step(self):
        """一次训练 step"""
        # 1. 收集 session
        trajectory = self._collect_session()
        
        # 2. 模拟引爆
        regret_events = self._simulate_regret_explosion(trajectory)
        
        # 3. 更新用户后悔池
        self._update_user_pool(regret_events)
        
        # 4. 计算 loss (简化版)
        T = len(trajectory)
        
        # 简化: 只取最后一个 timestep 的 reward
        final_reward = trajectory[-1]['reward']  # (B,)
        
        # Critic: 计算 V(s)
        history_items = trajectory[-1]['history_items']
        history_embs = self.actor.embed(history_items)
        v_out = self.critic(history_embs)  # (B, n_levels+1)
        V = v_out.mean()
        
        # 简化 Critic Loss
        target_V = final_reward.mean()
        critic_loss = F.mse_loss(V, target_V.detach())
        
        # Actor Loss
        actor_loss = -final_reward.mean()
        
        # 更新
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        
        # 记录
        self.history['actor_loss'].append(actor_loss.item())
        self.history['critic_loss'].append(critic_loss.item())
        self.history['reward'].append(final_reward.mean().item())
        self.history['regret_events'].append(len(regret_events))
        
        # 计算池大小
        pool_size = (self.user_regret_pool[:, :, -1] != -1).sum().item()
        self.history['pool_size'].append(pool_size)
        
        return len(regret_events)

# ==========================================
# 6. 运行测试
# ==========================================
if __name__ == "__main__":
    # 参数
    args = {
        'n_users': 100,
        'n_items': 1000,
        'batch_size': 8,
        'n_levels': 3,
        'slate_size': 3,
        'vocab_size': 16,
        'embed_dim': 32,
        'actor_lr': 0.001,
        'critic_lr': 0.001,
        'lambda_unlike': 1.0,
        'regret_gamma': 0.9,
        'regret_pool_size': 10
    }
    
    # 初始化
    print("=" * 50)
    print("RRCA + RAPI 最小实现测试")
    print("=" * 50)
    
    dataset = MiniDataset(
        n_users=args['n_users'],
        n_items=args['n_items']
    )
    env = MiniEnvironment(dataset)
    agent = RRCARAPIAgent(args, env)
    
    print(f"用户数: {args['n_users']}")
    print(f"物品数: {args['n_items']}")
    print(f"Batch size: {args['batch_size']}")
    print(f"后悔池大小: {args['regret_pool_size']}")
    print()
    
    # 训练
    print("开始训练...")
    for i in range(50):
        n_regret = agent.train_step()
        
        if (i + 1) % 10 == 0:
            avg_reward = np.mean(agent.history['reward'][-10:])
            avg_regret = np.mean(agent.history['regret_events'][-10:])
            pool_size = agent.history['pool_size'][-1]
            print(f"Step {i+1:3d} | Reward: {avg_reward:.3f} | "
                  f"Regret events: {avg_regret:.1f} | Pool size: {pool_size}")
    
    print()
    print("=" * 50)
    print("测试完成!")
    print("=" * 50)
    
    # 显示最终后悔池状态
    print("\n最终后悔池状态:")
    for b in range(min(3, args['batch_size'])):
        pool_b = agent.user_regret_pool[b]
        valid = pool_b[pool_b[:, -1] != -1]
        print(f"  用户 {b}: {len(valid)} 条后悔记录")
        if len(valid) > 0:
            print(f"    第一条: sid={valid[0, :3].tolist()}, phi={valid[0, 3].item()/100:.2f}")
