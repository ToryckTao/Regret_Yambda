"""
RRCA + RAPI 最小实现测试 - 纯验证流程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==========================================
# 1. 最小数据集
# ==========================================
class MiniDataset:
    def __init__(self, n_users=100, n_items=1000):
        self.n_users = n_users
        self.n_items = n_items
        self.user_history = {}
        for u in range(n_users):
            self.user_history[u] = list(np.random.randint(0, n_items, 20))
    
    def sample_users(self, batch_size):
        user_ids = np.random.randint(0, self.n_users, batch_size)
        return user_ids, [self.user_history[u] for u in user_ids]

# ==========================================
# 2. 最小环境
# ==========================================
class MiniEnvironment:
    def __init__(self, dataset):
        self.dataset = dataset
        
    def reset(self, batch_size):
        _, histories = self.dataset.sample_users(batch_size)
        return histories
    
    def step(self, actions):
        B, K = actions.shape
        # 随机奖励
        reward = torch.rand(B)
        done = torch.rand(B) > 0.8
        # sara_delta_t 和 sara_unlike_prob
        sara_delta_t = torch.randint(0, 5, (B, K))
        sara_unlike_prob = torch.rand(B, K) * 0.3
        return reward, done, {
            'sara_delta_t': sara_delta_t,
            'sara_unlike_prob': sara_unlike_prob,
            'actions': actions
        }

# ==========================================
# 3. 最小 Actor
# ==========================================
class MiniActor(nn.Module):
    def __init__(self, n_items=1000):
        super().__init__()
        self.embed = nn.Embedding(n_items, 32)
        self.fc = nn.Linear(32, 16)
        
    def forward(self, history):
        emb = self.embed(history[:, -1])
        logits = self.fc(emb)
        probs = F.softmax(logits, dim=-1)
        tokens = torch.multinomial(probs, 3, replacement=True)
        return tokens

# ==========================================
# 4. RRCA + RAPI Agent
# ==========================================
class RRCARAPIAgent:
    def __init__(self, args, env):
        self.env = env
        self.n_levels = args['n_levels']
        self.n_items = args['n_items']
        
        self.actor = MiniActor(n_items=args['n_items'])
        self.lambda_unlike = args['lambda_unlike']
        self.regret_gamma = args['regret_gamma']
        self.regret_pool_size = args['regret_pool_size']
        
        # 用户独立后悔池: (B, N, L+1)
        B = args['batch_size']
        self.user_regret_pool = torch.full((B, self.regret_pool_size, self.n_levels + 1), 
                                          -1, dtype=torch.long)
        
        self.history = {'regret_events': [], 'pool_size': []}
    
    def _sync_pool(self):
        """同步池到全局"""
        pool_list = []
        phi_list = []
        B = self.user_regret_pool.shape[0]
        for b in range(B):
            pool_b = self.user_regret_pool[b]
            valid = pool_b[pool_b[:, -1] != -1]
            if len(valid) > 0:
                pool_list.append(valid[:, :self.n_levels])
                phi_list.append(valid[:, self.n_levels].float() / 100)
        if pool_list:
            return torch.cat(pool_list, dim=0), torch.cat(phi_list, dim=0)
        return None, None
    
    def collect_session(self):
        """收集 session"""
        histories = self.env.reset(self.user_regret_pool.shape[0])
        history_t = torch.tensor(histories, dtype=torch.long)
        
        trajectory = []
        for step in range(20):
            pool_tokens, _ = self._sync_pool()
            tokens = self.actor(history_t)
            actions = tokens % self.n_items
            reward, done, info = self.env.step(actions)
            
            trajectory.append({
                'history': history_t.clone(),
                'tokens': tokens,
                'actions': actions,
                'reward': reward,
                'done': done,
                'info': info
            })
            history_t = torch.cat([history_t[:, 1:], actions], dim=1)
            if done.all():
                break
        return trajectory
    
    def simulate_regret(self, trajectory):
        """模拟引爆"""
        T = len(trajectory)
        regret_events = []
        
        for t in range(T):
            info = trajectory[t]['info']
            sara_delta_t = info['sara_delta_t']
            sara_unlike_prob = info['sara_unlike_prob']
            B, K = sara_delta_t.shape
            
            for b in range(B):
                for k in range(K):
                    delta_t = sara_delta_t[b, k].item()
                    if delta_t == 0:
                        continue
                    t_exp = t + delta_t
                    if t_exp >= T:
                        continue
                    if torch.rand(1).item() < sara_unlike_prob[b, k].item():
                        regret_events.append({
                            'batch_idx': b,
                            'tokens': trajectory[t]['tokens'][b, k],
                            'phi': -self.lambda_unlike * (self.regret_gamma ** delta_t)
                        })
        return regret_events
    
    def update_pool(self, regret_events):
        """更新用户后悔池"""
        for event in regret_events:
            b = event['batch_idx']
            tokens = event['tokens']
            phi = event['phi']
            
            pool_b = self.user_regret_pool[b]
            empty = (pool_b[:, -1] == -1).nonzero(as_tuple=True)[0]
            if len(empty) > 0:
                slot = empty[0]
                self.user_regret_pool[b, slot, :self.n_levels] = tokens.cpu()
                self.user_regret_pool[b, slot, self.n_levels] = int(phi * 100)
            else:
                self.user_regret_pool[b, 0, :self.n_levels] = tokens.cpu()
                self.user_regret_pool[b, 0, self.n_levels] = int(phi * 100)
    
    def train_step(self):
        """一次训练 step"""
        trajectory = self.collect_session()
        regret_events = self.simulate_regret(trajectory)
        self.update_pool(regret_events)
        
        self.history['regret_events'].append(len(regret_events))
        pool_size = (self.user_regret_pool[:, :, -1] != -1).sum().item()
        self.history['pool_size'].append(pool_size)
        
        return len(regret_events)

# ==========================================
# 5. 运行测试
# ==========================================
if __name__ == "__main__":
    args = {
        'n_users': 100,
        'n_items': 1000,
        'batch_size': 8,
        'n_levels': 3,
        'regret_pool_size': 10,
        'lambda_unlike': 1.0,
        'regret_gamma': 0.9
    }
    
    print("=" * 50)
    print("RRCA + RAPI 最小实现测试")
    print("=" * 50)
    
    dataset = MiniDataset(args['n_users'], args['n_items'])
    env = MiniEnvironment(dataset)
    agent = RRCARAPIAgent(args, env)
    
    print(f"用户数: {args['n_users']}, 物品数: {args['n_items']}")
    print(f"Batch: {args['batch_size']}, 后悔池: {args['regret_pool_size']}")
    print()
    
    # 训练
    for i in range(50):
        n_regret = agent.train_step()
        
        if (i + 1) % 10 == 0:
            avg_regret = np.mean(agent.history['regret_events'][-10:])
            pool_size = agent.history['pool_size'][-1]
            print(f"Step {i+1:3d} | Regret events: {avg_regret:.1f} | Pool size: {pool_size}")
    
    print()
    print("=" * 50)
    print("测试完成!")
    print("=" * 50)
    
    # 显示后悔池
    print("\n最终后悔池状态 (前3个用户):")
    for b in range(min(3, args['batch_size'])):
        pool_b = agent.user_regret_pool[b]
        valid = pool_b[pool_b[:, -1] != -1]
        print(f"  用户 {b}: {len(valid)} 条记录")
