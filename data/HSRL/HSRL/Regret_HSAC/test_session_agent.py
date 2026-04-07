import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. 模拟网络与环境 (Mock Facade)
# ==========================================
class MockActor(nn.Module):
    def __init__(self, vocab_size=100, embed_dim=32, k=3, l=4):
        super().__init__()
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.k = k
        self.l = l
        self.vocab_size = vocab_size

    def forward(self, obs):
        # 模拟生成 K 个物品，每个物品 L 层 Semantic ID
        B = obs.shape[0]
        context = torch.randn(B, 32)
        sid_logits = [self.fc(context) for _ in range(self.l)]
        sid_tokens = torch.randint(0, self.vocab_size, (B, self.k, self.l))
        return context, sid_logits, sid_tokens

class MockCritic(nn.Module):
    def __init__(self, embed_dim=32, l=4):
        super().__init__()
        self.fc = nn.Linear(embed_dim, l + 1)

    def forward(self, context_dict):
        context = context_dict['context_list']
        v_seq = self.fc(context) # (B, L+1)
        return {'v_seq': v_seq}

class MockFacade:
    def __init__(self, batch_size=8, vocab_size=100):
        self.actor = MockActor(vocab_size=vocab_size)
        self.critic = MockCritic()
        self.batch_size = batch_size
        self.step_count = 0

    def reset_env(self, args):
        self.step_count = 0
        return torch.randn(args['batch_size'], 10) # Mock obs

    def apply_policy(self, obs, actor, epsilon, do_explore):
        context, sid_logits, sid_tokens = actor(obs)
        return {
            'context_list': context,
            'sid_logits': sid_logits,
            'sid_tokens': sid_tokens
        }

    def env_step(self, policy_output):
        self.step_count += 1
        next_obs = torch.randn(self.batch_size, 10)
        reward = torch.rand(self.batch_size)
        # 模拟第 3 步开始有用户陆续退出，第 5 步全部退出
        done = torch.rand(self.batch_size) < (self.step_count / 5.0) 
        if self.step_count >= 5:
            done = torch.ones(self.batch_size, dtype=torch.bool)
        return next_obs, reward, done, {}

# ==========================================
# 2. 修复后的 SARA_Session_Agent
# ==========================================
class MockArgs:
    episode_batch_size = 4
    session_max_steps = 5
    actor_lr = 1e-3
    critic_lr = 1e-3
    actor_decay = 1e-5
    critic_decay = 1e-5
    target_mitigate_coef = 0.01
    advantage_bias = 0.0
    entropy_coef = 0.1
    token_lr = 1e-3
    behavior_lr = 1e-4
    behavior_decay = 1e-5

class SARA_Session_Agent:
    def __init__(self, args, facade):
        self.episode_batch_size = args.episode_batch_size
        self.session_max_steps = args.session_max_steps
        self.facade = facade
        self.actor = facade.actor
        self.critic = facade.critic
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        self.token_weight = nn.Parameter(torch.ones(4 + 1) / (4 + 1))
        
        self.gamma = 0.99
        self.advantage_bias = args.advantage_bias
        self.entropy_coef = args.entropy_coef

    def collect_session(self):
        observation = self.facade.reset_env({"batch_size": self.episode_batch_size})
        trajectory = []
        
        for step in range(self.session_max_steps):
            policy_output = self.facade.apply_policy(observation, self.actor, epsilon=0.0, do_explore=False)
            next_observation, reward, done, info = self.facade.env_step(policy_output)
            
            trajectory.append({
                'observation': observation,
                'policy_output': policy_output, # 直接缓存，避免后续重复计算
                'reward': reward,
                'done': done
            })
            observation = next_observation
            
            if done.all().item():
                break
        return trajectory

    def get_session_a2c_loss(self, trajectory):
        T = len(trajectory)
        B = self.episode_batch_size
        
        all_rewards = torch.stack([t['reward'] for t in trajectory]) # (T, B)
        all_dones = torch.stack([t['done'] for t in trajectory]).float() # (T, B)
        
        # 1. 直接从 trajectory 提取 context，不重新 forward
        all_context = torch.stack([t['policy_output']['context_list'] for t in trajectory]) # (T, B, D)
        
        V_s_list = []
        for t in range(T):
            V_out = self.critic({'context_list': all_context[t]})
            V_s_seq = V_out['v_seq']
            token_w = torch.softmax(self.token_weight, dim=0).unsqueeze(0).expand(B, -1)
            V_s_weighted = (V_s_seq * token_w).sum(dim=1)
            V_s_list.append(V_s_weighted)
            
        V_s_all = torch.stack(V_s_list) # (T, B)
        
        # TD Target
        with torch.no_grad():
            V_sp = V_s_all[-1] 
            Q_s_list = []
            for t in reversed(range(T)):
                if t == T - 1:
                    target = all_rewards[t] + self.gamma * (1 - all_dones[t]) * V_sp
                else:
                    target = all_rewards[t] + self.gamma * (1 - all_dones[t]) * V_s_all[t + 1]
                Q_s_list.append(target)
            Q_s_all = torch.stack(list(reversed(Q_s_list)))
            
            # 断开 Advantage 的计算图，这是避免 retain_graph 报错的核心
            advantage = Q_s_all - V_s_all 
            advantage = torch.clamp(advantage, -1, 1).detach()
            
        # Critic Loss
        value_loss = F.mse_loss(V_s_all, Q_s_all)
        self.critic_optimizer.zero_grad()
        value_loss.backward() # 安全：计算图在这里终结
        self.critic_optimizer.step()
        
        # Actor Loss
        actor_loss = 0.0
        entropy_raw = 0.0
        valid_steps = 0.0
        
        for t in range(T):
            # 添加 mask，过滤掉已经 done 的 batch item
            mask_t = 1.0 - all_dones[t]
            if mask_t.sum() == 0:
                continue
                
            po = trajectory[t]['policy_output']
            sid_tokens = po['sid_tokens'] # (B, K, L)
            sid_logits_list = po['sid_logits']
            B_t, K, L = sid_tokens.shape
            
            sid_tokens_flat = sid_tokens.view(B_t * K, L)
            level_probs = [torch.softmax(logits_l, dim=-1) for logits_l in sid_logits_list]
            level_probs_flat = [p.repeat_interleave(K, dim=0) for p in level_probs]
            
            nll_slot = 0.0
            for l in range(L):
                probs_l_flat = level_probs_flat[l]
                z_l = sid_tokens_flat[:, l].view(-1, 1)
                logp_l = torch.log(torch.gather(probs_l_flat, 1, z_l) + 1e-12).squeeze(1)
                nll_slot = nll_slot + (-logp_l)
                
            nll_per_sample = nll_slot.view(B_t, K).mean(dim=1) # (B,)
            
            # 使用掩码计算当前步的 actor loss
            actor_loss_t = nll_per_sample * (advantage[t] + self.advantage_bias)
            actor_loss_t = (actor_loss_t * mask_t).sum() / (mask_t.sum() + 1e-8)
            actor_loss += actor_loss_t
            valid_steps += 1.0
            
            for l in range(L):
                entropy_t = (level_probs[l] * torch.log(level_probs[l] + 1e-12)).sum(dim=-1).mean()
                entropy_raw += entropy_t
                
        actor_loss = actor_loss / (valid_steps + 1e-8)
        entropy_raw = entropy_raw / (T + 1e-8)
        
        total_actor = actor_loss + self.entropy_coef * entropy_raw
        self.actor_optimizer.zero_grad()
        total_actor.backward() # 安全：使用了提前缓存的 logits，无冲突
        self.actor_optimizer.step()
        
        return value_loss.item(), actor_loss.item(), entropy_raw.item(), advantage.mean().item()

    def step_train(self):
        trajectory = self.collect_session()
        c_loss, a_loss, ent_loss, adv = self.get_session_a2c_loss(trajectory)
        print(f"Traj Length: {len(trajectory)} | Critic Loss: {c_loss:.4f} | Actor Loss: {a_loss:.4f} | Adv: {adv:.4f}")

# ==========================================
# 3. 执行测试
# ==========================================
if __name__ == "__main__":
    args = MockArgs()
    facade = MockFacade(batch_size=args.episode_batch_size)
    agent = SARA_Session_Agent(args, facade)
    
    print("=== 开始跑 Session 级 Actor-Critic 训练测试 ===")
    for i in range(10):
        print(f"Epoch {i+1}: ", end="")
        agent.step_train()
    print("=== 测试通过，没有触发 OOM 或维度报错 ===")
