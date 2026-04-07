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
        self.sid_vocab_sizes = [vocab_size] * l
        self.sid_temp = 1.0
        self.sara_layer_weights = [0.05, 0.25, 0.7, 0.7][:l]
        self.sara_eta = 1.0

    def forward(self, obs):
        B = obs.shape[0]
        context = torch.randn(B, 32)
        sid_logits = [self.fc(context) for _ in range(self.l)]
        sid_tokens = torch.randint(0, self.vocab_size, (B, self.k, self.l))
        return context, sid_logits, sid_tokens
    
    def encode_state(self, feed_dict):
        # feed_dict = {'context': tensor(B, 10)}
        context_input = feed_dict['context']
        # 映射到 embedding 维度
        return {'state_emb': torch.randn(context_input.shape[0], 32, device=context_input.device)}
    
    def get_sara_logits(self, feed_dict, pool_tokens=None, pool_phis=None):
        """带 RAPI 拦截的推理"""
        enc = self.encode_state(feed_dict)
        context = enc['state_emb']
        B = context.shape[0]
        
        sid_logits = []
        context_list = [context]
        
        V_dim = self.sid_vocab_sizes[0]
        L = self.l
        tau = self.sid_temp
        
        C_curr = torch.zeros(B, L, V_dim, device=context.device)
        
        if pool_tokens is not None and pool_tokens.shape[0] > 0:
            N = pool_tokens.shape[0]
            B_pool = F.one_hot(pool_tokens, num_classes=V_dim).float()
        else:
            pool_tokens = None
            
        W = torch.tensor(self.sara_layer_weights, device=context.device)

        for l in range(L):
            logits_l = self.fc(context)  # [B, V]
            
            if pool_tokens is not None:
                if l == 0:
                    similarity_vec = torch.ones(B, N, device=context.device)
                else:
                    C_slice = C_curr[:, :l, :]
                    B_slice = B_pool[:, :l, :]
                    hit_matrix = (C_slice.unsqueeze(1) * B_slice.unsqueeze(0)).sum(dim=-1)
                    W_active = W[:l].unsqueeze(0).unsqueeze(0)
                    similarity_vec = (hit_matrix * W_active).sum(dim=-1)
                
                active_penalty = similarity_vec * pool_phis.unsqueeze(0)
                B_curr_level = B_pool[:, l, :]
                D = torch.matmul(active_penalty, B_curr_level)
                logits_l = logits_l - self.sara_eta * D

            sid_logits.append(logits_l)
            
            probs_l = F.softmax(logits_l / tau, dim=-1)
            C_curr[:, l, :] = probs_l
            
            emb_tbl = self.fc.weight
            exp_emb = torch.matmul(probs_l, emb_tbl)
            context = context - exp_emb
            context_list.append(context)
        
        return {'sid_logits': sid_logits, 'context_list': context_list}

class MockCritic(nn.Module):
    def __init__(self, embed_dim=32, l=4):
        super().__init__()
        self.fc = nn.Linear(embed_dim, l + 1)

    def forward(self, context_dict):
        context = context_dict['context_list']
        v_seq = self.fc(context)
        return {'v_seq': v_seq}

class MockFacade:
    def __init__(self, batch_size=8, vocab_size=100):
        self.actor = MockActor(vocab_size=vocab_size)
        self.critic = MockCritic()
        self.batch_size = batch_size
        self.step_count = 0
        self.global_sara_pool = {'tokens': [], 'phis': []}

    def reset_env(self, args):
        self.step_count = 0
        return {'context': torch.randn(args['batch_size'], 10)}

    def apply_policy(self, obs, actor, epsilon, do_explore):
        # obs 是 {'context': tensor(B, 10)}
        obs_context = obs['context']
        
        # 直接用 actor 的 get_sara_logits
        pool_tokens = None
        pool_phis = None
        if len(self.global_sara_pool['tokens']) > 0 and obs_context.shape[0] > 0:
            device = obs_context.device
            pool_tokens = torch.stack(self.global_sara_pool['tokens']).to(device)
            pool_phis = torch.stack(self.global_sara_pool['phis']).to(device)
        
        # 用 encode_state 格式
        feed_dict = {'context': obs_context}
        out_dict = actor.get_sara_logits(feed_dict, pool_tokens, pool_phis)
        
        # 用普通 forward 获取 sid_tokens
        context_out, sid_logits_out, sid_tokens = actor(obs_context)
        
        return {
            'context_list': out_dict['context_list'][-1],  # 取最后一个 context
            'sid_logits': out_dict['sid_logits'],
            'sid_tokens': sid_tokens
        }

    def env_step(self, policy_output):
        self.step_count += 1
        next_obs = {'context': torch.randn(self.batch_size, 10)}
        reward = torch.rand(self.batch_size)
        
        done = torch.rand(self.batch_size) < (self.step_count / 5.0) 
        if self.step_count >= 5:
            done = torch.ones(self.batch_size, dtype=torch.bool)
        
        B = self.batch_size
        K = 3
        sara_delta_t = torch.randint(0, 3, (B, K))
        sara_unlike_prob = torch.rand(B, K) * 0.3
        
        info = {
            'sara_delta_t': sara_delta_t,
            'sara_unlike_prob': sara_unlike_prob,
            'sid_tokens': torch.randint(0, 100, (B, K, 4))  # L=4
        }
        
        return next_obs, reward, done, info

# ==========================================
# 2. RRCA SARA_Session_Agent (含 RAPI) - 简化版测试
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
    
    # RRCA 参数
    lambda_unlike = 1.0
    lambda_undislike = 0.5
    regret_gamma = 0.9
    
    # RAPI 参数
    regret_pool_size = 10
    regret_eta = 1.0
    regret_W = '0.05,0.25,0.7,0.7'

class SARA_Session_Agent:
    def __init__(self, args, facade):
        self.episode_batch_size = args.episode_batch_size
        self.session_max_steps = args.session_max_steps
        self.facade = facade
        self.actor = facade.actor
        self.critic = facade.critic
        
        # 确保模型在训练模式
        self.actor.train()
        self.critic.train()
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)
        self.token_weight = nn.Parameter(torch.ones(4 + 1) / (4 + 1))
        
        self.gamma = 0.99
        self.advantage_bias = args.advantage_bias
        self.entropy_coef = args.entropy_coef
        
        # RRCA 参数
        self.lambda_unlike = args.lambda_unlike
        self.lambda_undislike = args.lambda_undislike
        self.regret_gamma = args.regret_gamma
        
        # RAPI 参数
        self.regret_pool_size = args.regret_pool_size
        self.regret_eta = args.regret_eta
        self.regret_W = torch.tensor([float(x) for x in args.regret_W.split(',')])
        self.L = facade.actor.l
        self.global_regret_pool = {'tokens': [], 'phis': []}
        
        # 记录历史
        self.training_history = {
            'regret_events': [],
            'pool_size': []
        }

    def _sync_regret_pool_to_facade(self):
        if len(self.global_regret_pool['tokens']) > 0:
            pool_tokens = torch.cat(self.global_regret_pool['tokens'], dim=0)
            pool_phis = torch.cat(self.global_regret_pool['phis'], dim=0)
            self.facade.global_sara_pool = {
                'tokens': [pool_tokens],
                'phis': [pool_phis]
            }
        else:
            self.facade.global_sara_pool = {'tokens': [], 'phis': []}

    def collect_session(self):
        observation = self.facade.reset_env({"batch_size": self.episode_batch_size})
        trajectory = []
        
        for step in range(self.session_max_steps):
            self._sync_regret_pool_to_facade()
            policy_output = self.facade.apply_policy(observation, self.actor, epsilon=0.0, do_explore=False)
            next_observation, reward, done, info = self.facade.env_step(policy_output)
            
            trajectory.append({
                'observation': observation,
                'policy_output': policy_output,
                'reward': reward,
                'done': done,
                'info': info
            })
            observation = next_observation
            
            if done.all().item():
                break
        return trajectory
    
    def _simulate_regret_explosion(self, trajectory):
        T = len(trajectory)
        B = self.episode_batch_size
        regret_events = []
        
        for t in range(T):
            info = trajectory[t]['info']
            sara_delta_t = info.get('sara_delta_t', None)
            sara_unlike_prob = info.get('sara_unlike_prob', None)
            sid_tokens = info.get('sid_tokens', None)
            
            if sara_delta_t is None or sara_unlike_prob is None:
                continue
            
            B_cur, K = sara_delta_t.shape
            
            for b in range(B_cur):
                for k in range(K):
                    delta_t = sara_delta_t[b, k].item()
                    unlike_prob = sara_unlike_prob[b, k].item()
                    
                    if delta_t == 0:
                        continue
                    
                    t_explode = t + delta_t
                    
                    if t_explode >= T:
                        continue
                    
                    if torch.rand(1).item() < unlike_prob:
                        if sid_tokens is not None:
                            item_sid_tokens = sid_tokens[b, k, :]
                        else:
                            item_sid_tokens = None
                        
                        regret_events.append({
                            't': t,
                            't_explode': t_explode,
                            'batch_idx': b,
                            'item_idx': k,
                            'psi': -self.lambda_unlike,
                            'sid_tokens': item_sid_tokens,
                            'phi': -self.lambda_unlike * (self.regret_gamma ** delta_t)
                        })
        
        return regret_events
    
    def _apply_rrca_advantage(self, advantage, regret_events, T, B):
        advantage_rrca = advantage.clone()
        
        for event in regret_events:
            t = event['t']
            t_explode = event['t_explode']
            b = event['batch_idx']
            psi = event['psi']
            gamma_pow = self.regret_gamma ** (t_explode - t)
            advantage_rrca[t, b] += psi * gamma_pow
        
        return advantage_rrca
    
    def _update_global_regret_pool(self, regret_events):
        for event in regret_events:
            sid_tokens = event.get('sid_tokens', None)
            phi = event.get('phi', None)
            
            if sid_tokens is None or phi is None:
                continue
            
            tokens_tensor = sid_tokens.unsqueeze(0).detach().cpu()
            phi_tensor = torch.tensor([phi], dtype=torch.float32).detach().cpu()
            
            self.global_regret_pool['tokens'].append(tokens_tensor)
            self.global_regret_pool['phis'].append(phi_tensor)
        
        while len(self.global_regret_pool['tokens']) > self.regret_pool_size:
            self.global_regret_pool['tokens'].pop(0)
            self.global_regret_pool['phis'].pop(0)

    def get_session_a2c_loss(self, trajectory):
        T = len(trajectory)
        B = self.episode_batch_size
        
        all_rewards = torch.stack([t['reward'] for t in trajectory])
        all_dones = torch.stack([t['done'] for t in trajectory]).float()
        
        all_context = torch.stack([t['policy_output']['context_list'] for t in trajectory])  # (T, B, D)
        
        # Critic: 计算 V(s) - 前向计算
        V_s_list = []
        for t in range(T):
            V_out = self.critic({'context_list': all_context[t]})
            V_s_seq = V_out['v_seq']  # (B, L+1)
            token_w = torch.softmax(self.token_weight, dim=0).unsqueeze(0).expand(B, -1)
            V_s_weighted = (V_s_seq * token_w).sum(dim=1)  # (B,)
            V_s_list.append(V_s_weighted)
            
        V_s_all = torch.stack(V_s_list)  # (T, B) - 需要梯度
        
        # TD Target - 单独计算 Q，不共享梯度
        with torch.no_grad():
            V_sp = V_s_all[-1].detach().clone()
            Q_s_list = []
            for t in reversed(range(T)):
                if t == T - 1:
                    target = all_rewards[t] + self.gamma * (1 - all_dones[t]) * V_sp
                else:
                    target = all_rewards[t] + self.gamma * (1 - all_dones[t]) * V_s_all[t + 1].detach().clone()
                Q_s_list.append(target)
            Q_s_target = torch.stack(list(reversed(Q_s_list)))  # (T, B) - 无梯度
            
            advantage = Q_s_target - V_s_all.detach()
            advantage = torch.clamp(advantage, -1, 1)
        
        # RRCA: 回溯修正
        with torch.no_grad():
            regret_events = self._simulate_regret_explosion(trajectory)
            advantage_rrca = self._apply_rrca_advantage(advantage, regret_events, T, B)
            advantage_rrca = advantage_rrca.detach()
        
        # RAPI: 更新后悔池
        self._update_global_regret_pool(regret_events)
        
        # Critic Loss - 使用 V_s_all 计算（保留梯度）
        value_loss = F.mse_loss(V_s_all, Q_s_target.detach())
        
        # Actor Loss - 用 RRCA 修正后的 advantage
        actor_loss = 0.0
        entropy_raw = 0.0
        valid_steps = 0.0
        
        for t in range(T):
            mask_t = 1.0 - all_dones[t]
            if mask_t.sum() == 0:
                continue
                
            po = trajectory[t]['policy_output']
            sid_tokens = po['sid_tokens']
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
                
            nll_per_sample = nll_slot.view(B_t, K).mean(dim=1)
            
            actor_loss_t = nll_per_sample * (advantage_rrca[t] + self.advantage_bias)
            actor_loss_t = (actor_loss_t * mask_t).sum() / (mask_t.sum() + 1e-8)
            actor_loss += actor_loss_t
            valid_steps += 1.0
            
            for l in range(L):
                entropy_t = (level_probs[l] * torch.log(level_probs[l] + 1e-12)).sum(dim=-1).mean()
                entropy_raw += entropy_t
                
        actor_loss = actor_loss / (valid_steps + 1e-8)
        entropy_raw = entropy_raw / (T + 1e-8)
        
        total_actor = actor_loss + self.entropy_coef * entropy_raw
        
        # 分开优化
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        
        self.actor_optimizer.zero_grad()
        total_actor.backward()
        self.actor_optimizer.step()
        
        return value_loss.item(), actor_loss.item(), entropy_raw.item(), advantage_rrca.mean().item(), len(regret_events)

    def step_train(self):
        trajectory = self.collect_session()
        c_loss, a_loss, ent_loss, adv, n_regret = self.get_session_a2c_loss(trajectory)
        pool_size = len(self.global_regret_pool['tokens'])
        print(f"Traj:{len(trajectory)} | C:{c_loss:.4f} | A:{a_loss:.4f} | Adv:{adv:.4f} | Regret:{n_regret} | Pool:{pool_size}")

# ==========================================
# 3. 执行测试
# ==========================================
if __name__ == "__main__":
    args = MockArgs()
    facade = MockFacade(batch_size=args.episode_batch_size)
    agent = SARA_Session_Agent(args, facade)
    
    print("=== RRCA + RAPI Session Agent 测试 ===")
    for i in range(10):
        print(f"Epoch {i+1}: ", end="")
        agent.step_train()
    print("=== 测试通过 ===")
