#!/usr/bin/env python3
"""
SARA 快速测试脚本
使用模拟数据快速验证代码流程
"""

import torch
import numpy as np
import argparse
import os
import sys

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.agents import BehaviorDDPG
from model.policy import SIDPolicy_credit
from model.critic import Token_Critic
from model.facade import SARAFacade_credit


class MockReader:
    """模拟数据读取器"""
    def __init__(self, n_user=100, n_item=1000, max_seq_len=50, portrait_len=32, item_vec_size=64):
        self.n_user = n_user
        self.n_item = n_item
        self.max_seq_len = max_seq_len
        self.portrait_len = portrait_len
        self.item_vec_size = item_vec_size
        
    def get_statistics(self):
        return {'n_user': self.n_user, 'n_item': self.n_item, 'max_seq_len': self.max_seq_len, 'item_vec_size': self.item_vec_size}
        
    def __iter__(self):
        return self
    
    def __next__(self):
        batch = {
            'user_profile': torch.randn(32, self.portrait_len),
            'history': torch.randint(1, self.n_item + 1, (32, self.max_seq_len)),
            'history_features': torch.randn(32, self.max_seq_len, self.item_vec_size),
        }
        raise StopIteration
    
    def __len__(self):
        return self.n_user
    
    def get_item_list_meta(self, item_ids):
        # 返回 item embedding
        n = len(item_ids) if hasattr(item_ids, '__len__') else 1
        return torch.randn(n, self.item_vec_size)


class MockUserResponseModel:
    """模拟用户响应模型"""
    def __init__(self, device='cpu'):
        self.device = device
        # 生成 [0, 1] 范围内的反馈
        self.predict = lambda x: torch.rand(x['history'].shape[0], 6, device=self.device)


class MockEnvironment:
    """模拟环境"""
    def __init__(self, args):
        self.action_space = {'item_id': ('nominal', 1000), 'item_feature': ('continuous', 64)}
        self.observation_space = {'history': ('sequence', 50, ('continuous', 64))}
        self.reader = MockReader()
        self.user_response_model = MockUserResponseModel()
        self.args = args
        self.args.item_vec_size = 64  # 确保有这个属性
        self.reward_history = [0.]  # 添加初始值
        self.step_history = [0.]  # 添加初始值
    
    def stop(self):
        pass  # Mock 环境不需要停止
        
    def reset(self, params={'batch_size': 1}):
        BS = params.get('batch_size', 1)
        return {
            'user_profile': torch.randn(BS, 32),
            'history': torch.randint(1, 1001, (BS, 50)),
            'history_features': torch.randn(BS, 50, 64),  # [batch, seq, vec]
        }
    
    def step(self, action_dict):
        batch_size = action_dict['action'].shape[0]
        obs = {
            'user_profile': torch.randn(batch_size, 32),
            'history': torch.randint(1, 1001, (batch_size, 50)),
            'history_features': torch.randn(batch_size, 50, 64),
        }
        reward = torch.randn(batch_size)
        done = torch.zeros(batch_size, dtype=torch.bool)
        # 修复: 确保 feedback 在 [0, 1] 范围内
        info = {'response': torch.rand(batch_size, 6), 'preds': torch.rand(batch_size, 6)}
        return obs, reward, done, info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--slate_size', type=int, default=6)
    parser.add_argument('--buffer_size', type=int, default=1000)
    parser.add_argument('--start_timestamp', type=int, default=100)
    parser.add_argument('--n_iter', type=int, default=10)
    parser.add_argument('--episode_batch_size', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_step_per_episode', type=int, default=10)
    parser.add_argument('--actor_lr', type=float, default=0.0001)
    parser.add_argument('--critic_lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--sid_levels', type=int, default=3)
    parser.add_argument('--sid_vocab_sizes', type=int, default=256)
    parser.add_argument('--item2sid', type=str, default=None)
    
    # Facade 需要的额外参数
    parser.add_argument('--noise_var', type=float, default=0.1)
    parser.add_argument('--q_laplace_smoothness', type=float, default=0.1)
    parser.add_argument('--topk_rate', type=float, default=0.0)
    parser.add_argument('--empty_start_rate', type=float, default=0.0)
    parser.add_argument('--train_every_n_step', type=int, default=1)
    
    # Agent 需要的参数
    parser.add_argument('--check_episode', type=int, default=10)
    parser.add_argument('--with_eval', type=bool, default=False)
    parser.add_argument('--save_path', type=str, default='output/test_model')
    parser.add_argument('--initial_greedy_epsilon', type=float, default=0.0)
    parser.add_argument('--final_greedy_epsilon', type=float, default=0.0)
    parser.add_argument('--elbow_greedy', type=float, default=0.5)
    parser.add_argument('--actor_decay', type=float, default=0.0)
    parser.add_argument('--critic_decay', type=float, default=0.0)
    parser.add_argument('--target_mitigate_coef', type=float, default=0.0)
    
    # BehaviorDDPG 需要的参数
    parser.add_argument('--behavior_lr', type=float, default=0.00001)
    parser.add_argument('--behavior_decay', type=float, default=0.0)
    
    # Critic 需要的参数
    parser.add_argument('--critic_hidden_dims', type=int, nargs='+', default=[128, 64])
    parser.add_argument('--critic_dropout_rate', type=float, default=0.2)
    
    args = parser.parse_args()
    args.n_iter = [args.n_iter]  # 转为列表
    args.critic_hidden_dims = [128, 64]
    args.device = 'cpu'  # 添加 device 属性
    
    # 创建模拟环境
    print("Creating mock environment...")
    env = MockEnvironment(args)
    
    # 创建 mock item2sid
    n_item = 1000
    item2sid = {i: tuple(np.random.randint(0, 256, 3).tolist()) for i in range(1, n_item + 1)}
    
    # 创建一个临时文件保存 mock item2sid
    import tempfile
    import pickle
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
        pickle.dump(item2sid, f)
        mock_item2sid_path = f.name
    
    args.item2sid = mock_item2sid_path  # 使用临时文件
    
    print("Creating policy...")
    # Mock policy 参数
    args.sasrec_n_layer = 2
    args.sasrec_d_model = 32
    args.sasrec_d_forward = 128
    args.sasrec_n_head = 4
    args.sasrec_dropout = 0.1
    args.sid_temp = 1.0
    args.sid_levels = 3
    args.sid_vocab_sizes = 256  # 必须是整数，policy内部会扩展为列表
    
    policy = SIDPolicy_credit(args, env)
    
    print("Creating critic...")
    args.critic_hidden_dims = [128, 64]
    args.critic_dropout_rate = 0.2
    critic = Token_Critic(args, env, policy)
    
    print("Creating facade...")
    facade = SARAFacade_credit(args, env, policy, critic)
    
    print("Creating agent...")
    agent = BehaviorDDPG(args, facade)
    
    print("Starting training...")
    agent.train()
    
    print("Done!")


if __name__ == '__main__':
    main()
