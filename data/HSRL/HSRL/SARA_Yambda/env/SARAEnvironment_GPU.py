"""
SARA Environment - 极简高效版 (移除内部冗余状态，修复 DataLoader 内存泄漏)
核心改动：
1. 奖励函数: r_t = ω_listen * played_ratio + ω_like * I_like - ω_dislike * I_dislike
2. 合成延迟反悔: 仅在 info 中返回 sara_delta_t 和 sara_unlike_prob，由 Agent 负责埋雷和引爆
3. 性能修复: 所有 DataLoader 的 num_workers 设置为 0，防止多进程炸弹拖死 CPU
"""

import numpy as np
import utils
import torch
import random
from copy import deepcopy
from argparse import Namespace
from torch.utils.data import DataLoader

from reader.YambdaDataReader import YambdaDataReader
from model.YambdaUserResponse import YambdaUserResponse
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from env.BaseRLEnvironment import BaseRLEnvironment


class SARAEnvironment_GPU(BaseRLEnvironment):
    @staticmethod
    def parse_model_args(parser):
        parser = BaseRLEnvironment.parse_model_args(parser)
        parser.add_argument('--urm_log_path', type=str, required=True, help='log path for saved user response model')
        parser.add_argument('--temper_sweet_point', type=float, default=0.9, help='between [0,1.0]')
        parser.add_argument('--temper_prob_lag', type=float, default=100, help='smaller value means larger prob change')

        # SARA 新增参数
        parser.add_argument('--omega_listen', type=float, default=1.0, help='played_ratio 权重')
        parser.add_argument('--omega_like', type=float, default=2.0, help='like 权重')
        parser.add_argument('--omega_dislike', type=float, default=1.0, help='dislike 权重')
        parser.add_argument('--lambda_unlike', type=float, default=1.0, help='unlike 惩罚权重')
        parser.add_argument('--lambda_undislike', type=float, default=0.5, help='undislike 奖励权重')
        parser.add_argument('--regret_gamma', type=float, default=0.9, help='时间折扣因子 γ')
        return parser

    def __init__(self, args):
        original_reward_func = args.reward_func
        if args.reward_func == 'direct_score':
            args.reward_func = 'mean_with_cost'

        super(SARAEnvironment_GPU, self).__init__(args)
        self.temper_sweet_point = args.temper_sweet_point
        self.temper_prob_lag = args.temper_prob_lag
        self.reward_func_name = original_reward_func

        self.omega_listen = args.omega_listen
        self.omega_like = args.omega_like
        self.omega_dislike = args.omega_dislike
        self.lambda_unlike = args.lambda_unlike
        self.lambda_undislike = args.lambda_undislike
        self.regret_gamma = args.regret_gamma

        import torch
        infile = open(args.urm_log_path, 'r')
        lines = infile.readlines()
        infile.close()
        
        if len(lines) >= 2 and lines[0].strip() and lines[1].strip():
            class_args = eval(lines[0])
            model_args = eval(lines[1])
        else:
            print("Warning: log file incomplete, loading params from checkpoint...")
            checkpoint_path = args.urm_log_path.replace('log/', '').replace('.model.log', '.model')
            ckpt = torch.load(checkpoint_path + '.checkpoint', map_location='cpu')
            if 'args' in ckpt:
                model_args = ckpt['args']
            else:
                model_args = Namespace(
                    model='YambdaUserResponse', reader='YambdaDataReader',
                    feature_dim=16, hidden_dims=[256], attn_n_head=2, dropout_rate=0.2,
                    max_seq_len=50, l2_coef=0.0001,
                    train_file='/root/autodl-tmp/data/HSRL/dataset/processed/debug_train.tsv',
                    val_file='/root/autodl-tmp/data/HSRL/dataset/processed/debug_val.tsv',
                    test_file='/root/autodl-tmp/data/HSRL/dataset/processed/debug_test.tsv',
                    item_meta_file='/root/autodl-tmp/data/HSRL/dataset/processed/item_meta.tsv',
                    data_separator='\t', meta_data_separator='\t',
                    model_path=checkpoint_path, n_worker=0  # FIX 1: num_workers=0
                )
            class_args = Namespace(model='YambdaUserResponse', reader='YambdaDataReader')
        
        print("Loading raw data")
        self.reader = YambdaDataReader(model_args)
        print("Loading user response model")
        self.user_response_model = YambdaUserResponse(model_args, self.reader, args.device)
        self.user_response_model.load_from_checkpoint(model_args.model_path, with_optimizer=False)
        self.user_response_model.to(args.device)

        stats = self.reader.get_statistics()
        self.action_space = {'item_id': ('nominal', stats['n_item']),
                             'item_feature': ('continuous', stats['item_vec_size'], 'normal')}
        self.observation_space = {'user_profile': ('continuous', stats['user_portrait_len'], 'positive'),
                                  'history': ('sequence', stats['max_seq_len'], ('continuous', stats['item_vec_size']))}

    def reset(self, params={'batch_size': 1, 'empty_history': True}):
        self.empty_history_flag = params['empty_history'] if 'empty_history' not in params else True
        BS = params['batch_size']
        if 'sample' in params:
            sample_info = params['sample']
        else:
            # FIX 2: 彻底禁止这里开启多进程 (num_workers=0)
            self.iter = iter(DataLoader(self.reader, batch_size=BS, shuffle=True, pin_memory=True, num_workers=0))
            sample_info = next(self.iter)
            sample_info = utils.wrap_batch(sample_info, device=self.user_response_model.device)

        # FIX 3: 移除无用的 user_ids 和 pending_regrets 内部状态，环境必须保持干净
        self.current_observation = {
            'user_profile': sample_info['user_profile'],  
            'history': sample_info['history'],  
            'history_features': sample_info['history_features'], 
            'cummulative_reward': torch.zeros(BS).to(self.user_response_model.device),
            'temper': torch.ones(BS).to(self.user_response_model.device) * self.initial_temper,
            'step': torch.zeros(BS).to(self.user_response_model.device),
        }
        self.reward_history = [0.]
        self.step_history = [0.]
        return deepcopy(self.current_observation)

    def _compute_sara_reward(self, response, preds, action_item_ids):
        played_ratio = torch.sigmoid(preds) 
        like_threshold = 0.5
        dislike_threshold = -0.5
        I_like = (preds > like_threshold).float() 
        I_dislike = (preds < dislike_threshold).float() 
        
        reward = (
            self.omega_listen * torch.mean(played_ratio, dim=-1) +
            self.omega_like * torch.mean(I_like, dim=-1) -
            self.omega_dislike * torch.mean(I_dislike, dim=-1)
        )
        return reward, None

    def step(self, step_dict):
        action = step_dict['action'] 
        action_features = step_dict['action_features']
        batch_data = {
            'user_profile': self.current_observation['user_profile'],
            'history': self.current_observation['history'],
            'history_features': self.current_observation['history_features'],
            'exposure_features': action_features
        }

        with torch.no_grad():
            output_dict = self.user_response_model(batch_data)
            preds = output_dict['preds']

            # ========== SARA: 预埋延迟反悔(抛出定时炸弹) ==========
            B, slate_size = preds.shape
            sara_delta_t = torch.zeros((B, slate_size), dtype=torch.long, device=preds.device)
            sara_unlike_prob = torch.zeros((B, slate_size), device=preds.device)

            positive_mask = (preds > 0)
            if positive_mask.any():
                sara_unlike_prob[positive_mask] = torch.exp(-preds[positive_mask] * 2.0)
                regret_happens = torch.bernoulli(sara_unlike_prob).bool()
                if regret_happens.any():
                    sampled_delays = torch.poisson(torch.ones_like(preds[regret_happens]) * 4.0) + 1
                    sara_delta_t[regret_happens] = sampled_delays.long()
            # =================================================

            if self.reward_func_name in ['direct_score', 'sara_reward']:
                immediate_reward, _ = self._compute_sara_reward(None, preds, action)
                immediate_reward = immediate_reward.detach()
                response = (preds > 0).float()
            else:
                response = (preds > 0).float()
                immediate_reward = self.reward_func(response).detach()
                immediate_reward = -torch.abs(immediate_reward - self.temper_sweet_point) + 1

            H_prime = torch.cat((self.current_observation['history'], action), dim=1)
            H_prime_features = torch.cat((self.current_observation['history_features'], action_features), dim=1)

            self.current_observation['history'] = H_prime[:, -self.reader.max_seq_len:]
            self.current_observation['history_features'] = H_prime_features[:, -self.reader.max_seq_len:, :]
            self.current_observation['cummulative_reward'] += immediate_reward

            temper_down = (-immediate_reward+1) * response.shape[1] + 1
            self.current_observation['temper'] -= temper_down
            done_mask = self.current_observation['temper'] < 1
            self.current_observation['step'] += 1

            if done_mask.sum() > 0:
                final_rewards = self.current_observation['cummulative_reward'][done_mask].detach().cpu().numpy()
                final_steps = self.current_observation['step'][done_mask].detach().cpu().numpy()
                self.reward_history.append(final_rewards[-1])
                self.step_history.append(final_steps[-1])
                
                new_sample_flag = False
                try:
                    sample_info = next(self.iter)
                    if sample_info['user_profile'].shape[0] != done_mask.shape[0]:
                        new_sample_flag = True
                except:
                    new_sample_flag = True
                    
                if new_sample_flag:
                    # FIX 4: 再次确保这里替换死掉的用户时，绝不拉起多进程！
                    self.iter = iter(DataLoader(self.reader, batch_size=done_mask.shape[0], shuffle=True,
                                                pin_memory=True, num_workers=0))
                    sample_info = next(self.iter)
                
                sample_info = utils.wrap_batch(sample_info, device=self.user_response_model.device)
                for obs_key in ['user_profile', 'history', 'history_features']:
                    self.current_observation[obs_key][done_mask] = sample_info[obs_key][done_mask]
                self.current_observation['cummulative_reward'][done_mask] *= 0
                self.current_observation['temper'][done_mask] = self.initial_temper
                self.current_observation['step'][done_mask] *= 0

        # SARA 核心：环境只负责抛出信息，不管后续记录！
        return deepcopy(self.current_observation), immediate_reward, done_mask, {
            'response': response,
            'preds': preds,
            'sara_delta_t': sara_delta_t,
            'sara_unlike_prob': sara_unlike_prob
        }

    def stop(self):
        self.iter = None

    def get_new_iterator(self, B):
        return iter(DataLoader(self.reader, batch_size=B, shuffle=True, pin_memory=True, num_workers=0))