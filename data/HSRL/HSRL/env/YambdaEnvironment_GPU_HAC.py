import numpy as np
import utils
import torch
import random
from copy import deepcopy
from argparse import Namespace
from torch.utils.data import DataLoader

from reader.RL4RSDataReader import RL4RSDataReader
from reader.YambdaDataReader import YambdaDataReader
from model.YambdaUserResponse import YambdaUserResponse
from env.BaseRLEnvironment import BaseRLEnvironment


class YambdaEnvironment_GPU_HAC(BaseRLEnvironment):
    """
    Yambda Environment for HAC agent.
    Modified from YambdaEnvironment_GPU to add previous_feedback in step return.
    This is required by HAC's extract_behavior_data() method.
    """
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
        - urm_log_path
        from BaseEnvironment
            - env_path
            - reward_func
            - max_step_per_episode
            - initial_temper
        '''
        parser = BaseRLEnvironment.parse_model_args(parser)
        parser.add_argument('--urm_log_path', type=str, required=True, help='log path for saved user response model')
        parser.add_argument('--temper_sweet_point', type=float, default=0.9, help='between [0,1.0]')
        parser.add_argument('--temper_prob_lag', type=float, default=100, help='smaller value means larger probability change from temper (min = 0)')
        return parser

    def __init__(self, args):
        # Temporary modification to avoid parent class eval error
        # Yambda handles direct_score in step function itself
        original_reward_func = args.reward_func
        if args.reward_func == 'direct_score':
            args.reward_func = 'mean_with_cost'  # Use an existing function to bypass parent check

        super(YambdaEnvironment_GPU_HAC, self).__init__(args)
        self.temper_sweet_point = args.temper_sweet_point
        self.temper_prob_lag = args.temper_prob_lag

        # Save reward_func name for distinguishing processing logic
        self.reward_func_name = original_reward_func

        infile = open(args.urm_log_path, 'r')
        class_args = eval(infile.readline()) # example: Namespace(model='YambdaUserResponse', reader='YambdaDataReader')
        model_args = eval(infile.readline()) # model parameters in Namespace
        print("Environment arguments: \n" + str(model_args))
        infile.close()
        print(f"Loading {class_args.reader} reader and {class_args.model} model")

        # Support both RL4RSDataReader and YambdaDataReader
        if class_args.reader == 'YambdaDataReader':
            self.reader = YambdaDataReader(model_args)
        elif class_args.reader == 'RL4RSDataReader':
            self.reader = RL4RSDataReader(model_args)
        else:
            raise ValueError(f"Unsupported reader: {class_args.reader}")
        print("Loading user response model")
        self.user_response_model = YambdaUserResponse(model_args, self.reader, args.device)
        self.user_response_model.load_from_checkpoint(model_args.model_path, with_optimizer = False)
        self.user_response_model.to(args.device)

        # spaces
        stats = self.reader.get_statistics()
        self.action_space = {'item_id': ('nominal', stats['n_item']),
                             'item_feature': ('continuous', stats['item_vec_size'], 'normal')}
        self.observation_space = {'user_profile': ('continuous', stats['user_portrait_len'], 'positive'),
                                  'history': ('sequence', stats['max_seq_len'], ('continuous', stats['item_vec_size']))}

    def reset(self, params = {'batch_size': 1, 'empty_history': True}):
        '''
        Reset environment with new sampled users
        @input:
        - params: {'batch_size': scalar,
                    'empty_history': True if start from empty history,
                    'initial_history': start with initial history, mu }
        @output:
        - observation: {'user_profile': (B, portrait_dim),
                        'history': (B, H),
                        'history_feature': (B, H, item_dim)
                        'history_feedback': (B, H, item_dim)}
        '''
        self.empty_history_flag = params['empty_history'] if 'empty_history' not in params else True
        BS = params['batch_size']
        observation = {'batch_size': BS}
        if 'sample' in params:
            sample_info = params['sample']
        else:
            self.iter = iter(DataLoader(self.reader, batch_size = BS, shuffle = True,
                                        pin_memory = True, num_workers = 0))
            sample_info = next(self.iter)
            sample_info = utils.wrap_batch(sample_info, device = self.user_response_model.device)
        self.current_observation = {
            'user_profile': sample_info['user_profile'],  # (B, user_dim)
            'history': sample_info['history'],  # (B, H)
            'history_features': sample_info['history_features'], # (B, H, item_dim)
            'cummulative_reward': torch.zeros(BS).to(self.user_response_model.device),
            'temper': torch.ones(BS).to(self.user_response_model.device) * self.initial_temper,
            'step': torch.zeros(BS).to(self.user_response_model.device),
        }
        self.reward_history = [0.]
        self.step_history = [0.]
        return deepcopy(self.current_observation)


    def step(self, step_dict):
        '''
        @input:
        - step_dict: {'action': (B, slate_size),
                        'action_features': (B, slate_size, item_dim) }
        '''
        # actions (exposures)
        action = step_dict['action'] # (B, slate_size), should be item ids only
        action_features = step_dict['action_features']
        batch_data = {
            'user_profile': self.current_observation['user_profile'],
            'history': self.current_observation['history'],  # Add history
            'history_features': self.current_observation['history_features'],
            'exposure_features': action_features
        }
        # URM forward
        with torch.no_grad():
            output_dict = self.user_response_model(batch_data)
            # YambdaUserResponse outputs continuous scalar preds (B, L)
            preds = output_dict['preds']

            # Determine reward function type
            if self.reward_func_name == 'direct_score':
                # direct_score: directly use model's continuous scalar as reward
                # preds shape: (B, L), average each slate to get (B,)
                immediate_reward = torch.mean(preds, dim=-1).detach()
                # For history update, also convert to binary feedback
                response = (preds > 0).float()  # (B, L)
            else:
                # Original logic: calculate reward based on click feedback
                # Convert continuous scalar to binary feedback (for existing reward_func)
                response = (preds > 0).float()  # (B, L)
                # reward (B,)
                immediate_reward = self.reward_func(response).detach()
                immediate_reward = -torch.abs(immediate_reward - self.temper_sweet_point) + 1

            # (B, H+slate_size)
            H_prime = torch.cat((self.current_observation['history'], action), dim = 1)
            # (B, H+slate_size, item_dim)
            H_prime_features = torch.cat((self.current_observation['history_features'], action_features), dim = 1)

            # Yambda minimal history update: regardless of user feedback, slide window forward
            # H_prime dimension is (B, H+slate_size), we directly截取最新的 max_seq_len 个物品
            self.current_observation['history'] = H_prime[:, -self.reader.max_seq_len:]
            self.current_observation['history_features'] = H_prime_features[:, -self.reader.max_seq_len:, :]

            self.current_observation['cummulative_reward'] += immediate_reward

            # temper update for leave model
            temper_down = (-immediate_reward+1) * response.shape[1] + 1
            self.current_observation['temper'] -= temper_down
            # leave signal
            done_mask = self.current_observation['temper'] < 1
            # step update
            self.current_observation['step'] += 1

            # update rows where user left
            if done_mask.sum() > 0:
                final_rewards = self.current_observation['cummulative_reward'][done_mask].detach().cpu().numpy()
                final_steps = self.current_observation['step'][done_mask].detach().cpu().numpy()
                self.reward_history.append(final_rewards[-1])
                self.step_history.append(final_steps[-1])
                # sample new users to fill in the blank
                new_sample_flag = False
                try:
                    sample_info = next(self.iter)
                    if sample_info['user_profile'].shape[0] != done_mask.shape[0]:
                        new_sample_flag = True
                except:
                    new_sample_flag = True
                if new_sample_flag:
                    self.iter = iter(DataLoader(self.reader, batch_size = done_mask.shape[0], shuffle = True,
                                                pin_memory = True, num_workers = 0))
                    sample_info = next(self.iter)
                sample_info = utils.wrap_batch(sample_info, device = self.user_response_model.device)
                for obs_key in ['user_profile', 'history', 'history_features']:
                    self.current_observation[obs_key][done_mask] = sample_info[obs_key][done_mask]
                self.current_observation['cummulative_reward'][done_mask] *= 0
                self.current_observation['temper'][done_mask] *= 0
                self.current_observation['temper'][done_mask] += self.initial_temper
            self.current_observation['step'][done_mask] *= 0

        # MODIFIED FOR HAC: Add previous_feedback for extract_behavior_data
        return deepcopy(self.current_observation), immediate_reward, done_mask, {'response': response, 'previous_feedback': response}

    def stop(self):
        self.iter = None

    def get_new_iterator(self, B):
        return iter(DataLoader(self.reader, batch_size = B, shuffle = True,
                               pin_memory = True, num_workers = 0))
