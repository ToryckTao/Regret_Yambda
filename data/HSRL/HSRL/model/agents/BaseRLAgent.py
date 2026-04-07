import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

import utils

class BaseRLAgent():
    @staticmethod
    def parse_model_args(parser):
        '''
        args:
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
        parser.add_argument('--gamma', type=float, default=0.95, 
                            help='reward discount')
        parser.add_argument('--n_iter', type=int, nargs='+', default=[2000], 
                            help='number of training iterations')
        parser.add_argument('--train_every_n_step', type=int, default=1, 
                            help='number of training iterations')
        parser.add_argument('--initial_greedy_epsilon', type=float, default=0.6, 
                            help='greedy probability for epsilon greedy exploration')
        parser.add_argument('--final_greedy_epsilon', type=float, default=0.05, 
                            help='greedy probability for epsilon greedy exploration')
        parser.add_argument('--elbow_greedy', type=float, default=0.5, 
                            help='greedy probability for epsilon greedy exploration')
        parser.add_argument('--check_episode', type=int, default=100, 
                            help='number of iterations to check output and evaluate')
        parser.add_argument('--with_eval', action='store_true',
                            help='do evaluation during training')
        parser.add_argument('--save_path', type=str, required=True, 
                            help='save path for networks')
        parser.add_argument('--use_wandb', action='store_true',
                            help='use wandb for logging')
        parser.add_argument('--wandb_project', type=str, default='yambda_ddpg',
                            help='wandb project name')
        parser.add_argument('--wandb_name', type=str, default=None,
                            help='wandb run name')
        return parser
    
    def __init__(self, args, facade):
        self.device = args.device
        self.gamma = args.gamma
        self.n_iter = [0] + args.n_iter
        self.train_every_n_step = args.train_every_n_step
        self.check_episode = args.check_episode
        self.with_eval = args.with_eval
        self.save_path = args.save_path
        self.facade = facade
        self.exploration_scheduler = utils.LinearScheduler(int(sum(args.n_iter) * args.elbow_greedy), 
                                                           args.final_greedy_epsilon, 
                                                           initial_p=args.initial_greedy_epsilon)
        
        # Wandb initialization
        self.use_wandb = getattr(args, 'use_wandb', False)
        self.wandb_project = getattr(args, 'wandb_project', 'yambda_ddpg')
        self.wandb_name = getattr(args, 'wandb_name', None)
        if self.use_wandb:
            import wandb
            wandb.init(project=self.wandb_project, name=self.wandb_name, dir=os.path.dirname(self.save_path))
            wandb.config.update(args)
        
        if len(self.n_iter) == 2:
            with open(self.save_path + ".report", 'w') as outfile:
                outfile.write(f"{args}\n")
    
    def train(self):
        if len(self.n_iter) > 2:
            self.load()
        
        t = time.time()
        
        # 检查是否断点续训（通过 continue_iter 属性）
        continue_iter = getattr(self, 'current_iter', 0)
        
        if continue_iter > 0:
            print(f"[CONTINUE] Skipping data preparation, starting from iteration {continue_iter}")
            self.facade.is_training_available = True
            self.facade.initialize_train()  # 只初始化buffer，不填充数据
            self.training_history = {"critic_loss": [], "actor_loss": []}
            observation = self.facade.reset_env({"batch_size": self.episode_batch_size})
            step_offset = continue_iter
        else:
            print("Run procedures before training")
            self.action_before_train()
            t = time.time()
            observation = self.facade.reset_env({"batch_size": self.episode_batch_size})
            step_offset = sum(self.n_iter[:-1])
        
        t = time.time()
        start_time = t
        # training
        print("Training:")
        for i in tqdm(range(step_offset, step_offset + self.n_iter[-1])):
            observation = self.run_episode_step(i, self.exploration_scheduler.value(i), observation, True)
            if i % self.train_every_n_step == 0:
                self.step_train()
            if i % self.check_episode == 0:
                t_ = time.time()
                print(f"Episode step {i}, time diff {t_ - t}, total time dif {t - start_time})")
                print(self.log_iteration(i))
                t = t_
                if i % (3*self.check_episode) == 0:
                    self.save()
        self.action_after_train()
        
    
    
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
        self.training_history = {"critic_loss": [], "actor_loss": []}
        
        print(f"Total {prepare_step} prepare steps")
    
    def action_after_train(self):
        self.facade.stop_env()
        if self.use_wandb:
            import wandb
            wandb.finish()
        
    def get_report(self):
        episode_report = self.facade.get_episode_report(10)
        train_report = {k: np.mean(v[-10:]) for k,v in self.training_history.items()}
        return episode_report, train_report
        
#     def run_an_episode(self, epsilon, initial_observation = None, with_train = False):
#         pass

    def run_episode_step(self, *episode_args):
        pass
    
    
    def step_train(self):
        pass
    
    
    def test(self):
        pass
    
    def log_iteration(self, step):
        episode_report, train_report = self.get_report()
        log_str = f"step: {step} @ episode report: {episode_report} @ step loss: {train_report}\n"
        with open(self.save_path + ".report", 'a') as outfile:
            outfile.write(log_str)
        
        # Wandb logging
        if self.use_wandb:
            import wandb
            log_dict = {'step': step}
            # Add episode report metrics
            for k, v in episode_report.items():
                log_dict[f'episode/{k}'] = v
            # Add train report metrics
            for k, v in train_report.items():
                log_dict[f'train/{k}'] = v
            wandb.log(log_dict)
        
        return log_str
    
    def save(self):
        pass
    
    def load(self):
        pass
    