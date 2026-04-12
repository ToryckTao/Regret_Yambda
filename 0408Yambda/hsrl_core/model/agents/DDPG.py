from __future__ import annotations

import copy

import torch

from model.agents.BaseRLAgent import BaseRLAgent


class DDPG(BaseRLAgent):
    """
    最小 DDPG 基类。

    Yambda 的具体 step_train/get_ddpg_loss 由 adapter/model/agents/DDPG.py 覆盖。
    """

    @staticmethod
    def parse_model_args(parser):
        parser = BaseRLAgent.parse_model_args(parser)
        parser.add_argument("--episode_batch_size", type=int, default=8)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--actor_lr", type=float, default=1e-4)
        parser.add_argument("--critic_lr", type=float, default=1e-4)
        parser.add_argument("--actor_decay", type=float, default=1e-4)
        parser.add_argument("--critic_decay", type=float, default=1e-4)
        parser.add_argument("--target_mitigate_coef", type=float, default=0.01)
        parser.add_argument("--entropy_coef", type=float, default=0.01)
        parser.add_argument("--bc_coef", type=float, default=0.1)
        return parser

    def __init__(self, args, facade):
        super().__init__(args, facade)
        self.episode_batch_size = args.episode_batch_size
        self.batch_size = args.batch_size
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.actor_decay = args.actor_decay
        self.critic_decay = args.critic_decay

        self.actor = facade.actor
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=args.actor_lr,
            weight_decay=args.actor_decay,
        )

        self.critic = facade.critic
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=args.critic_lr,
            weight_decay=args.critic_decay,
        )

        self.tau = args.target_mitigate_coef
        self.entropy_coef = getattr(args, "entropy_coef", 0.01)
        self.bc_coef = getattr(args, "bc_coef", 0.1)

    def action_before_train(self):
        self.facade.initialize_train()
        prepare_step = 0
        observation = self.facade.reset_env({"batch_size": self.episode_batch_size})
        while not self.facade.is_training_available:
            observation = self.run_episode_step(0, 1.0, observation, True)
            prepare_step += 1
        self.training_history = {
            "critic_loss": [],
            "actor_loss": [],
            "entropy_loss": [],
            "bc_loss": [],
            "advantage": [],
        }
        print(f"Total {prepare_step} prepare steps")

    def run_episode_step(self, *episode_args):
        episode_iter, epsilon, observation, do_buffer_update = episode_args
        with torch.no_grad():
            policy_output = self.facade.apply_policy(observation, self.actor, epsilon, do_explore=True)
            next_observation, reward, done, info = self.facade.env_step(policy_output)
            if do_buffer_update:
                self.facade.update_buffer(observation, policy_output, reward, done, next_observation, info)
        return next_observation

    def step_train(self):
        raise NotImplementedError("Yambda DDPG adapter overrides step_train().")

    def get_ddpg_loss(self, *args, **kwargs):
        raise NotImplementedError("Yambda DDPG adapter overrides get_ddpg_loss().")

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
