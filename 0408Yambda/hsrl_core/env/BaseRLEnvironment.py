from __future__ import annotations

from env import reward as reward_module


class BaseRLEnvironment:
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument("--env_path", type=str, required=True)
        parser.add_argument("--reward_func", type=str, default="mean_with_cost", help="reward function name")
        parser.add_argument("--max_step_per_episode", type=int, default=100)
        parser.add_argument("--initial_temper", type=int, required=100)
        return parser

    def __init__(self, args):
        self.env_path = args.env_path
        if not hasattr(reward_module, args.reward_func):
            raise ValueError(f"Unsupported reward_func in minimal core: {args.reward_func}")
        self.reward_func = getattr(reward_module, args.reward_func)
        self.max_step_per_episode = args.max_step_per_episode
        self.initial_temper = args.initial_temper

    def reset(self, params):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError
