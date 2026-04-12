from __future__ import annotations

import numpy as np


class SIDFacade_credit:
    """
    最小 SIDFacade 基类。

    Yambda candidate scoring / replay buffer 由 adapter/model/facade/SIDFacade_credit.py 覆盖。
    """

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument("--slate_size", type=int, default=6)
        parser.add_argument("--buffer_size", type=int, default=10000)
        parser.add_argument("--start_timestamp", type=int, default=1000)
        parser.add_argument("--noise_var", type=float, default=0)
        parser.add_argument("--q_laplace_smoothness", type=float, default=0.5)
        parser.add_argument("--topk_rate", type=float, default=1.0)
        parser.add_argument("--empty_start_rate", type=float, default=0)
        parser.add_argument("--item2sid", type=str, default="dataset/rl4rs/sid_index_item2sid.pkl")
        return parser

    def reset_env(self, initial_params={"batch_size": 1}):
        initial_params["empty_history"] = True if np.random.rand() < self.empty_start_rate else False
        return self.env.reset(initial_params)

    def env_step(self, policy_output):
        action_dict = {
            "action": policy_output["action"],
            "action_features": policy_output["action_features"],
        }
        return self.env.step(action_dict)

    def stop_env(self):
        self.env.stop()

    def get_episode_report(self, n_recent=10):
        recent_rewards = self.env.reward_history[-n_recent:]
        recent_steps = self.env.step_history[-n_recent:]
        return {
            "average_total_reward": np.mean(recent_rewards),
            "reward_variance": np.var(recent_rewards),
            "max_total_reward": np.max(recent_rewards),
            "min_total_reward": np.min(recent_rewards),
            "average_n_step": np.mean(recent_steps),
            "max_n_step": np.max(recent_steps),
            "min_n_step": np.min(recent_steps),
            "buffer_size": self.current_buffer_size,
        }
