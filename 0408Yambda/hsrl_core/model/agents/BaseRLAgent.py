from __future__ import annotations

import os
import time

import numpy as np
from tqdm import tqdm

import utils


class BaseRLAgent:
    @staticmethod
    def parse_model_args(parser):
        parser.add_argument("--gamma", type=float, default=0.95)
        parser.add_argument("--n_iter", type=int, nargs="+", default=[2000])
        parser.add_argument("--train_every_n_step", type=int, default=1)
        parser.add_argument("--initial_greedy_epsilon", type=float, default=0.6)
        parser.add_argument("--final_greedy_epsilon", type=float, default=0.05)
        parser.add_argument("--elbow_greedy", type=float, default=0.5)
        parser.add_argument("--check_episode", type=int, default=100)
        parser.add_argument("--with_eval", action="store_true")
        parser.add_argument("--save_path", type=str, required=True)
        parser.add_argument("--use_wandb", action="store_true")
        parser.add_argument("--wandb_project", type=str, default="yambda_ddpg")
        parser.add_argument("--wandb_name", type=str, default=None)
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
        self.exploration_scheduler = utils.LinearScheduler(
            int(sum(args.n_iter) * args.elbow_greedy),
            args.final_greedy_epsilon,
            initial_p=args.initial_greedy_epsilon,
        )
        self.use_wandb = getattr(args, "use_wandb", False)
        self.wandb_project = getattr(args, "wandb_project", "yambda_ddpg")
        self.wandb_name = getattr(args, "wandb_name", None)
        if self.use_wandb:
            import wandb

            wandb.init(project=self.wandb_project, name=self.wandb_name, dir=os.path.dirname(self.save_path))
            wandb.config.update(args)

        if len(self.n_iter) == 2:
            with open(self.save_path + ".report", "w", encoding="utf-8") as outfile:
                outfile.write(f"{args}\n")

    def train(self):
        if len(self.n_iter) > 2:
            self.load()

        continue_iter = getattr(self, "current_iter", 0)
        if continue_iter > 0:
            print(f"[CONTINUE] Skipping data preparation, starting from iteration {continue_iter}")
            self.facade.is_training_available = True
            self.facade.initialize_train()
            self.training_history = {"critic_loss": [], "actor_loss": []}
            observation = self.facade.reset_env({"batch_size": self.episode_batch_size})
            step_offset = continue_iter
        else:
            print("Run procedures before training")
            self.action_before_train()
            observation = self.facade.reset_env({"batch_size": self.episode_batch_size})
            step_offset = sum(self.n_iter[:-1])

        t = time.time()
        start_time = t
        print("Training:")
        pbar = tqdm(total=step_offset + self.n_iter[-1], desc="[DDPG]", ncols=80, initial=step_offset)
        for i in range(step_offset, step_offset + self.n_iter[-1]):
            observation = self.run_episode_step(i, self.exploration_scheduler.value(i), observation, True)
            if i % self.train_every_n_step == 0:
                self.step_train()
            if i % self.check_episode == 0:
                t_now = time.time()
                pbar.set_postfix_str(self._make_short_report(i), refresh=False)
                print(f"Episode step {i}, time diff {t_now - t}, total time dif {t - start_time})")
                print(self.log_iteration(i))
                t = t_now
                if i % (3 * self.check_episode) == 0:
                    self.save()
            pbar.update(1)
            pbar.set_postfix_str(self._make_short_report(i), refresh=False)
        pbar.close()
        self.action_after_train()

    def action_before_train(self):
        self.facade.initialize_train()
        prepare_step = 0
        observation = self.facade.reset_env({"batch_size": self.episode_batch_size})
        while not self.facade.is_training_available:
            observation = self.run_episode_step(0, 1.0, observation, True)
            prepare_step += 1
        self.training_history = {"critic_loss": [], "actor_loss": []}
        print(f"Total {prepare_step} prepare steps")

    def action_after_train(self):
        self.facade.stop_env()
        if self.use_wandb:
            import wandb

            wandb.finish()

    def get_report(self):
        episode_report = self.facade.get_episode_report(10)
        train_report = {key: np.mean(value[-10:]) for key, value in self.training_history.items()}
        return episode_report, train_report

    def _make_short_report(self, step: int) -> str:
        """构造 tqdm postfix 短报告。"""
        if not self.training_history:
            return ""
        parts = []
        for key in ["actor_loss", "critic_loss"]:
            if key in self.training_history and self.training_history[key]:
                parts.append(f"{key}={self.training_history[key][-1]:.4f}")
        episode_report, _ = self.get_report()
        if "average_total_reward" in episode_report:
            parts.append(f"reward={episode_report['average_total_reward']:.3f}")
        if "average_n_step" in episode_report:
            parts.append(f"step={episode_report['average_n_step']:.1f}")
        return " ".join(parts)

    def log_iteration(self, step):
        episode_report, train_report = self.get_report()
        log_str = f"step: {step} @ episode report: {episode_report} @ step loss: {train_report}\n"
        with open(self.save_path + ".report", "a", encoding="utf-8") as outfile:
            outfile.write(log_str)
        if self.use_wandb:
            import wandb

            log_dict = {"step": step}
            for key, value in episode_report.items():
                log_dict[f"episode/{key}"] = value
            for key, value in train_report.items():
                log_dict[f"train/{key}"] = value
            wandb.log(log_dict)
        return log_str

    def run_episode_step(self, *episode_args):
        raise NotImplementedError

    def step_train(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError
