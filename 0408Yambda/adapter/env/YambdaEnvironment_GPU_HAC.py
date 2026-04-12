import torch
import utils
from argparse import Namespace
from copy import deepcopy
from torch.utils.data import DataLoader

from env.BaseRLEnvironment import BaseRLEnvironment
from model.YambdaUserResponse import YambdaUserResponse
from reader.YambdaFeatureCacheReader import YambdaFeatureCacheReader


class YambdaEnvironment_GPU_HAC(BaseRLEnvironment):
    """
    Yambda 环境适配版。

    相对原 HSRL 版本的变化：
    - 支持 YambdaFeatureCacheReader
    - direct_score 直接使用 UserResponse 的连续预测值作为 reward
    - step 返回 previous_feedback，供 HAC/SID facade 写入 replay buffer
    """

    @staticmethod
    def parse_model_args(parser):
        parser = BaseRLEnvironment.parse_model_args(parser)
        parser.add_argument("--urm_log_path", type=str, required=True, help="log path for saved user response model")
        parser.add_argument("--temper_sweet_point", type=float, default=0.9, help="between [0,1.0]")
        parser.add_argument("--temper_prob_lag", type=float, default=100, help="smaller value means larger probability change")
        return parser

    def __init__(self, args):
        original_reward_func = args.reward_func
        if args.reward_func == "direct_score":
            args.reward_func = "mean_with_cost"

        super(YambdaEnvironment_GPU_HAC, self).__init__(args)
        self.temper_sweet_point = args.temper_sweet_point
        self.temper_prob_lag = args.temper_prob_lag
        self.reward_func_name = original_reward_func

        with open(args.urm_log_path, "r", encoding="utf-8") as infile:
            eval_scope = {"__builtins__": {}, "Namespace": Namespace}
            class_args = eval(infile.readline(), eval_scope)
            model_args = eval(infile.readline(), eval_scope)
        print("Environment arguments: \n" + str(model_args))
        print(f"Loading {class_args.reader} reader and {class_args.model} model")

        if class_args.reader == "YambdaFeatureCacheReader":
            self.reader = YambdaFeatureCacheReader(model_args)
        else:
            raise ValueError(
                f"Unsupported reader for standalone 0408Yambda baseline: {class_args.reader}. "
                "Expected YambdaFeatureCacheReader."
            )

        print("Loading user response model")
        self.user_response_model = YambdaUserResponse(model_args, self.reader, args.device)
        self.user_response_model.load_from_checkpoint(model_args.model_path, with_optimizer=False)
        self.user_response_model.to(args.device)

        stats = self.reader.get_statistics()
        self.action_space = {
            "item_id": ("nominal", stats["n_item"]),
            "item_feature": ("continuous", stats["item_vec_size"], "normal"),
        }
        self.observation_space = {
            "user_profile": ("continuous", stats["user_portrait_len"], "positive"),
            "history": ("sequence", stats["max_seq_len"], ("continuous", stats["item_vec_size"])),
        }

    def reset(self, params={"batch_size": 1, "empty_history": True}):
        """
        输入：
        - params["batch_size"]: batch size

        输出：
        - observation，包含 history / history_features 等环境状态
        """
        self.empty_history_flag = params["empty_history"] if "empty_history" in params else True
        batch_size = params["batch_size"]
        if "sample" in params:
            sample_info = params["sample"]
        else:
            self.iter = iter(DataLoader(self.reader, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0))
            sample_info = next(self.iter)
            sample_info = utils.wrap_batch(sample_info, device=self.user_response_model.device)

        self.current_observation = {
            "user_profile": sample_info["user_profile"],
            "history": sample_info["history"],
            "history_features": sample_info["history_features"],
            "cummulative_reward": torch.zeros(batch_size).to(self.user_response_model.device),
            "temper": torch.ones(batch_size).to(self.user_response_model.device) * self.initial_temper,
            "step": torch.zeros(batch_size).to(self.user_response_model.device),
        }
        self.reward_history = [0.0]
        self.step_history = [0.0]
        return deepcopy(self.current_observation)

    def step(self, step_dict):
        """
        输入：
        - step_dict["action"]: [B, slate_size]
        - step_dict["action_features"]: [B, slate_size, item_dim]

        输出：
        - next_observation, immediate_reward, done_mask, info
        """
        action = step_dict["action"]
        action_features = step_dict["action_features"]
        batch_data = {
            "user_profile": self.current_observation["user_profile"],
            "history": self.current_observation["history"],
            "history_features": self.current_observation["history_features"],
            "exposure_features": action_features,
        }

        with torch.no_grad():
            output_dict = self.user_response_model(batch_data)
            preds = output_dict["preds"]

            if self.reward_func_name == "direct_score":
                immediate_reward = torch.mean(preds, dim=-1).detach()
                response = (preds > 0).float()
            else:
                response = (preds > 0).float()
                immediate_reward = self.reward_func(response).detach()
                immediate_reward = -torch.abs(immediate_reward - self.temper_sweet_point) + 1

            h_prime = torch.cat((self.current_observation["history"], action), dim=1)
            h_prime_features = torch.cat((self.current_observation["history_features"], action_features), dim=1)
            self.current_observation["history"] = h_prime[:, -self.reader.max_seq_len:]
            self.current_observation["history_features"] = h_prime_features[:, -self.reader.max_seq_len:, :]

            self.current_observation["cummulative_reward"] += immediate_reward
            temper_down = (-immediate_reward + 1) * response.shape[1] + 1
            self.current_observation["temper"] -= temper_down
            done_mask = self.current_observation["temper"] < 1
            self.current_observation["step"] += 1

            if done_mask.sum() > 0:
                final_rewards = self.current_observation["cummulative_reward"][done_mask].detach().cpu().numpy()
                final_steps = self.current_observation["step"][done_mask].detach().cpu().numpy()
                self.reward_history.append(final_rewards[-1])
                self.step_history.append(final_steps[-1])

                new_sample_flag = False
                try:
                    sample_info = next(self.iter)
                    if sample_info["user_profile"].shape[0] != done_mask.shape[0]:
                        new_sample_flag = True
                except Exception:
                    new_sample_flag = True
                if new_sample_flag:
                    self.iter = iter(
                        DataLoader(self.reader, batch_size=done_mask.shape[0], shuffle=True, pin_memory=True, num_workers=0)
                    )
                    sample_info = next(self.iter)
                sample_info = utils.wrap_batch(sample_info, device=self.user_response_model.device)
                for obs_key in ["user_profile", "history", "history_features"]:
                    self.current_observation[obs_key][done_mask] = sample_info[obs_key][done_mask]
                self.current_observation["cummulative_reward"][done_mask] *= 0
                self.current_observation["temper"][done_mask] *= 0
                self.current_observation["temper"][done_mask] += self.initial_temper
            self.current_observation["step"][done_mask] *= 0

        return deepcopy(self.current_observation), immediate_reward, done_mask, {
            "response": response,
            "previous_feedback": response,
        }

    def stop(self):
        self.iter = None

    def get_new_iterator(self, batch_size):
        return iter(DataLoader(self.reader, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0))
