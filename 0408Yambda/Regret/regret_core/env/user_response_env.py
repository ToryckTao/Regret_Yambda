from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from regret_core.data.schema import EVENT_TYPE_TO_ID, ID_TO_EVENT_TYPE
from regret_core.data.transition_dataset import TransitionIterableDataset
from regret_core.model.user_response import RegretUserResponse


class RegretUserResponseEnv:
    """Yambda simulator environment backed by a trained RegretUserResponse.

    The wrapper follows the session-run transition semantics: each policy action
    is treated as one recommended item step, the simulator predicts behavior
    components, derives reward via the model's rule-based reward path, and then
    appends a synthetic step-level feedback signal to the history.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        transition_path: str | Path,
        dense_item_features_npy: str | Path | None = None,
        max_seq_len: int | None = None,
        device: str = "cpu",
        max_step_per_episode: int = 10,
        sample_response: bool = True,
        negative_patience: int = 5,
        reward_done_threshold: float | None = None,
        seed: int = 42,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        model_args = dict(ckpt["model_args"])
        # A simulator should predict behavior components and let the known reward
        # rule compose the scalar reward, not directly hallucinate reward.
        model_args.setdefault("decouple_reward_model", True)
        self.max_seq_len = int(max_seq_len or ckpt["data_args"]["max_seq_len"])
        self.feature_path = str(dense_item_features_npy or ckpt["data_args"]["dense_item_features_npy"])
        self.features = np.load(self.feature_path, mmap_mode="r")
        self.model = RegretUserResponse(**model_args).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.transition_path = transition_path
        self.max_step_per_episode = int(max_step_per_episode)
        self.sample_response = bool(sample_response)
        self.negative_patience = int(negative_patience)
        self.reward_done_threshold = reward_done_threshold
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(int(seed))
        self.iter = None
        self.current_observation = None

    def reset_from_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        batch_size = int(batch["history_ids"].shape[0])
        self.current_observation = {
            "user_id": batch["user_id"].to(self.device),
            "history_ids": batch["history_ids"].to(self.device),
            "history_features": batch["history_features"].to(self.device),
            "history_feedbacks": batch["history_feedbacks"].to(self.device),
            "history_event_type_ids": batch["history_event_type_ids"].to(self.device),
            "history_mask": batch["history_mask"].to(self.device),
            "cummulative_reward": torch.zeros(batch_size, device=self.device),
            "step": torch.zeros(batch_size, device=self.device, dtype=torch.long),
            "consecutive_negative": torch.zeros(batch_size, device=self.device, dtype=torch.long),
        }
        return deepcopy(self.current_observation)

    def reset(self, batch_size: int = 1) -> dict[str, torch.Tensor]:
        dataset = TransitionIterableDataset(
            self.transition_path,
            self.feature_path,
            max_seq_len=self.max_seq_len,
            max_rows=0,
        )
        self.iter = iter(DataLoader(dataset, batch_size=batch_size, num_workers=0))
        return self.reset_from_batch(next(self.iter))

    def _bernoulli(self, probs: torch.Tensor) -> torch.Tensor:
        probs = probs.clamp(0.0, 1.0)
        if self.sample_response:
            return torch.bernoulli(probs, generator=self.rng)
        return (probs >= 0.5).float()

    def _sample_play_bucket(self, play_bucket_probs: torch.Tensor) -> torch.Tensor:
        probs = play_bucket_probs.clamp_min(0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        if self.sample_response:
            return torch.multinomial(probs, num_samples=1, replacement=True, generator=self.rng).squeeze(1)
        return probs.argmax(dim=-1)

    def _dominant_event_type(
        self,
        listen: torch.Tensor,
        feedback_samples: torch.Tensor,
    ) -> torch.Tensor:
        event_type = torch.full_like(listen.long(), EVENT_TYPE_TO_ID["recommend"])
        event_type = torch.where(listen > 0.5, torch.full_like(event_type, EVENT_TYPE_TO_ID["listen"]), event_type)
        # Match session-run split priority: dislike > unlike > like > undislike > listen.
        event_type = torch.where(
            feedback_samples[:, 3] > 0.5,
            torch.full_like(event_type, EVENT_TYPE_TO_ID["undislike"]),
            event_type,
        )
        event_type = torch.where(
            feedback_samples[:, 0] > 0.5,
            torch.full_like(event_type, EVENT_TYPE_TO_ID["like"]),
            event_type,
        )
        event_type = torch.where(
            feedback_samples[:, 2] > 0.5,
            torch.full_like(event_type, EVENT_TYPE_TO_ID["unlike"]),
            event_type,
        )
        event_type = torch.where(
            feedback_samples[:, 1] > 0.5,
            torch.full_like(event_type, EVENT_TYPE_TO_ID["dislike"]),
            event_type,
        )
        return event_type

    def _append_history(
        self,
        action: torch.Tensor,
        action_features: torch.Tensor,
        reward: torch.Tensor,
        event_type: torch.Tensor,
    ) -> None:
        self.current_observation["history_ids"] = torch.cat(
            [self.current_observation["history_ids"], action.unsqueeze(1)],
            dim=1,
        )[:, -self.max_seq_len:]
        self.current_observation["history_features"] = torch.cat(
            [self.current_observation["history_features"], action_features.unsqueeze(1)],
            dim=1,
        )[:, -self.max_seq_len:, :]
        # session-run split stores the step-level reward as history signal.
        self.current_observation["history_feedbacks"] = torch.cat(
            [self.current_observation["history_feedbacks"], reward.unsqueeze(1)],
            dim=1,
        )[:, -self.max_seq_len:]
        self.current_observation["history_event_type_ids"] = torch.cat(
            [self.current_observation["history_event_type_ids"], event_type.unsqueeze(1)],
            dim=1,
        )[:, -self.max_seq_len:]
        self.current_observation["history_mask"] = (self.current_observation["history_ids"] > 0).float()

    def step(self, action: torch.Tensor, action_features: torch.Tensor | None = None):
        if self.current_observation is None:
            raise RuntimeError("Call reset() before step().")
        action = action.to(self.device).long()
        if action.dim() == 2:
            action = action[:, 0]
        if action_features is None:
            action_np = action.detach().cpu().numpy()
            action_features = torch.tensor(self.features[action_np].astype(np.float32), device=self.device)
        else:
            action_features = action_features.to(self.device)
            if action_features.dim() == 3:
                action_features = action_features[:, 0, :]

        batch = {
            "history_features": self.current_observation["history_features"],
            "history_feedbacks": self.current_observation["history_feedbacks"],
            "history_event_type_ids": self.current_observation["history_event_type_ids"],
            "history_mask": self.current_observation["history_mask"],
            "action_features": action_features,
            "prior_stats": torch.zeros(action.shape[0], self.model.prior_dim, device=self.device),
        }
        with torch.no_grad():
            out = self.model(batch)
            listen_prob = out["listen_prob"].detach()
            play_bucket_probs = out["play_bucket_probs"].detach()
            feedback_probs = out["feedback_probs"].detach()
            reward_feedback_probs = out["reward_feedback_probs"].detach()
            positive_gate = out["positive_gate"].detach()
            negative_prob = out["negative_prob"].detach()
            negative_type_probs = out["negative_type_probs"].detach()

        listen_sample = self._bernoulli(listen_prob)
        play_bucket_id = self._sample_play_bucket(play_bucket_probs)
        play_values = self.model.play_bucket_values.to(self.device)[play_bucket_id]
        play_value = play_values * listen_sample
        feedback_samples = self._bernoulli(reward_feedback_probs)
        reward = self.model.compose_reward(listen_sample, play_value, feedback_samples).detach()
        expected_reward = out["preds"].detach()
        feedback_signal = self.model.compose_feedback_signal(listen_sample, play_value, feedback_samples).detach()
        event_type = self._dominant_event_type(listen_sample, feedback_samples)
        self._append_history(action, action_features, reward, event_type)
        self.current_observation["cummulative_reward"] += reward
        self.current_observation["step"] += 1
        is_negative = reward < 0.0
        self.current_observation["consecutive_negative"] = torch.where(
            is_negative,
            self.current_observation["consecutive_negative"] + 1,
            torch.zeros_like(self.current_observation["consecutive_negative"]),
        )
        done = self.current_observation["step"] >= self.max_step_per_episode
        if self.negative_patience > 0:
            done = done | (self.current_observation["consecutive_negative"] >= self.negative_patience)
        if self.reward_done_threshold is not None:
            done = done | (self.current_observation["cummulative_reward"] <= float(self.reward_done_threshold))
        return deepcopy(self.current_observation), reward, done, {
            "response": (reward > 0).float(),
            "expected_reward": expected_reward,
            "listen_prob": listen_prob,
            "listen": listen_sample,
            "play_bucket_probs": play_bucket_probs,
            "play_bucket_id": play_bucket_id,
            "play_ratio": play_value,
            "feedback_probs": feedback_probs,
            "reward_feedback_probs": reward_feedback_probs,
            "feedback_samples": feedback_samples,
            "positive_gate": positive_gate,
            "negative_prob": negative_prob,
            "negative_type_probs": negative_type_probs,
            "feedback_signal": feedback_signal,
            "event_type_id": event_type,
            "event_type": [ID_TO_EVENT_TYPE.get(int(x), "unknown") for x in event_type.detach().cpu().tolist()],
            "cummulative_reward": self.current_observation["cummulative_reward"].detach().clone(),
            "step": self.current_observation["step"].detach().clone(),
        }

    @staticmethod
    def write_env_meta(checkpoint_path: str | Path, out_path: str | Path) -> None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"checkpoint_path": str(checkpoint_path)}, indent=2), encoding="utf-8")
