from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from regret_core.data.schema import EVENT_TYPE_TO_ID, RewardWeights


def build_mlp_head(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim * 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim * 2, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


def multiclass_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: torch.Tensor | None = None,
    gamma: float = 2.0,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, targets, reduction="none")
    probs = torch.softmax(logits, dim=-1)
    pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
    focal_factor = (1.0 - pt) ** float(gamma)
    loss = ce * focal_factor
    if alpha is not None:
        alpha_t = alpha.to(logits.device)[targets]
        loss = loss * alpha_t
        return loss.sum() / alpha_t.sum().clamp_min(1.0)
    return loss.mean()


class RegretUserResponse(nn.Module):
    """Predict user feedback components and derive reward from them."""

    def __init__(
        self,
        item_dim: int = 128,
        hidden_dim: int = 128,
        event_vocab_size: int = 8,
        prior_dim: int = 8,
        regret_num_classes: int = 4,
        dropout: float = 0.1,
        history_heuristic_init: float = 0.2,
        reward_config: dict[str, Any] | None = None,
        decouple_reward_model: bool = True,
    ) -> None:
        super().__init__()
        self.item_dim = int(item_dim)
        self.hidden_dim = int(hidden_dim)
        self.event_vocab_size = int(event_vocab_size)
        self.prior_dim = int(prior_dim)
        self.regret_num_classes = int(regret_num_classes)
        self.decouple_reward_model = bool(decouple_reward_model)
        self.reward_config = RewardWeights().__dict__.copy()
        if reward_config:
            self.reward_config.update(dict(reward_config))
        self.reward_version = str(self.reward_config["reward_version"])
        history_heuristic_init = float(min(max(history_heuristic_init, 1e-4), 1.0 - 1e-4))
        play_bucket_values = torch.tensor([0.0, 0.1, 0.5, 0.9], dtype=torch.float32)
        self.register_buffer("play_bucket_values", play_bucket_values, persistent=False)
        self.play_num_buckets = int(self.play_bucket_values.numel())
        self.play_num_thresholds = int(self.play_num_buckets - 1)
        self.sim_feature_dim = 8
        self.listen_event_id = int(EVENT_TYPE_TO_ID["listen"])
        self.like_event_id = int(EVENT_TYPE_TO_ID["like"])
        self.dislike_event_id = int(EVENT_TYPE_TO_ID["dislike"])
        self.unlike_event_id = int(EVENT_TYPE_TO_ID["unlike"])
        self.undislike_event_id = int(EVENT_TYPE_TO_ID["undislike"])
        self.recommend_event_id = int(EVENT_TYPE_TO_ID["recommend"])
        self.positive_play_threshold = float(self.reward_config["positive_play_threshold"])
        self.low_play_threshold = float(self.reward_config["low_play_regret_threshold"])

        self.item_proj = nn.Linear(item_dim, hidden_dim)
        self.event_emb = nn.Embedding(event_vocab_size, hidden_dim)
        self.event_value_proj = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.explicit_feedback_proj = nn.Linear(1, hidden_dim)
        self.history_heuristic_gate_logit = nn.Parameter(
            torch.tensor(math.log(history_heuristic_init / (1.0 - history_heuristic_init)), dtype=torch.float32)
        )
        self.prior_proj = nn.Sequential(
            nn.LayerNorm(prior_dim),
            nn.Linear(prior_dim, hidden_dim),
            nn.GELU(),
        )
        self.sim_proj = nn.Sequential(
            nn.LayerNorm(self.sim_feature_dim),
            nn.Linear(self.sim_feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        pair_dim = hidden_dim * 9
        self.listen_head = build_mlp_head(pair_dim, hidden_dim, 1, dropout)
        self.play_head = build_mlp_head(pair_dim, hidden_dim, self.play_num_thresholds, dropout)
        self.feedback_head = build_mlp_head(pair_dim, hidden_dim, 4, dropout)
        self.negative_head = build_mlp_head(pair_dim, hidden_dim, 1, dropout)
        self.negative_type_head = build_mlp_head(pair_dim, hidden_dim, regret_num_classes - 1, dropout)
        self.reward_scorer_input_dim = (
            1  # base_reward
            + 1  # listen_prob
            + 1  # play_prob
            + self.play_num_buckets  # play_bucket_probs
            + 4  # reward_feedback_probs
            + 1  # negative_prob
            + (regret_num_classes - 1)  # negative_type_probs
            + 1  # positive_gate
            + 1  # feedback_signal
        )
        self.reward_scorer = build_mlp_head(self.reward_scorer_input_dim, hidden_dim, 1, dropout)
        # Conservative initialization so the learned scorer can refine the
        # base reward path without overwhelming it at the beginning.
        self.reward_scorer_blend_logit = nn.Parameter(torch.tensor(math.log(0.2 / 0.8), dtype=torch.float32))

    def build_feedback_masks(
        self,
        history_mask: torch.Tensor,
        event_ids: torch.Tensor,
        history_feedbacks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        is_listen = event_ids == self.listen_event_id
        explicit_pos_mask = (event_ids == self.like_event_id) | (event_ids == self.undislike_event_id)
        explicit_neg_mask = (event_ids == self.dislike_event_id) | (event_ids == self.unlike_event_id)
        high_play_mask = is_listen & (history_feedbacks >= self.positive_play_threshold)
        low_play_mask = is_listen & (history_feedbacks <= self.low_play_threshold)
        pos_mask = history_mask & (explicit_pos_mask | high_play_mask)
        neg_mask = history_mask & (explicit_neg_mask | low_play_mask)
        return history_mask, pos_mask, neg_mask

    def aggregate_history(
        self,
        hist: torch.Tensor,
        mask: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        mask_f = mask.unsqueeze(-1).float()
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        mean_user = (hist * mask_f).sum(dim=1) / denom
        scores = (hist * action.unsqueeze(1)).sum(dim=-1) / math.sqrt(self.hidden_dim)
        scores = torch.where(mask, scores, torch.full_like(scores, -1e9))
        any_valid = mask.any(dim=1, keepdim=True)
        weights = torch.softmax(scores, dim=1)
        weights = torch.where(any_valid, weights, torch.zeros_like(weights))
        attn_user = (hist * weights.unsqueeze(-1)).sum(dim=1)
        pooled = self.norm(mean_user + attn_user)
        return pooled * any_valid.float()

    def encode_user(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history_features = batch["history_features"]
        history_feedbacks = batch["history_feedbacks"].unsqueeze(-1)
        event_ids = batch["history_event_type_ids"].clamp(min=0, max=self.event_vocab_size - 1)
        history_mask = batch["history_mask"] > 0
        mask = history_mask.unsqueeze(-1)

        event_emb = self.event_emb(event_ids)
        # Only keep scalar values that have a stable meaning:
        # real listen played ratio, and synthetic rollout feedback on recommend events.
        value_mask = ((event_ids == self.listen_event_id) | (event_ids == self.recommend_event_id)).unsqueeze(-1)
        event_values = torch.where(value_mask, history_feedbacks, torch.zeros_like(history_feedbacks))
        event_signal = self.event_value_proj(torch.cat([event_emb, event_values], dim=-1))
        explicit_mask = (
            (event_ids > 0)
            & (event_ids != self.listen_event_id)
            & (event_ids != self.recommend_event_id)
        ).unsqueeze(-1)
        explicit_values = torch.where(explicit_mask, history_feedbacks, torch.zeros_like(history_feedbacks))
        heuristic_scale = torch.sigmoid(self.history_heuristic_gate_logit)
        explicit_feedback_signal = heuristic_scale * self.explicit_feedback_proj(explicit_values)

        hist = self.item_proj(history_features)
        hist = hist + event_signal + explicit_feedback_signal
        hist = self.norm(self.dropout(hist))

        action = self.item_proj(batch["action_features"])
        all_mask, pos_mask, neg_mask = self.build_feedback_masks(history_mask, event_ids, batch["history_feedbacks"])
        user_all = self.aggregate_history(hist, all_mask, action)
        user_pos = self.aggregate_history(hist, pos_mask, action)
        user_neg = self.aggregate_history(hist, neg_mask, action)
        return user_all, user_pos, user_neg

    def encode_pair(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        user_all, user_pos, user_neg = self.encode_user(batch)
        action = self.item_proj(batch["action_features"])
        prior = self.prior_proj(batch["prior_stats"])
        sim = self.sim_proj(self.compute_similarity_features(batch))
        return torch.cat(
            [
                user_all,
                user_pos,
                user_neg,
                action,
                user_all * action,
                user_pos * action,
                user_neg * action,
                prior,
                sim,
            ],
            dim=-1,
        )

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.float()
        return (values * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1.0)

    @staticmethod
    def _masked_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked = torch.where(mask, values, torch.full_like(values, -1e9))
        max_values = masked.max(dim=1).values
        has_any = mask.any(dim=1)
        return torch.where(has_any, max_values, torch.zeros_like(max_values))

    def compute_similarity_features(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        history_features = batch["history_features"]
        action_features = batch["action_features"]
        history_mask = batch["history_mask"] > 0
        event_ids = batch["history_event_type_ids"]
        history_feedbacks = batch["history_feedbacks"]

        hist_norm = F.normalize(history_features, dim=-1, eps=1e-8)
        action_norm = F.normalize(action_features, dim=-1, eps=1e-8)
        cosine = (hist_norm * action_norm.unsqueeze(1)).sum(dim=-1)

        explicit_pos_mask = (event_ids == self.like_event_id) | (event_ids == self.undislike_event_id)
        explicit_neg_mask = (event_ids == self.dislike_event_id) | (event_ids == self.unlike_event_id)
        high_play_mask = (event_ids == self.listen_event_id) & (history_feedbacks >= self.positive_play_threshold)
        low_play_mask = (event_ids == self.listen_event_id) & (history_feedbacks <= self.low_play_threshold)

        pos_mask = history_mask & (explicit_pos_mask | high_play_mask)
        neg_mask = history_mask & (explicit_neg_mask | low_play_mask)

        valid_count = history_mask.float().sum(dim=1).clamp_min(1.0)
        pos_frac = pos_mask.float().sum(dim=1) / valid_count
        neg_frac = neg_mask.float().sum(dim=1) / valid_count

        sim_features = torch.stack(
            [
                self._masked_max(cosine, history_mask),
                self._masked_mean(cosine, history_mask),
                self._masked_max(cosine, pos_mask),
                self._masked_mean(cosine, pos_mask),
                self._masked_max(cosine, neg_mask),
                self._masked_mean(cosine, neg_mask),
                pos_frac,
                neg_frac,
            ],
            dim=-1,
        )
        return sim_features

    def compute_play_distribution(
        self,
        play_ordinal_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cond_probs = torch.sigmoid(play_ordinal_logits)
        gt_zero = cond_probs[..., 0]
        gt_low = gt_zero * cond_probs[..., 1]
        gt_mid = gt_low * cond_probs[..., 2]
        play_bucket_probs = torch.stack(
            [
                1.0 - gt_zero,
                gt_zero - gt_low,
                gt_low - gt_mid,
                gt_mid,
            ],
            dim=-1,
        ).clamp_min(0.0)
        play_bucket_probs = play_bucket_probs / play_bucket_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return cond_probs, play_bucket_probs

    def compose_reward(
        self,
        listen_prob: torch.Tensor,
        play_prob: torch.Tensor,
        feedback_probs: torch.Tensor,
    ) -> torch.Tensor:
        effective_like = feedback_probs[..., 0]
        effective_dislike = feedback_probs[..., 1]
        effective_unlike = feedback_probs[..., 2]
        effective_undislike = feedback_probs[..., 3]
        if self.reward_version == "v2":
            play_term = listen_prob * (2.0 * play_prob - 1.0)
            reward_raw = (
                play_term
                + float(self.reward_config["v2_like"]) * effective_like
                - float(self.reward_config["v2_dislike"]) * effective_dislike
                - float(self.reward_config["v2_unlike"]) * effective_unlike
                + float(self.reward_config["v2_undislike"]) * effective_undislike
            )
            return reward_raw.clamp(
                min=float(self.reward_config["v2_clip_min"]),
                max=float(self.reward_config["v2_clip_max"]),
            )
        reward_raw = (
            float(self.reward_config["play"]) * play_prob
            + float(self.reward_config["like"]) * effective_like
            - float(self.reward_config["dislike"]) * effective_dislike
            - float(self.reward_config["unlike"]) * effective_unlike
            + float(self.reward_config["undislike"]) * effective_undislike
        )
        return reward_raw.clamp(
            min=float(self.reward_config["clip_min"]),
            max=float(self.reward_config["clip_max"]),
        )

    def apply_reward_feedback_gate(
        self,
        listen_prob: torch.Tensor,
        play_bucket_probs: torch.Tensor,
        raw_feedback_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Positive explicit feedback should only contribute to reward when
        # the model also believes the item was truly listened to and not in
        # the low-engagement play buckets.
        positive_gate = listen_prob * (play_bucket_probs[..., 2] + play_bucket_probs[..., 3])
        reward_feedback_probs = torch.stack(
            [
                raw_feedback_probs[..., 0] * positive_gate,
                raw_feedback_probs[..., 1],
                raw_feedback_probs[..., 2],
                raw_feedback_probs[..., 3] * positive_gate,
            ],
            dim=-1,
        )
        return reward_feedback_probs, positive_gate

    def compose_feedback_signal(
        self,
        listen_prob: torch.Tensor,
        play_prob: torch.Tensor,
        feedback_probs: torch.Tensor,
    ) -> torch.Tensor:
        return (
            listen_prob * play_prob
            + feedback_probs[..., 0]
            - feedback_probs[..., 1]
            - 0.5 * feedback_probs[..., 2]
            + 0.5 * feedback_probs[..., 3]
        )

    def squash_to_reward_range(self, raw: torch.Tensor) -> torch.Tensor:
        if self.reward_version == "v2":
            clip_min = float(self.reward_config["v2_clip_min"])
            clip_max = float(self.reward_config["v2_clip_max"])
        else:
            clip_min = float(self.reward_config["clip_min"])
            clip_max = float(self.reward_config["clip_max"])
        center = 0.5 * (clip_min + clip_max)
        half = 0.5 * (clip_max - clip_min)
        return center + half * torch.tanh(raw)

    def score_reward_from_heads(
        self,
        base_reward: torch.Tensor,
        listen_prob: torch.Tensor,
        play_prob: torch.Tensor,
        play_bucket_probs: torch.Tensor,
        feedback_probs: torch.Tensor,
        negative_prob: torch.Tensor,
        negative_type_probs: torch.Tensor,
        positive_gate: torch.Tensor,
        feedback_signal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scorer_inputs = torch.cat(
            [
                base_reward.unsqueeze(-1),
                listen_prob.unsqueeze(-1),
                play_prob.unsqueeze(-1),
                play_bucket_probs,
                feedback_probs,
                negative_prob.unsqueeze(-1),
                negative_type_probs,
                positive_gate.unsqueeze(-1),
                feedback_signal.unsqueeze(-1),
            ],
            dim=-1,
        )
        scorer_raw = self.reward_scorer(scorer_inputs).squeeze(-1)
        scorer_reward = self.squash_to_reward_range(scorer_raw)
        reward_scorer_blend = torch.sigmoid(self.reward_scorer_blend_logit)
        preds = (1.0 - reward_scorer_blend) * base_reward + reward_scorer_blend * scorer_reward
        if self.reward_version == "v2":
            preds = preds.clamp(
                min=float(self.reward_config["v2_clip_min"]),
                max=float(self.reward_config["v2_clip_max"]),
            )
        else:
            preds = preds.clamp(
                min=float(self.reward_config["clip_min"]),
                max=float(self.reward_config["clip_max"]),
            )
        return preds, scorer_reward

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        features = self.encode_pair(batch)
        listen_logits = self.listen_head(features).squeeze(-1)
        play_ordinal_logits = self.play_head(features)
        feedback_logits = self.feedback_head(features)
        negative_logits = self.negative_head(features).squeeze(-1)
        negative_type_logits = self.negative_type_head(features)
        listen_prob = torch.sigmoid(listen_logits)
        play_ordinal_cond_probs, play_bucket_probs = self.compute_play_distribution(play_ordinal_logits)
        play_prob = (play_bucket_probs * self.play_bucket_values.to(play_bucket_probs.device)).sum(dim=-1)
        feedback_probs = torch.sigmoid(feedback_logits)
        negative_prob = torch.sigmoid(negative_logits)
        negative_type_probs = torch.softmax(negative_type_logits, dim=-1)
        reward_feedback_probs, positive_gate = self.apply_reward_feedback_gate(
            listen_prob,
            play_bucket_probs,
            feedback_probs,
        )
        base_preds = self.compose_reward(listen_prob, play_prob, reward_feedback_probs)
        feedback_signal = self.compose_feedback_signal(listen_prob, play_prob, reward_feedback_probs)
        if self.decouple_reward_model:
            preds = base_preds
            scorer_reward = base_preds
            reward_scorer_blend = torch.zeros((), device=base_preds.device, dtype=base_preds.dtype)
        else:
            preds, scorer_reward = self.score_reward_from_heads(
                base_preds,
                listen_prob,
                play_prob,
                play_bucket_probs,
                reward_feedback_probs,
                negative_prob,
                negative_type_probs,
                positive_gate,
                feedback_signal,
            )
            reward_scorer_blend = torch.sigmoid(self.reward_scorer_blend_logit)
        regret_probs = torch.cat(
            [
                (1.0 - negative_prob).unsqueeze(-1),
                negative_prob.unsqueeze(-1) * negative_type_probs,
            ],
            dim=-1,
        ).clamp_min(1e-8)
        regret_logits = regret_probs.log()
        return {
            "preds": preds,
            "base_preds": base_preds,
            "listen_logits": listen_logits,
            "listen_prob": listen_prob,
            "play_ordinal_logits": play_ordinal_logits,
            "play_ordinal_cond_probs": play_ordinal_cond_probs,
            "play_bucket_probs": play_bucket_probs,
            "play_prob": play_prob,
            "feedback_logits": feedback_logits,
            "feedback_probs": feedback_probs,
            "negative_logits": negative_logits,
            "negative_prob": negative_prob,
            "negative_type_logits": negative_type_logits,
            "negative_type_probs": negative_type_probs,
            "scorer_reward": scorer_reward,
            "reward_scorer_blend": reward_scorer_blend,
            "reward_feedback_probs": reward_feedback_probs,
            "positive_gate": positive_gate,
            "feedback_signal": feedback_signal,
            "regret_logits": regret_logits,
        }

    def loss(
        self,
        batch: dict[str, torch.Tensor],
        reward_loss_weight: float = 1.0,
        reward_sample_weights: torch.Tensor | None = None,
        listen_loss_weight: float = 0.0,
        listen_pos_weight: torch.Tensor | None = None,
        play_loss_weight: float = 0.0,
        play_bucket_weights: torch.Tensor | None = None,
        feedback_loss_weight: float = 0.0,
        feedback_pos_weights: torch.Tensor | None = None,
        negative_pos_weight: torch.Tensor | None = None,
        negative_type_weights: torch.Tensor | None = None,
        negative_type_focal_gamma: float = 2.0,
        regret_loss_weight: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        out = self.forward(batch)
        target = batch["reward"].float()
        mse_per = F.mse_loss(out["preds"], target, reduction="none")
        mse_metric = mse_per.mean()
        if reward_sample_weights is not None:
            sample_weights = reward_sample_weights.to(device=target.device, dtype=target.dtype)
            reward_loss = (mse_per * sample_weights).sum() / sample_weights.sum().clamp_min(1.0)
        else:
            reward_loss = mse_metric
        mae = F.l1_loss(out["preds"], target)
        listen_loss = target.new_tensor(0.0)
        listen_acc = target.new_tensor(0.0)
        if "listen_target" in batch:
            listen_target = batch["listen_target"].float()
            listen_loss = F.binary_cross_entropy_with_logits(
                out["listen_logits"],
                listen_target,
                pos_weight=listen_pos_weight.to(out["listen_logits"].device) if listen_pos_weight is not None else None,
            )
            listen_acc = ((out["listen_prob"] >= 0.5) == (listen_target >= 0.5)).float().mean()
        play_loss = target.new_tensor(0.0)
        play_mae = target.new_tensor(0.0)
        play_bucket_acc = target.new_tensor(0.0)
        if "play_target" in batch:
            play_target = batch["play_target"].float()
            play_mae = F.l1_loss(out["play_prob"], play_target)
        if "play_ordinal_targets" in batch:
            play_ordinal_targets = batch["play_ordinal_targets"].float()
            play_ordinal_valid_mask = batch.get("play_ordinal_valid_mask")
            if play_ordinal_valid_mask is None:
                play_ordinal_valid_mask = torch.ones_like(play_ordinal_targets)
            else:
                play_ordinal_valid_mask = play_ordinal_valid_mask.float()
            play_bce = F.binary_cross_entropy_with_logits(
                out["play_ordinal_logits"],
                play_ordinal_targets,
                reduction="none",
            )
            if play_bucket_weights is not None and "play_bucket_id" in batch:
                play_bucket_target = batch["play_bucket_id"].long()
                play_sample_weights = play_bucket_weights.to(out["play_ordinal_logits"].device)[play_bucket_target].unsqueeze(-1)
            else:
                play_sample_weights = torch.ones_like(play_ordinal_valid_mask)
            play_weight_mask = play_ordinal_valid_mask * play_sample_weights
            play_loss = (play_bce * play_weight_mask).sum() / play_weight_mask.sum().clamp_min(1.0)
        elif "play_bucket_id" in batch:
            play_bucket_target = batch["play_bucket_id"].long()
            play_loss = F.cross_entropy(
                out["play_bucket_probs"].clamp_min(1e-8).log(),
                play_bucket_target,
                weight=play_bucket_weights.to(out["play_bucket_probs"].device) if play_bucket_weights is not None else None,
            )
            play_bucket_acc = (out["play_bucket_probs"].argmax(dim=-1) == play_bucket_target).float().mean()
        elif "play_target" in batch:
            play_target = batch["play_target"].float()
            play_loss = F.mse_loss(out["play_prob"], play_target)
        if "play_bucket_id" in batch:
            play_bucket_target = batch["play_bucket_id"].long()
            play_bucket_acc = (out["play_bucket_probs"].argmax(dim=-1) == play_bucket_target).float().mean()
        feedback_loss = target.new_tensor(0.0)
        feedback_acc = target.new_tensor(0.0)
        if "feedback_targets" in batch:
            feedback_targets = batch["feedback_targets"].float()
            feedback_valid_mask = batch.get("feedback_valid_mask")
            if feedback_valid_mask is None:
                feedback_valid_mask = torch.ones_like(feedback_targets)
            else:
                feedback_valid_mask = feedback_valid_mask.float()
            feedback_bce = F.binary_cross_entropy_with_logits(
                out["feedback_logits"],
                feedback_targets,
                reduction="none",
                pos_weight=feedback_pos_weights.to(out["feedback_logits"].device) if feedback_pos_weights is not None else None,
            )
            feedback_loss = (feedback_bce * feedback_valid_mask).sum() / feedback_valid_mask.sum().clamp_min(1.0)
            feedback_match = ((out["feedback_probs"] >= 0.5) == (feedback_targets >= 0.5)).float()
            feedback_acc = (feedback_match * feedback_valid_mask).sum() / feedback_valid_mask.sum().clamp_min(1.0)
        negative_loss = target.new_tensor(0.0)
        negative_acc = target.new_tensor(0.0)
        negative_type_loss = target.new_tensor(0.0)
        negative_type_acc = target.new_tensor(0.0)
        regret_loss = target.new_tensor(0.0)
        regret_acc = target.new_tensor(0.0)
        if "regret_type_id" in batch:
            regret_target = batch["regret_type_id"].long()
            negative_target = (regret_target > 0).float()
            negative_loss = F.binary_cross_entropy_with_logits(
                out["negative_logits"],
                negative_target,
                pos_weight=negative_pos_weight.to(out["negative_logits"].device) if negative_pos_weight is not None else None,
            )
            negative_acc = ((out["negative_prob"] >= 0.5) == (negative_target >= 0.5)).float().mean()
            negative_mask = regret_target > 0
            if bool(negative_mask.any()):
                negative_type_target = regret_target[negative_mask] - 1
                if float(negative_type_focal_gamma) > 0.0:
                    negative_type_loss = multiclass_focal_loss(
                        out["negative_type_logits"][negative_mask],
                        negative_type_target,
                        alpha=negative_type_weights,
                        gamma=float(negative_type_focal_gamma),
                    )
                else:
                    negative_type_loss = F.cross_entropy(
                        out["negative_type_logits"][negative_mask],
                        negative_type_target,
                        weight=negative_type_weights.to(out["negative_type_logits"].device) if negative_type_weights is not None else None,
                    )
                negative_type_acc = (
                    out["negative_type_logits"][negative_mask].argmax(dim=-1) == negative_type_target
                ).float().mean()
            regret_loss = negative_loss + negative_type_loss
            regret_acc = (out["regret_logits"].argmax(dim=-1) == regret_target).float().mean()
        loss = (
            float(reward_loss_weight) * reward_loss
            + float(listen_loss_weight) * listen_loss
            + float(play_loss_weight) * play_loss
            + float(feedback_loss_weight) * feedback_loss
            + float(regret_loss_weight) * regret_loss
        )
        simulator_loss = (
            float(listen_loss_weight) * listen_loss
            + float(play_loss_weight) * play_loss
            + float(feedback_loss_weight) * feedback_loss
            + float(regret_loss_weight) * regret_loss
        )
        return {
            "loss": loss,
            "mse": mse_metric.detach(),
            "reward_loss": reward_loss.detach(),
            "simulator_loss": simulator_loss.detach(),
            "mae": mae.detach(),
            "listen_loss": listen_loss.detach(),
            "listen_acc": listen_acc.detach(),
            "play_loss": play_loss.detach(),
            "play_mae": play_mae.detach(),
            "play_bucket_acc": play_bucket_acc.detach(),
            "feedback_loss": feedback_loss.detach(),
            "feedback_acc": feedback_acc.detach(),
            "negative_loss": negative_loss.detach(),
            "negative_acc": negative_acc.detach(),
            "negative_type_loss": negative_type_loss.detach(),
            "negative_type_acc": negative_type_acc.detach(),
            "regret_loss": regret_loss.detach(),
            "regret_acc": regret_acc.detach(),
            "preds": out["preds"],
            "listen_prob": out["listen_prob"],
            "play_prob": out["play_prob"],
            "play_bucket_probs": out["play_bucket_probs"],
            "feedback_probs": out["feedback_probs"],
            "negative_prob": out["negative_prob"],
            "negative_type_probs": out["negative_type_probs"],
            "reward_feedback_probs": out["reward_feedback_probs"],
            "positive_gate": out["positive_gate"],
            "feedback_signal": out["feedback_signal"],
            "regret_logits": out["regret_logits"],
        }
