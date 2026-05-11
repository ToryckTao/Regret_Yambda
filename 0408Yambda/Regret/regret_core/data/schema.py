from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EVENT_TYPE_TO_ID = {
    "pad": 0,
    "listen": 1,
    "like": 2,
    "dislike": 3,
    "unlike": 4,
    "undislike": 5,
    "recommend": 6,
}
ID_TO_EVENT_TYPE = {v: k for k, v in EVENT_TYPE_TO_ID.items()}

REGRET_TYPE_TO_ID = {
    "none": 0,
    "low_play": 1,
    "dislike": 2,
    "unlike": 3,
}
ID_TO_REGRET_TYPE = {v: k for k, v in REGRET_TYPE_TO_ID.items()}


@dataclass
class RewardWeights:
    reward_version: str = "v1"
    play: float = 1.0
    like: float = 1.0
    dislike: float = 1.0
    unlike: float = 1.5
    undislike: float = 0.5
    clip_min: float = -1.0
    clip_max: float = 2.0
    positive_play_threshold: float = 0.8
    low_play_regret_threshold: float = 0.2
    undo_grace_seconds: int = 10
    timestamp_unit_seconds: float = 5.0
    v2_like: float = 0.8
    v2_dislike: float = 1.2
    v2_unlike: float = 0.6
    v2_undislike: float = 0.2
    v2_clip_min: float = -2.0
    v2_clip_max: float = 2.0


def history_signal(event_type: str, played_ratio_norm: float) -> float:
    if event_type == "listen":
        return float(played_ratio_norm)
    if event_type == "like":
        return 1.0
    if event_type == "dislike":
        return -1.0
    if event_type == "unlike":
        return -0.5
    if event_type == "undislike":
        return 0.5
    return 0.0


def effective_feedback(events: list[dict], undo_grace_seconds: int, timestamp_unit_seconds: float = 5.0) -> dict[str, int]:
    pending_like_times: list[int] = []
    pending_dislike_times: list[int] = []
    effective_dislike_count = 0
    effective_unlike_count = 0
    effective_undislike_count = 0
    canceled_like_count = 0
    canceled_dislike_count = 0
    for event in sorted(events, key=lambda item: (int(item["timestamp"]), int(item.get("raw_pos", 0)))):
        event_type = str(event["event_type"])
        timestamp = int(event["timestamp"])
        if event_type == "like":
            pending_like_times.append(timestamp)
        elif event_type == "unlike":
            if pending_like_times:
                like_time = pending_like_times.pop()
                gap = (timestamp - like_time) * float(timestamp_unit_seconds)
                if 0 <= gap <= undo_grace_seconds:
                    canceled_like_count += 1
                else:
                    effective_unlike_count += 1
            else:
                effective_unlike_count += 1
        elif event_type == "dislike":
            pending_dislike_times.append(timestamp)
        elif event_type == "undislike":
            if pending_dislike_times:
                dislike_time = pending_dislike_times.pop()
                gap = (timestamp - dislike_time) * float(timestamp_unit_seconds)
                if 0 <= gap <= undo_grace_seconds:
                    canceled_dislike_count += 1
                else:
                    effective_dislike_count += 1
                    effective_undislike_count += 1
            else:
                effective_undislike_count += 1
    effective_dislike_count += len(pending_dislike_times)
    return {
        "effective_like": int(len(pending_like_times) > 0),
        "effective_dislike": int(effective_dislike_count > 0),
        "effective_unlike": int(effective_unlike_count > 0),
        "effective_undislike": int(effective_undislike_count > 0),
        "canceled_like_count": int(canceled_like_count),
        "canceled_dislike_count": int(canceled_dislike_count),
    }


def summarize_events(events: list[dict], weights: RewardWeights) -> dict:
    event_types = [str(event["event_type"]) for event in events]
    play_ratios = [float(event["played_ratio_norm"]) for event in events if event["event_type"] == "listen"]
    max_play_ratio = max(play_ratios) if play_ratios else 0.0
    mean_play_ratio = float(np.mean(play_ratios)) if play_ratios else 0.0
    has_like = int("like" in event_types)
    has_dislike = int("dislike" in event_types)
    has_unlike = int("unlike" in event_types)
    has_undislike = int("undislike" in event_types)
    n_listen = int(sum(1 for item in event_types if item == "listen"))
    feedback = effective_feedback(events, int(weights.undo_grace_seconds), float(weights.timestamp_unit_seconds))
    if weights.reward_version == "v2":
        play = float(np.clip(max_play_ratio, 0.0, 1.0))
        play_reward = 2.0 * play - 1.0 if n_listen > 0 else 0.0
        reward_raw = (
            play_reward
            + weights.v2_like * feedback["effective_like"]
            - weights.v2_dislike * feedback["effective_dislike"]
            - weights.v2_unlike * feedback["effective_unlike"]
            + weights.v2_undislike * feedback["effective_undislike"]
        )
        reward_scaled = float(np.clip(reward_raw, weights.v2_clip_min, weights.v2_clip_max))
        if feedback["effective_dislike"]:
            regret_type = "dislike"
            regret_strength = float(weights.v2_dislike)
        elif feedback["effective_unlike"]:
            regret_type = "unlike"
            regret_strength = float(weights.v2_unlike)
        elif n_listen > 0 and max_play_ratio < weights.low_play_regret_threshold and not feedback["effective_like"]:
            regret_type = "low_play"
            regret_strength = float(weights.low_play_regret_threshold - max_play_ratio)
        else:
            regret_type = "none"
            regret_strength = 0.0
        feedback_label = int(
            (feedback["effective_like"] or max_play_ratio >= weights.positive_play_threshold)
            and not feedback["effective_dislike"]
            and not feedback["effective_unlike"]
        )
    else:
        feedback = {
            "effective_like": has_like,
            "effective_dislike": has_dislike,
            "effective_unlike": has_unlike,
            "effective_undislike": has_undislike,
            "canceled_like_count": 0,
            "canceled_dislike_count": 0,
        }
        reward_raw = (
            weights.play * max_play_ratio
            + weights.like * has_like
            - weights.dislike * has_dislike
            - weights.unlike * has_unlike
            + weights.undislike * has_undislike
        )
        reward_scaled = float(np.clip(reward_raw, weights.clip_min, weights.clip_max))
        if has_unlike:
            regret_type = "unlike"
            regret_strength = float(weights.unlike)
        elif has_dislike:
            regret_type = "dislike"
            regret_strength = float(weights.dislike)
        elif n_listen > 0 and max_play_ratio < weights.low_play_regret_threshold and not has_like:
            regret_type = "low_play"
            regret_strength = float(weights.low_play_regret_threshold - max_play_ratio)
        else:
            regret_type = "none"
            regret_strength = 0.0
        feedback_label = int((has_like or max_play_ratio >= weights.positive_play_threshold) and not has_dislike and not has_unlike)
    return {
        "n_events": int(len(events)),
        "n_listen": n_listen,
        "max_play_ratio": float(max_play_ratio),
        "mean_play_ratio": float(mean_play_ratio),
        "has_like": has_like,
        "has_dislike": has_dislike,
        "has_unlike": has_unlike,
        "has_undislike": has_undislike,
        **feedback,
        "reward_raw": float(reward_raw),
        "reward_scaled": reward_scaled,
        "feedback_label": feedback_label,
        "regret_type": regret_type,
        "regret_strength": regret_strength,
    }


def target_history_stats(history_events: list[dict], target_dense_item_id: int) -> dict:
    target_events = [event for event in history_events if int(event["dense_item_id"]) == int(target_dense_item_id)]
    if not target_events:
        return {
            "hist_target_n_events": 0,
            "hist_target_n_listen": 0,
            "hist_target_max_play_ratio": 0.0,
            "hist_target_mean_play_ratio": 0.0,
            "hist_target_like_count": 0,
            "hist_target_dislike_count": 0,
            "hist_target_unlike_count": 0,
            "hist_target_undislike_count": 0,
        }
    play_ratios = [float(event["played_ratio_norm"]) for event in target_events if event["event_type"] == "listen"]
    event_types = [str(event["event_type"]) for event in target_events]
    return {
        "hist_target_n_events": int(len(target_events)),
        "hist_target_n_listen": int(sum(1 for item in event_types if item == "listen")),
        "hist_target_max_play_ratio": float(max(play_ratios) if play_ratios else 0.0),
        "hist_target_mean_play_ratio": float(np.mean(play_ratios) if play_ratios else 0.0),
        "hist_target_like_count": int(sum(1 for item in event_types if item == "like")),
        "hist_target_dislike_count": int(sum(1 for item in event_types if item == "dislike")),
        "hist_target_unlike_count": int(sum(1 for item in event_types if item == "unlike")),
        "hist_target_undislike_count": int(sum(1 for item in event_types if item == "undislike")),
    }


PRIOR_STAT_COLUMNS = [
    "hist_target_n_events",
    "hist_target_n_listen",
    "hist_target_max_play_ratio",
    "hist_target_mean_play_ratio",
    "hist_target_like_count",
    "hist_target_dislike_count",
    "hist_target_unlike_count",
    "hist_target_undislike_count",
]
