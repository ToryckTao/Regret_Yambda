#!/usr/bin/env python3
"""主线数据切分：把 Yambda 原始多事件序列流式切成 RL transition parquet。

核心逻辑：按时间间隔切 session，再把同 session 内连续同 item 的事件聚合成一个 step。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from regret_core.data.schema import (  # noqa: E402
    EVENT_TYPE_TO_ID,
    RewardWeights,
    history_signal,
    summarize_events,
    target_history_stats,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Regret transition parquet shards")
    parser.add_argument("--multi_event_parquet", default=str(REPO_ROOT / "../0330Yambda/data/sequential-50m/multi_event.parquet"))
    parser.add_argument("--orig2dense_npy", default=str(PROJECT_ROOT / "artifacts/mappings/raw_rqkmeans/orig2dense_item_id.npy"))
    parser.add_argument("--out_root", default=str(PROJECT_ROOT / "artifacts/transitions/raw_rqkmeans"))
    parser.add_argument("--history_len", type=int, default=50)
    parser.add_argument("--trajectory_mode", choices=["session_run", "item_window"], default="session_run")
    parser.add_argument("--session_gap_seconds", type=int, default=0)
    parser.add_argument("--max_session_span_seconds", type=int, default=21600)
    parser.add_argument("--max_run_events", type=int, default=100)
    parser.add_argument("--replay_eval_steps", type=int, default=0)
    parser.add_argument("--close_gap_seconds", type=int, default=3600)
    parser.add_argument("--timestamp_unit_seconds", type=float, default=5.0)
    parser.add_argument("--anchor_policy", choices=["first_visible", "listen_only"], default="first_visible")
    parser.add_argument("--max_users", type=int, default=0)
    parser.add_argument("--shard_rows", type=int, default=200000)
    parser.add_argument("--reward_version", choices=["v1", "v2"], default="v1")
    parser.add_argument("--reward_w_play", type=float, default=1.0)
    parser.add_argument("--reward_w_like", type=float, default=1.0)
    parser.add_argument("--reward_w_dislike", type=float, default=1.0)
    parser.add_argument("--reward_w_unlike", type=float, default=1.5)
    parser.add_argument("--reward_w_undislike", type=float, default=0.5)
    parser.add_argument("--reward_clip_min", type=float, default=-1.0)
    parser.add_argument("--reward_clip_max", type=float, default=2.0)
    parser.add_argument("--positive_play_threshold", type=float, default=0.8)
    parser.add_argument("--low_play_regret_threshold", type=float, default=0.2)
    parser.add_argument("--undo_grace_seconds", type=int, default=10)
    parser.add_argument("--reward_v2_w_like", type=float, default=0.8)
    parser.add_argument("--reward_v2_w_dislike", type=float, default=1.2)
    parser.add_argument("--reward_v2_w_unlike", type=float, default=0.6)
    parser.add_argument("--reward_v2_w_undislike", type=float, default=0.2)
    parser.add_argument("--reward_v2_clip_min", type=float, default=-2.0)
    parser.add_argument("--reward_v2_clip_max", type=float, default=2.0)
    parser.add_argument("--precompute_regret_memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--regret_memory_size", type=int, default=20)
    parser.add_argument("--regret_memory_gamma", type=float, default=0.9)
    parser.add_argument("--regret_memory_min_phi", type=float, default=1e-6)
    parser.add_argument(
        "--regret_memory_scope",
        choices=["all_failed", "revision_only"],
        default="all_failed",
        help="all_failed stores low_play/dislike/unlike; revision_only stores unlike only.",
    )
    return parser.parse_args()


def iter_user_rows(parquet_path: Path, batch_size: int = 32) -> Iterable[dict]:
    pf = pq.ParquetFile(parquet_path)
    columns = [
        "uid",
        "timestamp",
        "item_id",
        "is_organic",
        "played_ratio_pct",
        "track_length_seconds",
        "event_type",
    ]
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        pyd = batch.to_pydict()
        for idx in range(len(pyd["uid"])):
            yield {key: pyd[key][idx] for key in columns}


def expand_user_events(row: dict, orig2dense: np.ndarray) -> tuple[list[dict], dict]:
    events: list[dict] = []
    raw_counts: Counter[str] = Counter()
    kept_counts: Counter[str] = Counter()
    missing = 0
    for pos, orig_item_id_raw in enumerate(row["item_id"]):
        orig_item_id = int(orig_item_id_raw)
        event_type = str(row["event_type"][pos])
        raw_counts[event_type] += 1
        if orig_item_id >= len(orig2dense):
            missing += 1
            continue
        dense_item_id = int(orig2dense[orig_item_id])
        if dense_item_id <= 0:
            missing += 1
            continue
        ratio = row["played_ratio_pct"][pos]
        ratio_norm = 0.0 if ratio is None else float(ratio) / 100.0
        kept_counts[event_type] += 1
        events.append(
            {
                "uid": int(row["uid"]),
                "timestamp": int(row["timestamp"][pos]),
                "orig_item_id": orig_item_id,
                "dense_item_id": dense_item_id,
                "played_ratio_norm": ratio_norm,
                "event_type": event_type,
                "event_type_id": int(EVENT_TYPE_TO_ID.get(event_type, 0)),
                "history_signal": history_signal(event_type, ratio_norm),
                "raw_pos": int(pos),
            }
        )
    events.sort(key=lambda event: (event["timestamp"], event["raw_pos"]))
    return events, {
        "raw_events": len(row["timestamp"]),
        "kept_events": len(events),
        "missing_mapping": missing,
        "raw_event_type_counts": dict(raw_counts),
        "kept_event_type_counts": dict(kept_counts),
    }


def build_episodes(
    events: list[dict],
    close_gap_seconds: int,
    anchor_policy: str,
    timestamp_unit_seconds: float,
) -> list[dict]:
    episodes: list[dict] = []
    open_by_item: dict[int, dict] = {}
    for global_idx, event in enumerate(events):
        item_id = int(event["dense_item_id"])
        current = open_by_item.get(item_id)
        elapsed_seconds = (
            int(event["timestamp"]) - int(current["last_timestamp"])
        ) * float(timestamp_unit_seconds) if current is not None else 0.0
        should_open = current is None or elapsed_seconds > close_gap_seconds
        if should_open:
            current = {
                "dense_item_id": item_id,
                "orig_item_id": int(event["orig_item_id"]),
                "start_idx": global_idx,
                "end_idx": global_idx,
                "last_timestamp": int(event["timestamp"]),
                "event_indices": [global_idx],
            }
            episodes.append(current)
            open_by_item[item_id] = current
        else:
            current["end_idx"] = global_idx
            current["last_timestamp"] = int(event["timestamp"])
            current["event_indices"].append(global_idx)
    if anchor_policy == "listen_only":
        episodes = [ep for ep in episodes if events[ep["event_indices"][0]]["event_type"] == "listen"]
    episodes.sort(key=lambda ep: ep["start_idx"])
    return episodes


def build_sessions(
    events: list[dict],
    session_gap_seconds: int,
    max_session_span_seconds: int,
    timestamp_unit_seconds: float,
) -> list[dict]:
    sessions: list[dict] = []
    if not events:
        return sessions
    start_idx = 0
    session_id = 0
    for global_idx in range(1, len(events)):
        gap_seconds = (
            int(events[global_idx]["timestamp"]) - int(events[global_idx - 1]["timestamp"])
        ) * float(timestamp_unit_seconds)
        span_seconds = (
            int(events[global_idx]["timestamp"]) - int(events[start_idx]["timestamp"])
        ) * float(timestamp_unit_seconds)
        if gap_seconds > session_gap_seconds or (
            max_session_span_seconds > 0 and span_seconds > max_session_span_seconds
        ):
            event_indices = list(range(start_idx, global_idx))
            sessions.append(
                {
                    "session_id": session_id,
                    "start_idx": start_idx,
                    "end_idx": global_idx - 1,
                    "event_indices": event_indices,
                    "start_time": int(events[start_idx]["timestamp"]),
                    "end_time": int(events[global_idx - 1]["timestamp"]),
                    "n_events": len(event_indices),
                }
            )
            session_id += 1
            start_idx = global_idx
    event_indices = list(range(start_idx, len(events)))
    sessions.append(
        {
            "session_id": session_id,
            "start_idx": start_idx,
            "end_idx": len(events) - 1,
            "event_indices": event_indices,
            "start_time": int(events[start_idx]["timestamp"]),
            "end_time": int(events[-1]["timestamp"]),
            "n_events": len(event_indices),
        }
    )
    return sessions


def step_history_signal(summary: dict) -> float:
    return float(summary.get("reward_scaled", 0.0) or 0.0)


def step_event_type_id(summary: dict, anchor_event_type_id: int) -> int:
    if int(summary.get("effective_dislike", summary.get("has_dislike", 0)) or 0):
        return int(EVENT_TYPE_TO_ID["dislike"])
    if int(summary.get("effective_unlike", summary.get("has_unlike", 0)) or 0):
        return int(EVENT_TYPE_TO_ID["unlike"])
    if int(summary.get("effective_like", summary.get("has_like", 0)) or 0):
        return int(EVENT_TYPE_TO_ID["like"])
    if int(summary.get("effective_undislike", summary.get("has_undislike", 0)) or 0):
        return int(EVENT_TYPE_TO_ID["undislike"])
    if int(summary.get("n_listen", 0) or 0) > 0:
        return int(EVENT_TYPE_TO_ID["listen"])
    return int(anchor_event_type_id)


def build_session_run_steps(
    events: list[dict],
    sessions: list[dict],
    anchor_policy: str,
    weights: RewardWeights,
    max_run_events: int = 0,
) -> list[dict]:
    steps: list[dict] = []

    def add_step(session: dict, run_indices: list[int], session_step_idx: int) -> None:
        if not run_indices:
            return
        anchor_event = events[run_indices[0]]
        if anchor_policy == "listen_only" and anchor_event["event_type"] != "listen":
            return
        step_events = [events[idx] for idx in run_indices]
        summary = summarize_events(step_events, weights)
        steps.append(
            {
                "session_id": int(session["session_id"]),
                "session_start_time": int(session["start_time"]),
                "session_end_time": int(session["end_time"]),
                "session_event_count": int(session["n_events"]),
                "session_step_idx": int(session_step_idx),
                "user_step_idx": int(len(steps)),
                "dense_item_id": int(anchor_event["dense_item_id"]),
                "orig_item_id": int(anchor_event["orig_item_id"]),
                "start_idx": int(run_indices[0]),
                "end_idx": int(run_indices[-1]),
                "event_indices": list(run_indices),
                "step_start_time": int(events[run_indices[0]]["timestamp"]),
                "step_end_time": int(events[run_indices[-1]]["timestamp"]),
                "summary": summary,
                "history_signal": step_history_signal(summary),
                "history_event_type_id": step_event_type_id(summary, int(anchor_event["event_type_id"])),
            }
        )

    for session in sessions:
        indices = session["event_indices"]
        if not indices:
            continue
        run_indices = [indices[0]]
        session_step_idx = 0
        prev_item = int(events[indices[0]]["dense_item_id"])
        for idx in indices[1:]:
            item_id = int(events[idx]["dense_item_id"])
            if item_id != prev_item:
                add_step(session, run_indices, session_step_idx)
                session_step_idx += 1
                run_indices = [idx]
                prev_item = item_id
            elif max_run_events > 0 and len(run_indices) >= max_run_events:
                add_step(session, run_indices, session_step_idx)
                session_step_idx += 1
                run_indices = [idx]
                prev_item = item_id
            else:
                run_indices.append(idx)
        add_step(session, run_indices, session_step_idx)
    return steps


def target_step_history_stats(history_steps: list[dict], target_dense_item_id: int) -> dict:
    target_steps = [step for step in history_steps if int(step["dense_item_id"]) == int(target_dense_item_id)]
    if not target_steps:
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
    summaries = [step["summary"] for step in target_steps]
    play_ratios = [float(item["max_play_ratio"]) for item in summaries if int(item["n_listen"]) > 0]
    listen_count = int(sum(int(item["n_listen"]) for item in summaries))
    return {
        "hist_target_n_events": int(sum(int(item["n_events"]) for item in summaries)),
        "hist_target_n_listen": listen_count,
        "hist_target_max_play_ratio": float(max(play_ratios) if play_ratios else 0.0),
        "hist_target_mean_play_ratio": float(np.mean(play_ratios) if play_ratios else 0.0),
        "hist_target_like_count": int(sum(int(item["has_like"]) for item in summaries)),
        "hist_target_dislike_count": int(sum(int(item["has_dislike"]) for item in summaries)),
        "hist_target_unlike_count": int(sum(int(item["has_unlike"]) for item in summaries)),
        "hist_target_undislike_count": int(sum(int(item["has_undislike"]) for item in summaries)),
    }


def step_failed_memory_signal(step: dict, scope: str) -> tuple[int, float] | None:
    summary = step["summary"]
    regret_type = str(summary.get("regret_type", "none"))
    strength = float(summary.get("regret_strength", 0.0) or 0.0)
    if regret_type == "unlike":
        return 3, max(strength, 1e-6)
    if scope == "revision_only":
        return None
    if regret_type == "dislike":
        return 2, max(strength, 1e-6)
    if regret_type == "low_play":
        return 1, max(strength, 1e-6)
    return None


def attach_regret_memory_snapshots(
    steps: list[dict],
    memory_size: int,
    gamma: float,
    min_phi: float,
    scope: str,
) -> dict[str, int]:
    stats = Counter()
    memory_entries: list[dict[str, Any]] = []
    memory_size = int(max(memory_size, 0))
    gamma = float(gamma)
    min_phi = float(min_phi)
    for step_pos, step in enumerate(steps):
        item_ids: list[int] = []
        phis: list[float] = []
        type_ids: list[int] = []
        if memory_size > 0:
            for entry in memory_entries:
                delta_tr = max(0, int(step["user_step_idx"]) - int(entry["user_step_idx"]))
                phi = float(entry["strength"]) * (gamma ** delta_tr)
                if phi < min_phi:
                    continue
                item_ids.append(int(entry["dense_item_id"]))
                phis.append(float(phi))
                type_ids.append(int(entry["regret_type_id"]))
                if len(item_ids) >= memory_size:
                    break
        step["regret_memory_item_ids"] = item_ids
        step["regret_memory_phis"] = phis
        step["regret_memory_type_ids"] = type_ids
        stats["snapshot_rows"] += 1
        stats["snapshot_entries"] += len(item_ids)

        signal = step_failed_memory_signal(step, scope)
        if signal is None:
            continue
        regret_type_id, strength = signal
        memory_entries.insert(
            0,
            {
                "dense_item_id": int(step["dense_item_id"]),
                "regret_type_id": int(regret_type_id),
                "strength": float(strength),
                "user_step_idx": int(step["user_step_idx"]),
            },
        )
        stats["memory_insertions"] += 1
        if regret_type_id == 1:
            stats["memory_low_play"] += 1
        elif regret_type_id == 2:
            stats["memory_dislike"] += 1
        elif regret_type_id == 3:
            stats["memory_unlike"] += 1
        # Keep enough recent raw entries to fill future snapshots after decay,
        # but avoid unbounded per-user memory on extremely long histories.
        if memory_size > 0:
            memory_entries = memory_entries[: max(memory_size * 8, memory_size)]
    return {key: int(value) for key, value in stats.items()}


def assign_splits(n_episode: int) -> list[str]:
    if n_episode <= 0:
        return []
    if n_episode == 1:
        return ["train"]
    if n_episode == 2:
        return ["train", "test"]
    split = ["train"] * n_episode
    split[-2] = "val"
    split[-1] = "test"
    return split


def make_row(
    transition_id: int,
    split_name: str,
    user_id: int,
    events: list[dict],
    episode: dict,
    history_len: int,
    weights: RewardWeights,
) -> dict[str, Any]:
    indices = episode["event_indices"]
    start_idx = int(indices[0])
    end_idx = int(indices[-1])
    target_event = events[start_idx]
    episode_events = [events[idx] for idx in indices]
    history_events = events[max(0, start_idx - history_len): start_idx]
    next_history_events = events[max(0, end_idx + 1 - history_len): end_idx + 1]
    summary = summarize_events(episode_events, weights)
    prior = target_history_stats(history_events, int(target_event["dense_item_id"]))
    return {
        "transition_id": int(transition_id),
        "split": split_name,
        "user_id": int(user_id),
        "episode_start_time": int(events[start_idx]["timestamp"]),
        "episode_end_time": int(events[end_idx]["timestamp"]),
        "target_orig_item_id": int(target_event["orig_item_id"]),
        "target_dense_item_id": int(target_event["dense_item_id"]),
        "anchor_event_type": str(target_event["event_type"]),
        "anchor_event_type_id": int(target_event["event_type_id"]),
        "history_item_ids": [int(event["dense_item_id"]) for event in history_events],
        "history_feedbacks": [float(event["history_signal"]) for event in history_events],
        "history_event_type_ids": [int(event["event_type_id"]) for event in history_events],
        "next_history_item_ids": [int(event["dense_item_id"]) for event in next_history_events],
        "next_history_feedbacks": [float(event["history_signal"]) for event in next_history_events],
        "next_history_event_type_ids": [int(event["event_type_id"]) for event in next_history_events],
        **summary,
        **prior,
    }


def make_step_row(
    transition_id: int,
    split_name: str,
    user_id: int,
    events: list[dict],
    steps: list[dict],
    step_pos: int,
    history_len: int,
) -> dict[str, Any]:
    step = steps[step_pos]
    start_idx = int(step["start_idx"])
    end_idx = int(step["end_idx"])
    target_event = events[start_idx]
    history_steps = steps[max(0, step_pos - history_len): step_pos]
    next_history_steps = steps[max(0, step_pos + 1 - history_len): step_pos + 1]
    prior = target_step_history_stats(history_steps, int(target_event["dense_item_id"]))
    return {
        "transition_id": int(transition_id),
        "split": split_name,
        "user_id": int(user_id),
        "session_id": int(step["session_id"]),
        "session_start_time": int(step["session_start_time"]),
        "session_end_time": int(step["session_end_time"]),
        "session_event_count": int(step["session_event_count"]),
        "session_step_idx": int(step["session_step_idx"]),
        "user_step_idx": int(step["user_step_idx"]),
        "step_time": int(step["step_end_time"]),
        "step_start_time": int(step["step_start_time"]),
        "step_end_time": int(step["step_end_time"]),
        "episode_start_time": int(step["step_start_time"]),
        "episode_end_time": int(step["step_end_time"]),
        "target_orig_item_id": int(target_event["orig_item_id"]),
        "target_dense_item_id": int(target_event["dense_item_id"]),
        "anchor_event_type": str(target_event["event_type"]),
        "anchor_event_type_id": int(target_event["event_type_id"]),
        "history_item_ids": [int(item["dense_item_id"]) for item in history_steps],
        "history_feedbacks": [float(item["history_signal"]) for item in history_steps],
        "history_event_type_ids": [int(item["history_event_type_id"]) for item in history_steps],
        "next_history_item_ids": [int(item["dense_item_id"]) for item in next_history_steps],
        "next_history_feedbacks": [float(item["history_signal"]) for item in next_history_steps],
        "next_history_event_type_ids": [int(item["history_event_type_id"]) for item in next_history_steps],
        "regret_memory_item_ids": [int(item) for item in step.get("regret_memory_item_ids", [])],
        "regret_memory_phis": [float(item) for item in step.get("regret_memory_phis", [])],
        "regret_memory_type_ids": [int(item) for item in step.get("regret_memory_type_ids", [])],
        **step["summary"],
        **prior,
    }


class ShardedParquetWriter:
    def __init__(self, out_root: Path, shard_rows: int) -> None:
        self.out_root = out_root
        self.shard_rows = shard_rows
        self.buffers: dict[str, list[dict]] = defaultdict(list)
        self.shard_idx: dict[str, int] = defaultdict(int)
        for split in ["train", "val", "test"]:
            (self.out_root / split).mkdir(parents=True, exist_ok=True)

    def add(self, split: str, row: dict) -> None:
        (self.out_root / split).mkdir(parents=True, exist_ok=True)
        buf = self.buffers[split]
        buf.append(row)
        if len(buf) >= self.shard_rows:
            self.flush(split)

    def flush(self, split: str) -> None:
        buf = self.buffers[split]
        if not buf:
            return
        out_path = self.out_root / split / f"part-{self.shard_idx[split]:05d}.parquet"
        table = pa.Table.from_pylist(buf)
        pq.write_table(table, out_path)
        self.shard_idx[split] += 1
        self.buffers[split] = []

    def close(self) -> None:
        for split in list(self.buffers.keys()):
            self.flush(split)


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    session_gap_seconds = int(args.session_gap_seconds or args.close_gap_seconds)
    orig2dense = np.load(args.orig2dense_npy, mmap_mode="r")
    weights = RewardWeights(
        reward_version=args.reward_version,
        play=args.reward_w_play,
        like=args.reward_w_like,
        dislike=args.reward_w_dislike,
        unlike=args.reward_w_unlike,
        undislike=args.reward_w_undislike,
        clip_min=args.reward_clip_min,
        clip_max=args.reward_clip_max,
        positive_play_threshold=args.positive_play_threshold,
        low_play_regret_threshold=args.low_play_regret_threshold,
        undo_grace_seconds=args.undo_grace_seconds,
        timestamp_unit_seconds=args.timestamp_unit_seconds,
        v2_like=args.reward_v2_w_like,
        v2_dislike=args.reward_v2_w_dislike,
        v2_unlike=args.reward_v2_w_unlike,
        v2_undislike=args.reward_v2_w_undislike,
        v2_clip_min=args.reward_v2_clip_min,
        v2_clip_max=args.reward_v2_clip_max,
    )
    writer = ShardedParquetWriter(out_root, args.shard_rows)
    transition_id = 1
    split_counts = Counter()
    regret_counts = Counter()
    reward_values: list[float] = []
    total_stats = Counter()
    users_processed = 0
    pbar = tqdm(iter_user_rows(Path(args.multi_event_parquet)), desc="[split]", unit="users", ncols=80)
    for row in pbar:
        if args.max_users and users_processed >= args.max_users:
            break
        users_processed += 1
        events, stats = expand_user_events(row, orig2dense)
        total_stats["raw_events"] += stats["raw_events"]
        total_stats["kept_events"] += stats["kept_events"]
        total_stats["missing_mapping"] += stats["missing_mapping"]
        if not events:
            continue
        if args.trajectory_mode == "session_run":
            sessions = build_sessions(
                events,
                session_gap_seconds,
                int(args.max_session_span_seconds),
                args.timestamp_unit_seconds,
            )
            steps = build_session_run_steps(events, sessions, args.anchor_policy, weights, int(args.max_run_events))
            if args.precompute_regret_memory:
                memory_stats = attach_regret_memory_snapshots(
                    steps,
                    memory_size=int(args.regret_memory_size),
                    gamma=float(args.regret_memory_gamma),
                    min_phi=float(args.regret_memory_min_phi),
                    scope=str(args.regret_memory_scope),
                )
                for key, value in memory_stats.items():
                    total_stats[f"regret_memory_{key}"] += int(value)
            total_stats["sessions"] += len(sessions)
            total_stats["session_events"] += sum(int(session["n_events"]) for session in sessions)
            total_stats["steps"] += len(steps)
            total_stats["step_events"] += sum(int(step["summary"]["n_events"]) for step in steps)
            split_plan = assign_splits(len(steps))
            for step_pos, split_name in enumerate(split_plan):
                out_row = make_step_row(
                    transition_id=transition_id,
                    split_name=split_name,
                    user_id=int(row["uid"]),
                    events=events,
                    steps=steps,
                    step_pos=step_pos,
                    history_len=args.history_len,
                )
                writer.add(split_name, out_row)
                split_counts[split_name] += 1
                regret_counts[out_row["regret_type"]] += 1
                if len(reward_values) < 1_000_000:
                    reward_values.append(float(out_row["reward_scaled"]))
                transition_id += 1
            replay_eval_steps = int(args.replay_eval_steps)
            if replay_eval_steps > 0 and steps:
                replay_test_start = max(0, len(steps) - replay_eval_steps)
                replay_val_start = max(0, replay_test_start - replay_eval_steps)
                replay_ranges = {
                    "replay_val": range(replay_val_start, replay_test_start),
                    "replay_test": range(replay_test_start, len(steps)),
                }
                for replay_split, replay_positions in replay_ranges.items():
                    for step_pos in replay_positions:
                        out_row = make_step_row(
                            transition_id=transition_id,
                            split_name=replay_split,
                            user_id=int(row["uid"]),
                            events=events,
                            steps=steps,
                            step_pos=step_pos,
                            history_len=args.history_len,
                        )
                        writer.add(replay_split, out_row)
                        split_counts[replay_split] += 1
                        total_stats[f"{replay_split}_steps"] += 1
                        transition_id += 1
        else:
            episodes = build_episodes(events, args.close_gap_seconds, args.anchor_policy, args.timestamp_unit_seconds)
            total_stats["episodes"] += len(episodes)
            split_plan = assign_splits(len(episodes))
            for split_name, episode in zip(split_plan, episodes):
                out_row = make_row(
                    transition_id=transition_id,
                    split_name=split_name,
                    user_id=int(row["uid"]),
                    events=events,
                    episode=episode,
                    history_len=args.history_len,
                    weights=weights,
                )
                writer.add(split_name, out_row)
                split_counts[split_name] += 1
                regret_counts[out_row["regret_type"]] += 1
                if len(reward_values) < 1_000_000:
                    reward_values.append(float(out_row["reward_scaled"]))
                transition_id += 1
        if users_processed % 200 == 0:
            pbar.set_postfix_str(
                f"train={split_counts['train']:,} val={split_counts['val']:,} test={split_counts['test']:,}",
                refresh=False,
            )
    pbar.close()
    writer.close()
    rewards = np.asarray(reward_values, dtype=np.float64)
    meta = {
        "args": vars(args),
        "resolved_session_gap_seconds": int(session_gap_seconds),
        "reward_weights": weights.__dict__,
        "users_processed": int(users_processed),
        "total_stats": {key: int(value) for key, value in total_stats.items()},
        "split_counts": {key: int(value) for key, value in split_counts.items()},
        "regret_counts": {key: int(value) for key, value in regret_counts.items()},
        "reward_sample": {
            "n": int(rewards.size),
            "mean": float(rewards.mean()) if rewards.size else None,
            "std": float(rewards.std()) if rewards.size else None,
            "min": float(rewards.min()) if rewards.size else None,
            "p50": float(np.quantile(rewards, 0.5)) if rewards.size else None,
            "p90": float(np.quantile(rewards, 0.9)) if rewards.size else None,
            "max": float(rewards.max()) if rewards.size else None,
            "negative_share": float((rewards < 0).mean()) if rewards.size else None,
        },
        "outputs": {
            split: str(out_root / split)
            for split in sorted(split_counts.keys())
        },
    }
    (out_root / "split.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] transitions saved to {out_root}")


if __name__ == "__main__":
    main()
