#!/usr/bin/env python3
"""
把 Yambda 的 multi_event 直接流式切成 train / val / test。

设计目标：
1. 不再先落一个全量巨型 rich 中间表
2. 直接按用户时间顺序，把 decision sample 分发到 train / val / test
3. 保留后续 HPN / RL 都会用到的关键字段

输出列：
- sequence_id
- user_id
- anchor_time
- anchor_event_type
- target_orig_item_id
- target_dense_item_id
- slate_of_items
- feedback_label
- user_clicks
- user_mid_history
- user_click_history
- next_user_mid_history
- next_user_click_history
- user_like_history
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_YAMBA_DATA_DIR = Path(os.environ.get("YAMBA_DATA_DIR", "/Users/Toryck/Coding/DATASET/Yambda"))


def parse_args() -> argparse.Namespace:
    """输入：命令行参数。输出：参数对象 Namespace。"""
    parser = argparse.ArgumentParser(description="Split Yambda decision samples into train/val/test TSV files")
    parser.add_argument(
        "--multi_event_parquet",
        type=str,
        default=str(DEFAULT_YAMBA_DATA_DIR / "sequential/50m/multi_event.parquet"),
        help="Yambda sequential multi_event parquet path",
    )
    parser.add_argument(
        "--orig2dense_npy",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/mappings/yambda_orig2dense_item_id.npy"),
        help="orig_item_id -> dense_item_id mapping path",
    )
    parser.add_argument(
        "--history_len",
        type=int,
        default=50,
        help="状态里保留多少个历史事件",
    )
    parser.add_argument(
        "--close_gap_seconds",
        type=int,
        default=3600,
        help="同一 (user,item) 两次事件超过该间隔时切新 episode",
    )
    parser.add_argument(
        "--anchor_policy",
        type=str,
        default="first_visible",
        choices=["first_visible", "listen_only"],
        help="episode 保留策略",
    )
    parser.add_argument(
        "--positive_play_threshold",
        type=float,
        default=0.8,
        help="高完成度 listen 视作正反馈的阈值",
    )
    parser.add_argument(
        "--reward_w_play",
        type=float,
        default=1.0,
        help="reward 里 max_play_ratio 的权重",
    )
    parser.add_argument(
        "--reward_w_like",
        type=float,
        default=1.0,
        help="reward 里 like 的权重",
    )
    parser.add_argument(
        "--reward_w_dislike",
        type=float,
        default=1.0,
        help="reward 里 dislike 的负权重",
    )
    parser.add_argument(
        "--reward_w_unlike",
        type=float,
        default=0.0,
        help="reward 里 unlike 的负权重。第一版默认不并入主 reward",
    )
    parser.add_argument(
        "--reward_w_undislike",
        type=float,
        default=0.0,
        help="reward 里 undislike 的正权重。第一版默认不并入主 reward",
    )
    parser.add_argument(
        "--history_like_value",
        type=float,
        default=1.0,
        help="历史里的 like 信号值",
    )
    parser.add_argument(
        "--history_dislike_value",
        type=float,
        default=-1.0,
        help="历史里的 dislike 信号值",
    )
    parser.add_argument(
        "--history_unlike_value",
        type=float,
        default=-0.5,
        help="历史里的 unlike 信号值",
    )
    parser.add_argument(
        "--history_undislike_value",
        type=float,
        default=0.5,
        help="历史里的 undislike 信号值",
    )
    parser.add_argument(
        "--max_users",
        type=int,
        default=0,
        help="调试开关。0 表示处理全部用户",
    )
    parser.add_argument(
        "--output_train_tsv",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/processed/train.tsv"),
        help="训练集 TSV 输出路径",
    )
    parser.add_argument(
        "--output_val_tsv",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/processed/val.tsv"),
        help="验证集 TSV 输出路径",
    )
    parser.add_argument(
        "--output_test_tsv",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/processed/test.tsv"),
        help="测试集 TSV 输出路径",
    )
    parser.add_argument(
        "--output_meta",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/processed/split.meta.json"),
        help="切分元信息输出路径",
    )
    return parser.parse_args()


def iter_user_rows(parquet_path: Path, batch_size: int = 32) -> Iterable[dict]:
    """
    输入：
    - parquet_path: Yambda multi_event.parquet 路径
    - batch_size: 每批读取多少个用户行

    输出：
    - 逐个 yield 用户行字典；每行内的 timestamp/item_id/event_type 都是该用户的事件列表
    """
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
        n_user = len(pyd["uid"])
        for i in range(n_user):
            yield {k: pyd[k][i] for k in columns}


def history_signal_from_event(event: dict, args: argparse.Namespace) -> float:
    """
    输入：
    - event: 一个原子交互事件
    - args: reward/history 参数

    输出：
    - 写入 user_click_history 的标量信号
    """
    event_type = event["event_type"]
    if event_type == "listen":
        return float(event["played_ratio_norm"])
    if event_type == "like":
        return float(args.history_like_value)
    if event_type == "dislike":
        return float(args.history_dislike_value)
    if event_type == "unlike":
        return float(args.history_unlike_value)
    if event_type == "undislike":
        return float(args.history_undislike_value)
    return 0.0


def expand_user_events(row: dict, orig2dense: np.ndarray) -> tuple[list[dict], dict]:
    """
    输入：
    - row: multi_event 里一个用户的一整行
    - orig2dense: 原始 item_id 到 dense item_id 的映射

    输出：
    - events: 展开的时间有序原子事件
    - stats: 缺 embedding / 事件类型统计
    """
    timestamps = row["timestamp"]
    item_ids = row["item_id"]
    is_organic = row["is_organic"]
    played_ratio_pct = row["played_ratio_pct"]
    track_lengths = row["track_length_seconds"]
    event_types = row["event_type"]

    events = []
    skipped_missing_mapping = 0
    raw_event_type_counts: Counter[str] = Counter()
    kept_event_type_counts: Counter[str] = Counter()
    missing_mapping_event_type_counts: Counter[str] = Counter()

    for i, orig_item_id_raw in enumerate(item_ids):
        orig_item_id = int(orig_item_id_raw)
        event_type = str(event_types[i])
        raw_event_type_counts[event_type] += 1

        if orig_item_id >= len(orig2dense):
            skipped_missing_mapping += 1
            missing_mapping_event_type_counts[event_type] += 1
            continue

        dense_item_id = int(orig2dense[orig_item_id])
        if dense_item_id <= 0:
            skipped_missing_mapping += 1
            missing_mapping_event_type_counts[event_type] += 1
            continue

        ratio = played_ratio_pct[i]
        ratio_norm = 0.0 if ratio is None else float(ratio) / 100.0
        kept_event_type_counts[event_type] += 1

        events.append(
            {
                "uid": int(row["uid"]),
                "timestamp": int(timestamps[i]),
                "orig_item_id": orig_item_id,
                "dense_item_id": dense_item_id,
                "is_organic": None if is_organic[i] is None else int(is_organic[i]),
                "played_ratio_pct": None if ratio is None else int(ratio),
                "played_ratio_norm": ratio_norm,
                "track_length_seconds": None if track_lengths[i] is None else int(track_lengths[i]),
                "event_type": event_type,
                "raw_pos": i,
            }
        )

    events.sort(key=lambda x: (x["timestamp"], x["raw_pos"]))
    return events, {
        "total_events_raw": len(timestamps),
        "total_events_kept": len(events),
        "skipped_missing_mapping": skipped_missing_mapping,
        "raw_event_type_counts": dict(raw_event_type_counts),
        "kept_event_type_counts": dict(kept_event_type_counts),
        "missing_mapping_event_type_counts": dict(missing_mapping_event_type_counts),
    }


def build_episodes(events: list[dict], close_gap_seconds: int, anchor_policy: str) -> list[dict]:
    """
    输入：
    - events: 一个用户的全局时间排序事件流
    - close_gap_seconds: 同一 item episode 的关闭阈值
    - anchor_policy: first_visible 或 listen_only

    输出：
    - episodes: 每条包含该 episode 的 event_indices
    """
    episodes: list[dict] = []
    open_episode_by_item: dict[int, dict] = {}

    for global_idx, event in enumerate(events):
        item_id = int(event["dense_item_id"])
        current = open_episode_by_item.get(item_id)
        should_open = current is None or (event["timestamp"] - current["last_timestamp"] > close_gap_seconds)

        if should_open:
            current = {
                "dense_item_id": event["dense_item_id"],
                "orig_item_id": event["orig_item_id"],
                "start_idx": global_idx,
                "end_idx": global_idx,
                "last_timestamp": event["timestamp"],
                "event_indices": [global_idx],
            }
            episodes.append(current)
            open_episode_by_item[item_id] = current
        else:
            current["end_idx"] = global_idx
            current["last_timestamp"] = event["timestamp"]
            current["event_indices"].append(global_idx)

    if anchor_policy == "listen_only":
        episodes = [ep for ep in episodes if events[ep["event_indices"][0]]["event_type"] == "listen"]

    episodes.sort(key=lambda ep: ep["start_idx"])
    return episodes


def aggregate_episode_feedback(episode_events: list[dict], args: argparse.Namespace) -> tuple[float, int, dict]:
    """
    输入：
    - episode_events: 同一 user-item episode 内的全部事件
    - args: reward 参数

    输出：
    - reward: episode 聚合出的标量奖励
    - binary_feedback: 给 user response / BC 使用的二值反馈
    - summary: 便于元信息排查的摘要
    """
    event_types = [event["event_type"] for event in episode_events]
    play_ratios = [event["played_ratio_norm"] for event in episode_events if event["event_type"] == "listen"]

    max_play_ratio = max(play_ratios) if play_ratios else 0.0
    mean_play_ratio = float(np.mean(play_ratios)) if play_ratios else 0.0
    has_like = int("like" in event_types)
    has_dislike = int("dislike" in event_types)
    has_unlike = int("unlike" in event_types)
    has_undislike = int("undislike" in event_types)

    reward = (
        args.reward_w_play * max_play_ratio
        + args.reward_w_like * has_like
        - args.reward_w_dislike * has_dislike
        - args.reward_w_unlike * has_unlike
        + args.reward_w_undislike * has_undislike
    )

    positive_signal = bool(has_like or (max_play_ratio >= args.positive_play_threshold))
    negative_signal = bool(has_dislike)
    binary_feedback = int(positive_signal and not negative_signal)
    summary = {
        "n_events": len(episode_events),
        "event_types": event_types,
        "n_listen": int(sum(1 for item in event_types if item == "listen")),
        "max_play_ratio": float(max_play_ratio),
        "mean_play_ratio": float(mean_play_ratio),
        "has_like": has_like,
        "has_dislike": has_dislike,
        "has_unlike": has_unlike,
        "has_undislike": has_undislike,
        "positive_signal": int(positive_signal),
        "negative_signal": int(negative_signal),
    }
    return float(reward), binary_feedback, summary


def assign_user_splits(n_episode: int) -> list[str]:
    """
    输入：
    - n_episode: 某个用户最终保留下来的 episode 数量

    输出：
    - 与 episode 数量等长的 split 列表，每个元素是 train/val/test

    规则：
    - 1 条：train
    - 2 条：train, test
    - >=3 条：前面 train，倒数第二条 val，最后一条 test
    """
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


def to_list_str(seq: list[Any]) -> str:
    """输入：Python list。输出：兼容后续 TSV 读取的字符串。"""
    return json.dumps(seq, ensure_ascii=True, separators=(",", ":"))


def make_split_row(
    sample_id: int,
    user_id: int,
    target_event: dict,
    history_events: list[dict],
    next_history_events: list[dict],
    reward: float,
    feedback_label: int,
) -> dict[str, Any]:
    """
    输入：
    - 当前样本的核心信息
    输出：
    - 一行 split TSV 记录
    """
    history_ids = [int(ev["dense_item_id"]) for ev in history_events]
    history_signals = [float(ev["history_signal"]) for ev in history_events]
    next_history_ids = [int(ev["dense_item_id"]) for ev in next_history_events]
    next_history_signals = [float(ev["history_signal"]) for ev in next_history_events]
    target_dense_item_id = int(target_event["dense_item_id"])

    return {
        "sequence_id": sample_id,
        "user_id": int(user_id),
        "anchor_time": int(target_event["timestamp"]),
        "anchor_event_type": str(target_event["event_type"]),
        "target_orig_item_id": int(target_event["orig_item_id"]),
        "target_dense_item_id": target_dense_item_id,
        "slate_of_items": to_list_str([target_dense_item_id]),
        "feedback_label": int(feedback_label),
        "user_clicks": float(reward),
        "user_mid_history": to_list_str(history_ids),
        "user_click_history": to_list_str(history_signals),
        "next_user_mid_history": to_list_str(next_history_ids),
        "next_user_click_history": to_list_str(next_history_signals),
        "user_like_history": "[]",
    }


def main() -> None:
    """主入口：从 multi_event 流式生成并切分 Yambda HSRL 样本。"""
    args = parse_args()

    parquet_path = Path(args.multi_event_parquet)
    orig2dense_path = Path(args.orig2dense_npy)
    if not orig2dense_path.exists():
        raise FileNotFoundError(
            f"Missing mapping file: {orig2dense_path}. "
            "请先运行 02_build_item_sid.py 生成 dense 映射。"
        )

    train_path = Path(args.output_train_tsv)
    val_path = Path(args.output_val_tsv)
    test_path = Path(args.output_test_tsv)
    meta_path = Path(args.output_meta)
    for p in [train_path, val_path, test_path, meta_path]:
        p.parent.mkdir(parents=True, exist_ok=True)

    orig2dense = np.load(orig2dense_path, mmap_mode="r")
    columns = [
        "sequence_id",
        "user_id",
        "anchor_time",
        "anchor_event_type",
        "target_orig_item_id",
        "target_dense_item_id",
        "slate_of_items",
        "feedback_label",
        "user_clicks",
        "user_mid_history",
        "user_click_history",
        "next_user_mid_history",
        "next_user_click_history",
        "user_like_history",
    ]

    total_users = 0
    total_raw_events = 0
    total_kept_events = 0
    total_missing_mapping = 0
    split_counts = {"train": 0, "val": 0, "test": 0}
    next_sequence_id = 1

    train_f = train_path.open("w", encoding="utf-8", newline="")
    val_f = val_path.open("w", encoding="utf-8", newline="")
    test_f = test_path.open("w", encoding="utf-8", newline="")
    try:
        writers = {
            "train": csv.DictWriter(train_f, fieldnames=columns, delimiter="\t"),
            "val": csv.DictWriter(val_f, fieldnames=columns, delimiter="\t"),
            "test": csv.DictWriter(test_f, fieldnames=columns, delimiter="\t"),
        }
        for writer in writers.values():
            writer.writeheader()

        for row in iter_user_rows(parquet_path):
            total_users += 1
            if args.max_users and total_users > args.max_users:
                break

            events, stats = expand_user_events(row, orig2dense)
            total_raw_events += stats["total_events_raw"]
            total_kept_events += stats["total_events_kept"]
            total_missing_mapping += stats["skipped_missing_mapping"]

            if not events:
                continue

            for ev in events:
                ev["history_signal"] = history_signal_from_event(ev, args)

            episodes = build_episodes(
                events=events,
                close_gap_seconds=args.close_gap_seconds,
                anchor_policy=args.anchor_policy,
            )
            if not episodes:
                continue

            split_plan = assign_user_splits(len(episodes))
            for split_name, ep in zip(split_plan, episodes):
                episode_indices = ep["event_indices"]
                start_idx = episode_indices[0]
                end_idx = episode_indices[-1]

                episode_events = [events[i] for i in episode_indices]
                target_event = events[start_idx]
                history_events = events[max(0, start_idx - args.history_len): start_idx]
                next_history_events = events[max(0, end_idx + 1 - args.history_len): end_idx + 1]

                reward, binary_feedback, _summary = aggregate_episode_feedback(episode_events, args)
                split_row = make_split_row(
                    sample_id=next_sequence_id,
                    user_id=int(row["uid"]),
                    target_event=target_event,
                    history_events=history_events,
                    next_history_events=next_history_events,
                    reward=reward,
                    feedback_label=binary_feedback,
                )
                writers[split_name].writerow(split_row)
                split_counts[split_name] += 1
                next_sequence_id += 1

            if total_users % 200 == 0:
                print(
                    f"[progress] users={total_users:,}, train={split_counts['train']:,}, "
                    f"val={split_counts['val']:,}, test={split_counts['test']:,}, "
                    f"missing_mapping={total_missing_mapping:,}"
                )
    finally:
        train_f.close()
        val_f.close()
        test_f.close()

    meta = {
        "multi_event_parquet": str(parquet_path),
        "orig2dense_npy": str(orig2dense_path),
        "history_len": int(args.history_len),
        "close_gap_seconds": int(args.close_gap_seconds),
        "anchor_policy": args.anchor_policy,
        "positive_play_threshold": float(args.positive_play_threshold),
        "reward_weights": {
            "play": float(args.reward_w_play),
            "like": float(args.reward_w_like),
            "dislike": float(args.reward_w_dislike),
            "unlike": float(args.reward_w_unlike),
            "undislike": float(args.reward_w_undislike),
        },
        "run_stats": {
            "users_processed": int(min(total_users, args.max_users) if args.max_users else total_users),
            "total_raw_events": int(total_raw_events),
            "total_kept_events": int(total_kept_events),
            "total_missing_mapping": int(total_missing_mapping),
            "split_counts": {k: int(v) for k, v in split_counts.items()},
            "sequence_id_max": int(next_sequence_id - 1),
        },
        "outputs": {
            "train_tsv": str(train_path),
            "val_tsv": str(val_path),
            "test_tsv": str(test_path),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("[done]")
    print(f"  train: {train_path}")
    print(f"  val:   {val_path}")
    print(f"  test:  {test_path}")
    print(f"  meta:  {meta_path}")


if __name__ == "__main__":
    main()
