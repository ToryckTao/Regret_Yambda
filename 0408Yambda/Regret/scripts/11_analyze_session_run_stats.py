#!/usr/bin/env python3
"""数据统计：统计 session 长度、step 数量、失败记忆池覆盖等 session-run 切分质量。"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze session-run transition granularity.")
    parser.add_argument(
        "--transition_root",
        default="artifacts/transitions/raw_rqkmeans_v2_session_run_1h_span6h_run100_replay50",
    )
    parser.add_argument("--split", default="train", help="Split directory under transition_root, or 'all'.")
    parser.add_argument("--batch_size", type=int, default=262144)
    parser.add_argument("--max_files", type=int, default=0)
    parser.add_argument("--out_json", default="")
    return parser.parse_args()


def list_parquet_files(root: Path, split: str) -> list[Path]:
    if split == "all":
        files: list[Path] = []
        for name in ("train", "val", "test", "replay_val", "replay_test"):
            files.extend(sorted((root / name).glob("*.parquet")))
        return files
    path = root / split
    if path.is_file():
        return [path]
    return sorted(path.glob("*.parquet"))


def weighted_quantile(counter: Counter[int], q: float) -> int:
    if not counter:
        return 0
    total = sum(counter.values())
    threshold = max(1, int((total * q) + 0.999999))
    running = 0
    for value in sorted(counter):
        running += counter[value]
        if running >= threshold:
            return int(value)
    return int(max(counter))


def summarize_counter(counter: Counter[int]) -> dict[str, float]:
    n = int(sum(counter.values()))
    if n == 0:
        return {"n": 0, "mean": 0.0, "min": 0, "p50": 0, "p90": 0, "p99": 0, "max": 0}
    total = sum(int(value) * int(count) for value, count in counter.items())
    return {
        "n": n,
        "mean": float(total / n),
        "min": int(min(counter)),
        "p50": int(weighted_quantile(counter, 0.50)),
        "p90": int(weighted_quantile(counter, 0.90)),
        "p99": int(weighted_quantile(counter, 0.99)),
        "max": int(max(counter)),
    }


def iter_batches(files: Iterable[Path], batch_size: int):
    columns = ["user_id", "session_id", "session_event_count", "n_events"]
    for file_path in files:
        parquet_file = pq.ParquetFile(file_path)
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            yield file_path, batch.to_pydict()


def main() -> None:
    args = parse_args()
    root = Path(args.transition_root)
    files = list_parquet_files(root, args.split)
    if args.max_files > 0:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No parquet files found for split={args.split} under {root}")

    user_session_counts: Counter[int] = Counter()
    session_step_counts: dict[int, int] = defaultdict(int)
    session_event_counts: dict[int, int] = {}
    step_event_count_hist: Counter[int] = Counter()
    users_seen: set[int] = set()
    rows = 0

    for _file_path, batch in iter_batches(files, int(args.batch_size)):
        user_ids = batch["user_id"]
        session_ids = batch["session_id"]
        session_event_counts_batch = batch["session_event_count"]
        n_events = batch["n_events"]
        for user_id, session_id, session_event_count, step_event_count in zip(
            user_ids, session_ids, session_event_counts_batch, n_events
        ):
            user_id = int(user_id)
            session_id = int(session_id)
            session_key = (user_id << 32) | session_id
            if session_key not in session_event_counts:
                session_event_counts[session_key] = int(session_event_count)
                user_session_counts[user_id] += 1
                users_seen.add(user_id)
            session_step_counts[session_key] += 1
            step_event_count_hist[int(step_event_count)] += 1
            rows += 1

    session_event_hist = Counter(session_event_counts.values())
    session_step_hist = Counter(session_step_counts.values())
    user_session_hist = Counter(user_session_counts.values())

    result = {
        "transition_root": str(root),
        "split": args.split,
        "files": len(files),
        "rows_steps": rows,
        "users": len(users_seen),
        "sessions": len(session_event_counts),
        "avg_events_per_step": float(
            sum(value * count for value, count in step_event_count_hist.items()) / max(rows, 1)
        ),
        "user_sessions": summarize_counter(user_session_hist),
        "session_events": summarize_counter(session_event_hist),
        "session_steps": summarize_counter(session_step_hist),
        "step_events": summarize_counter(step_event_count_hist),
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
