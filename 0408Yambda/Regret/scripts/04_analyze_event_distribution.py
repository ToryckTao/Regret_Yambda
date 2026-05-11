#!/usr/bin/env python3
"""数据分析：统计 Yambda 原始事件分布，用来解释 reward 权重和论文数据描述。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent


LISTEN_BINS = [
    (0.0, 0.0, "eq_0"),
    (0.0, 0.05, "(0,0.05]"),
    (0.05, 0.10, "(0.05,0.10]"),
    (0.10, 0.20, "(0.10,0.20]"),
    (0.20, 0.40, "(0.20,0.40]"),
    (0.40, 0.60, "(0.40,0.60]"),
    (0.60, 0.80, "(0.60,0.80]"),
    (0.80, 0.95, "(0.80,0.95]"),
    (0.95, 1.00, "(0.95,1.00]"),
    (1.00, math.inf, ">1.00"),
]

GAP_BINS = [
    (0, 10, "<=10s"),
    (10, 60, "10s-1m"),
    (60, 300, "1m-5m"),
    (300, 3600, "5m-1h"),
    (3600, 86400, "1h-1d"),
    (86400, 604800, "1d-7d"),
    (604800, math.inf, ">7d"),
]

FEEDBACK_TYPES = {"like", "dislike", "unlike", "undislike"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze raw Yambda event distributions")
    parser.add_argument(
        "--multi_event_parquet",
        default=str(REPO_ROOT / "../0330Yambda/data/sequential-50m/multi_event.parquet"),
    )
    parser.add_argument("--timestamp_unit_seconds", type=float, default=5.0)
    parser.add_argument("--max_users", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_json", default=str(PROJECT_ROOT / "artifacts/diagnostics/event_distribution.json"))
    return parser.parse_args()


def iter_user_rows(parquet_path: Path, batch_size: int) -> Iterable[dict]:
    pf = pq.ParquetFile(parquet_path)
    columns = ["uid", "timestamp", "item_id", "played_ratio_pct", "event_type"]
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        pyd = batch.to_pydict()
        for idx in range(len(pyd["uid"])):
            yield {key: pyd[key][idx] for key in columns}


def update_numeric(values: list[float], value: float) -> None:
    if math.isfinite(value):
        values.append(float(value))


def quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ["min", "p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max"]}
    values = sorted(values)
    n = len(values)

    def at(q: float) -> float:
        idx = min(max(int(round(q * (n - 1))), 0), n - 1)
        return float(values[idx])

    return {
        "min": float(values[0]),
        "p01": at(0.01),
        "p05": at(0.05),
        "p10": at(0.10),
        "p25": at(0.25),
        "p50": at(0.50),
        "p75": at(0.75),
        "p90": at(0.90),
        "p95": at(0.95),
        "p99": at(0.99),
        "max": float(values[-1]),
    }


def binned_ratio(value: float) -> str:
    for lower, upper, label in LISTEN_BINS:
        if label == "eq_0" and value == 0.0:
            return label
        if value > lower and value <= upper:
            return label
    return ">1.00"


def binned_gap(seconds: float) -> str:
    for lower, upper, label in GAP_BINS:
        if seconds <= upper and seconds > lower:
            return label
        if label == "<=10s" and seconds >= 0 and seconds <= upper:
            return label
    return ">7d"


def summarize_list(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "std": None, **quantiles(values)}
    n = len(values)
    mean = sum(values) / n
    var = sum((x - mean) * (x - mean) for x in values) / n
    return {"n": n, "mean": float(mean), "std": float(math.sqrt(var)), **quantiles(values)}


def quantiles_from_counts(counts: Counter[int], scale: float = 100.0) -> dict[str, float | None]:
    total = sum(counts.values())
    if total <= 0:
        return {key: None for key in ["min", "p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max"]}
    sorted_items = sorted(counts.items())
    targets = {
        "p01": int(round(0.01 * (total - 1))) + 1,
        "p05": int(round(0.05 * (total - 1))) + 1,
        "p10": int(round(0.10 * (total - 1))) + 1,
        "p25": int(round(0.25 * (total - 1))) + 1,
        "p50": int(round(0.50 * (total - 1))) + 1,
        "p75": int(round(0.75 * (total - 1))) + 1,
        "p90": int(round(0.90 * (total - 1))) + 1,
        "p95": int(round(0.95 * (total - 1))) + 1,
        "p99": int(round(0.99 * (total - 1))) + 1,
    }
    out = {"min": float(sorted_items[0][0] / scale), "max": float(sorted_items[-1][0] / scale)}
    seen = 0
    pending = dict(targets)
    for raw_value, count in sorted_items:
        seen += count
        for key, target in list(pending.items()):
            if seen >= target:
                out[key] = float(raw_value / scale)
                del pending[key]
        if not pending:
            break
    return {key: out.get(key) for key in ["min", "p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max"]}


def summarize_ratio_counts(counts: Counter[int], scale: float = 100.0) -> dict:
    total = sum(counts.values())
    if total <= 0:
        return {"n": 0, "mean": None, "std": None, **quantiles_from_counts(counts, scale)}
    sum_x = sum((raw / scale) * count for raw, count in counts.items())
    mean = sum_x / total
    var = sum((((raw / scale) - mean) ** 2) * count for raw, count in counts.items()) / total
    return {
        "n": int(total),
        "mean": float(mean),
        "std": float(math.sqrt(var)),
        **quantiles_from_counts(counts, scale),
    }


def analyze(args: argparse.Namespace) -> dict:
    event_counts: Counter[str] = Counter()
    listen_bins: Counter[str] = Counter()
    listen_ratio_counts: Counter[int] = Counter()
    feedback_prev_listen_ratios: dict[str, list[float]] = defaultdict(list)
    feedback_prev_listen_gaps: dict[str, list[float]] = defaultdict(list)
    pair_gaps: dict[str, list[float]] = defaultdict(list)
    pair_gap_bins: dict[str, Counter[str]] = defaultdict(Counter)
    pair_counts: Counter[str] = Counter()
    per_user_events: list[float] = []
    users_processed = 0
    total_events = 0
    pbar = tqdm(iter_user_rows(Path(args.multi_event_parquet), args.batch_size), desc="[analyze]", unit="users", ncols=80)
    for row in pbar:
        if args.max_users and users_processed >= args.max_users:
            break
        users_processed += 1
        n_events = len(row["timestamp"])
        total_events += n_events
        per_user_events.append(float(n_events))

        last_like_time: dict[int, int] = {}
        last_dislike_time: dict[int, int] = {}
        last_listen: dict[int, tuple[int, float]] = {}
        order = sorted(range(n_events), key=lambda idx: (int(row["timestamp"][idx]), idx))
        for pos in order:
            event_type = str(row["event_type"][pos])
            event_counts[event_type] += 1
            timestamp = int(row["timestamp"][pos])
            item_id = int(row["item_id"][pos])
            ratio_raw = row["played_ratio_pct"][pos]
            ratio = 0.0 if ratio_raw is None else float(ratio_raw) / 100.0
            if event_type == "listen":
                listen_ratio_counts[int(ratio_raw or 0)] += 1
                listen_bins[binned_ratio(ratio)] += 1
                last_listen[item_id] = (timestamp, ratio)
                continue

            if event_type in FEEDBACK_TYPES and item_id in last_listen:
                prev_ts, prev_ratio = last_listen[item_id]
                gap = (timestamp - prev_ts) * float(args.timestamp_unit_seconds)
                if gap >= 0:
                    update_numeric(feedback_prev_listen_ratios[event_type], prev_ratio)
                    update_numeric(feedback_prev_listen_gaps[event_type], float(gap))

            if event_type == "like":
                last_like_time[item_id] = timestamp
            elif event_type == "unlike":
                if item_id in last_like_time:
                    gap = (timestamp - last_like_time[item_id]) * float(args.timestamp_unit_seconds)
                    if gap >= 0:
                        pair_gaps["like_to_unlike"].append(float(gap))
                        pair_gap_bins["like_to_unlike"][binned_gap(gap)] += 1
                        pair_counts["unlike_with_prior_like"] += 1
                    else:
                        pair_counts["unlike_prior_like_negative_gap"] += 1
                    del last_like_time[item_id]
                else:
                    pair_counts["unlike_without_prior_like"] += 1
            elif event_type == "dislike":
                last_dislike_time[item_id] = timestamp
            elif event_type == "undislike":
                if item_id in last_dislike_time:
                    gap = (timestamp - last_dislike_time[item_id]) * float(args.timestamp_unit_seconds)
                    if gap >= 0:
                        pair_gaps["dislike_to_undislike"].append(float(gap))
                        pair_gap_bins["dislike_to_undislike"][binned_gap(gap)] += 1
                        pair_counts["undislike_with_prior_dislike"] += 1
                    else:
                        pair_counts["undislike_prior_dislike_negative_gap"] += 1
                    del last_dislike_time[item_id]
                else:
                    pair_counts["undislike_without_prior_dislike"] += 1

        if users_processed % 200 == 0:
            pbar.set_postfix_str(f"events={total_events:,}", refresh=False)
    pbar.close()

    event_share = {
        key: float(value / max(total_events, 1))
        for key, value in sorted(event_counts.items(), key=lambda item: (-item[1], item[0]))
    }
    listen_total = event_counts.get("listen", 0)
    feedback_rates_per_1k_listen = {
        key: float(event_counts.get(key, 0) * 1000.0 / max(listen_total, 1))
        for key in sorted(FEEDBACK_TYPES)
    }
    listen_bin_share = {key: float(value / max(listen_total, 1)) for key, value in listen_bins.items()}
    pair_summary = {}
    for key, gaps in pair_gaps.items():
        pair_summary[key] = {
            "gap_seconds": summarize_list(gaps),
            "gap_bins": dict(pair_gap_bins[key]),
            "gap_bin_share": {
                bin_key: float(bin_value / max(len(gaps), 1))
                for bin_key, bin_value in pair_gap_bins[key].items()
            },
        }
    feedback_context = {}
    for key in sorted(FEEDBACK_TYPES):
        feedback_context[key] = {
            "prev_listen_ratio": summarize_list(feedback_prev_listen_ratios[key]),
            "prev_listen_gap_seconds": summarize_list(feedback_prev_listen_gaps[key]),
            "with_prior_listen_share": float(
                len(feedback_prev_listen_ratios[key]) / max(event_counts.get(key, 0), 1)
            ),
        }
    return {
        "args": vars(args),
        "users_processed": int(users_processed),
        "total_events": int(total_events),
        "per_user_events": summarize_list(per_user_events),
        "event_counts": dict(event_counts),
        "event_share": event_share,
        "feedback_rates_per_1k_listen": feedback_rates_per_1k_listen,
        "listen_ratio": summarize_ratio_counts(listen_ratio_counts),
        "listen_ratio_bins": dict(listen_bins),
        "listen_ratio_bin_share": listen_bin_share,
        "feedback_pair_counts": dict(pair_counts),
        "feedback_pair_summary": pair_summary,
        "feedback_context": feedback_context,
    }


def main() -> None:
    args = parse_args()
    result = analyze(args)
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[done] diagnostics saved to {out_path}")


if __name__ == "__main__":
    main()
