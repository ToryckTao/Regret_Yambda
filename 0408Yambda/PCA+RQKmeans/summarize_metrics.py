#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize SID variant metrics into a report table")
    parser.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--include_smoke", action="store_true", help="Include directories whose name starts with smoke")
    return parser.parse_args()


def safe_get(d: dict, path: list[str], default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def pick_reconstruction(m: dict) -> dict:
    recon = m.get("reconstruction", {})
    if "quantized_original_space" in recon:
        return recon["quantized_original_space"]
    return recon


def summarize_one(path: Path) -> dict:
    m = json.loads(path.read_text(encoding="utf-8"))
    per_level = m["sid_distribution"]["per_level"]
    util_avg = sum(x["utilization"] for x in per_level) / len(per_level)
    entropy_avg = sum(x["entropy_normalized"] for x in per_level) / len(per_level)
    gini_avg = sum(x["gini"] for x in per_level) / len(per_level)
    recon = pick_reconstruction(m)
    loc = m.get("sampled_locality", {})
    return {
        "variant": safe_get(m, ["extra", "variant"], path.parent.name),
        "metrics_path": str(path),
        "utilization_avg": util_avg,
        "entropy_norm_avg": entropy_avg,
        "gini_avg": gini_avg,
        "full_sid_excess_collision_rate": safe_get(m, ["collision", "full_sid", "excess_collision_rate"]),
        "full_sid_items_in_collision_groups_rate": safe_get(
            m, ["collision", "full_sid", "items_in_collision_groups_rate"]
        ),
        "recon_mse_original": recon.get("mse") if isinstance(recon, dict) else None,
        "recon_cosine_error_original": recon.get("mean_cosine_error") if isinstance(recon, dict) else None,
        "nn_mean_sid_hamming": loc.get("mean_sid_hamming"),
        "nn_mean_common_prefix_len": loc.get("mean_common_prefix_len"),
        "nn_same_level1_rate": loc.get("same_level1_rate"),
        "nn_same_level12_rate": loc.get("same_level12_rate"),
        "nn_same_full_sid_rate": loc.get("same_full_sid_rate"),
    }


def format_value(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def write_markdown(path: Path, rows: list[dict]) -> None:
    columns = [
        "variant",
        "utilization_avg",
        "entropy_norm_avg",
        "full_sid_excess_collision_rate",
        "recon_mse_original",
        "recon_cosine_error_original",
        "nn_same_level1_rate",
        "nn_same_level12_rate",
        "nn_mean_sid_hamming",
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(c)) for c in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    artifacts = Path(args.artifacts)
    out_dir = Path(args.out_dir) if args.out_dir else artifacts / "summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_paths = []
    for p in artifacts.rglob("metrics.json"):
        if "summary" in p.parts:
            continue
        if not args.include_smoke and any(part.startswith("smoke") for part in p.parts):
            continue
        metrics_paths.append(p)
    metrics_paths = sorted(metrics_paths)
    rows = [summarize_one(p) for p in metrics_paths]
    rows.sort(key=lambda x: x["variant"])

    json_path = out_dir / "sid_variant_comparison.json"
    csv_path = out_dir / "sid_variant_comparison.csv"
    md_path = out_dir / "sid_variant_comparison.md"

    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")
    write_markdown(md_path, rows)
    print(f"[done] summarized {len(rows)} metrics files into {out_dir}")


if __name__ == "__main__":
    main()
