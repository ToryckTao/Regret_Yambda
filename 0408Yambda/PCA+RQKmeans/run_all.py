#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_EMBEDDINGS = "/Users/Toryck/Coding/DATASET/Yambda/embeddings.parquet"


PCA_VARIANTS = {
    "pca32_nowhiten": {"pca_dim": 32, "whiten": False},
    "pca64_nowhiten": {"pca_dim": 64, "whiten": False},
    "pca96_nowhiten": {"pca_dim": 96, "whiten": False},
    "pca64_whiten": {"pca_dim": 64, "whiten": True},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run semantic-ID baseline comparisons")
    parser.add_argument("--embeddings", default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--embedding_column", default="normalized_embed", choices=["embed", "normalized_embed"])
    parser.add_argument("--out_root", default=str(ROOT / "artifacts"))
    parser.add_argument("--max_rows", type=int, default=0)
    parser.add_argument("--train_rows", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=65536)
    parser.add_argument("--kmeans_iter", type=int, default=50)
    parser.add_argument("--nredo", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_points_per_centroid", type=int, default=100000)
    parser.add_argument("--locality_sample_size", type=int, default=200000)
    parser.add_argument("--locality_query_size", type=int, default=5000)
    parser.add_argument("--locality_topk", type=int, default=10)
    parser.add_argument("--rqvae_ckpt", default=None)
    parser.add_argument(
        "--only",
        nargs="*",
        default=["raw", "pca32_nowhiten", "pca64_nowhiten", "pca96_nowhiten", "pca64_whiten", "rqvae"],
        help="Subset: raw, pca32_nowhiten, pca64_nowhiten, pca96_nowhiten, pca64_whiten, rqvae",
    )
    parser.add_argument("--quiet_faiss", action="store_true")
    return parser.parse_args()


def base_args(args: argparse.Namespace) -> list[str]:
    return [
        "--embeddings", args.embeddings,
        "--embedding_column", args.embedding_column,
        "--max_rows", str(args.max_rows),
        "--batch_size", str(args.batch_size),
        "--locality_sample_size", str(args.locality_sample_size),
        "--locality_query_size", str(args.locality_query_size),
        "--locality_topk", str(args.locality_topk),
    ]


def run(cmd: list[str]) -> None:
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    selected = set(args.only)

    if "raw" in selected:
        cmd = [
            sys.executable,
            str(ROOT / "rawRQKmeans" / "run_raw_rqkmeans.py"),
            *base_args(args),
            "--out_dir", str(Path(args.out_root) / "rawRQKmeans"),
            "--train_rows", str(args.train_rows),
            "--kmeans_iter", str(args.kmeans_iter),
            "--nredo", str(args.nredo),
            "--seed", str(args.seed),
            "--max_points_per_centroid", str(args.max_points_per_centroid),
        ]
        if args.quiet_faiss:
            cmd.append("--quiet_faiss")
        run(cmd)

    for name, cfg in PCA_VARIANTS.items():
        if name not in selected:
            continue
        cmd = [
            sys.executable,
            str(ROOT / "PCA_RQKmeans" / "run_pca_rqkmeans.py"),
            *base_args(args),
            "--out_root", str(Path(args.out_root) / "PCA_RQKmeans"),
            "--variant_name", name,
            "--pca_dim", str(cfg["pca_dim"]),
            "--train_rows", str(args.train_rows),
            "--kmeans_iter", str(args.kmeans_iter),
            "--nredo", str(args.nredo),
            "--seed", str(args.seed),
            "--max_points_per_centroid", str(args.max_points_per_centroid),
        ]
        if cfg["whiten"]:
            cmd.append("--whiten")
        if args.quiet_faiss:
            cmd.append("--quiet_faiss")
        run(cmd)

    if "rqvae" in selected:
        if not args.rqvae_ckpt:
            print("[skip] rqvae selected but --rqvae_ckpt is empty")
        else:
            run([
                sys.executable,
                str(ROOT / "RQVAE" / "run_rqvae_diagnostics.py"),
                *base_args(args),
                "--out_dir", str(Path(args.out_root) / "RQVAE"),
                "--rqvae_ckpt", args.rqvae_ckpt,
            ])


if __name__ == "__main__":
    main()
