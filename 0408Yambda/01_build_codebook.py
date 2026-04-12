#!/usr/bin/env python3
"""
在 Yambda embeddings 上训练 HSRL 风格的残差平衡 KMeans codebook。

这个脚本参考原 HSRL 的 build_codebook / k_means_acc 思路。

相对原版的改动：
  1. 输入改为 embeddings.parquet
  2. 用 pyarrow 流式读取，不一次性读入全量 13G 文件
  3. 默认先做 reservoir sampling，再在样本上训练 codebook
  4. 默认使用 normalized_embed

输出：
  - 一个保存 codebooks 的 .npz
  - 一个记录训练参数和统计信息的 .json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq
import torch


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_YAMBA_DATA_DIR = Path(os.environ.get("YAMBA_DATA_DIR", "/Users/Toryck/Coding/DATASET/Yambda"))


class SemanticIDGeneratorTorch:
    """原始 HSRL 预编码里用到的“逐层残差 + 平衡 KMeans”实现。"""

    def __init__(
        self,
        n_levels: int = 4,
        codebook_size: int = 256,
        device: str = "cuda",
        max_iter: int = 30,
    ) -> None:
        self.n_levels = n_levels
        self.codebook_size = codebook_size
        self.max_iter = max_iter
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.codebooks: list[torch.Tensor | None] = [None for _ in range(n_levels)]

    def balanced_kmeans(
        self,
        data: torch.Tensor,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        输入：
          data: [n_samples, dim] 的向量矩阵
          k: 当前层 codebook 大小
        输出：
          centers: [k, dim]，当前层聚类中心
          labels:  [n_samples]，每个样本分到哪个中心
        """
        n_samples = data.shape[0]
        if k > n_samples:
            raise ValueError(f"codebook_size={k} cannot exceed sample_size={n_samples}")

        init_idx = torch.randperm(n_samples, device=data.device)[:k]
        centers = data[init_idx].clone()
        labels = torch.zeros(n_samples, dtype=torch.long, device=data.device)

        target_size = n_samples // k
        remainder = n_samples % k

        for iter_idx in range(self.max_iter):
            distances = torch.cdist(data, centers)
            cluster_sizes = torch.zeros(k, dtype=torch.long, device=data.device)
            new_labels = torch.empty_like(labels)

            # Keep the original balanced assignment rule for HSRL compatibility.
            for i in range(n_samples):
                nearest = torch.argsort(distances[i])
                for idx in nearest:
                    cap = target_size + (1 if int(idx) < remainder else 0)
                    if cluster_sizes[idx] < cap:
                        new_labels[i] = idx
                        cluster_sizes[idx] += 1
                        break

            new_centers = torch.zeros_like(centers)
            for c in range(k):
                mask = new_labels == c
                if mask.any():
                    new_centers[c] = data[mask].mean(dim=0)
                else:
                    new_centers[c] = centers[c]

            if torch.allclose(centers, new_centers, atol=1e-4):
                labels = new_labels
                print(f"[balanced_kmeans] converged at iter {iter_idx + 1}")
                break

            centers = new_centers
            labels = new_labels
        else:
            print(f"[balanced_kmeans] reached max_iter={self.max_iter} without full convergence")

        return centers, labels

    def fit(self, embeddings: np.ndarray) -> "SemanticIDGeneratorTorch":
        emb = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
        residual = emb.clone()

        for level in range(self.n_levels):
            print(f"[fit] training level {level + 1}/{self.n_levels}")
            codebook, _ = self.balanced_kmeans(residual, self.codebook_size)
            self.codebooks[level] = codebook.detach().cpu()

            if level < self.n_levels - 1:
                dists = torch.cdist(residual, codebook)
                nearest = dists.argmin(dim=1)
                residual = residual - codebook[nearest]
                residual_norm = torch.norm(residual).item()
                print(f"[fit] residual norm after level {level + 1}: {residual_norm:.6f}")
                if residual_norm < 1e-6:
                    print(f"[fit] residual nearly zero, stop early at level {level + 1}")
                    break

        return self


def parse_args() -> argparse.Namespace:
    """输入：命令行参数。输出：参数对象 Namespace。"""
    parser = argparse.ArgumentParser(description="Train residual balanced k-means codebook on Yambda embeddings")
    parser.add_argument(
        "--embeddings_parquet",
        type=str,
        default=str(DEFAULT_YAMBA_DATA_DIR / "embeddings.parquet"),
        help="Path to Yambda embeddings.parquet",
    )
    parser.add_argument(
        "--embedding_column",
        type=str,
        default="normalized_embed",
        choices=["embed", "normalized_embed"],
        help="Which embedding column to use for codebook training",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=200000,
        help="Number of embeddings sampled for codebook fitting",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        help="Parquet streaming batch size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reservoir sampling",
    )
    parser.add_argument(
        "--n_levels",
        type=int,
        default=4,
        help="Number of residual quantization levels",
    )
    parser.add_argument(
        "--codebook_size",
        type=int,
        default=256,
        help="Vocabulary size per level",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=30,
        help="Max balanced k-means iterations per level",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Training device; falls back to cpu if cuda is unavailable",
    )
    parser.add_argument(
        "--output_npz",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/codebook/yambda_rq_codebook.npz"),
        help="Output codebook npz path",
    )
    parser.add_argument(
        "--output_meta",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/codebook/yambda_rq_codebook.meta.json"),
        help="Output metadata json path",
    )
    return parser.parse_args()


def iter_embedding_batches(
    parquet_path: Path,
    column: str,
    batch_size: int,
) -> Iterable[np.ndarray]:
    """
    输入：
      - parquet_path: embeddings.parquet 路径
      - column: 读取哪一列，embed 或 normalized_embed
      - batch_size: 每次流式读取多少条
    输出：
      - 迭代返回 shape=(B, D) 的 numpy 向量批次
    """
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=[column]):
        values = batch.column(0).to_pylist()
        arr = np.asarray(values, dtype=np.float32)
        yield arr


def reservoir_sample_embeddings(
    parquet_path: Path,
    column: str,
    sample_size: int,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """
    输入：
      - parquet_path: embeddings.parquet
      - column: 向量列名
      - sample_size: 目标采样数
      - batch_size: 流式读取 batch 大小
      - seed: 随机种子
    输出：
      - sample: shape=(N, D) 的采样向量矩阵
      - info: 记录扫描行数、实际采样数、向量维度的字典
    """
    rng = np.random.default_rng(seed)
    sample = None
    total_seen = 0
    dim = None

    for batch_idx, arr in enumerate(iter_embedding_batches(parquet_path, column, batch_size), start=1):
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D embedding batch, got shape={arr.shape}")
        if dim is None:
            dim = arr.shape[1]
            take = max(1, sample_size)
            sample = np.empty((take, dim), dtype=np.float32)
        elif arr.shape[1] != dim:
            raise ValueError(f"Inconsistent embedding dim: expected {dim}, got {arr.shape[1]}")

        for row in arr:
            if total_seen < sample.shape[0]:
                sample[total_seen] = row
            else:
                j = rng.integers(0, total_seen + 1)
                if j < sample.shape[0]:
                    sample[j] = row
            total_seen += 1

        if batch_idx % 50 == 0:
            print(f"[sample] processed {total_seen:,} rows")

    if sample is None:
        raise RuntimeError(f"No embeddings found in {parquet_path}")

    if total_seen < sample.shape[0]:
        sample = sample[:total_seen]

    return sample, {
        "rows_seen": int(total_seen),
        "sample_size_effective": int(sample.shape[0]),
        "embedding_dim": int(sample.shape[1]),
    }


def compute_sample_stats(sample: np.ndarray) -> dict:
    """输入：采样向量矩阵。输出：均值、方差、范数等简单统计。"""
    return {
        "mean": float(sample.mean()),
        "std": float(sample.std()),
        "min": float(sample.min()),
        "max": float(sample.max()),
        "mean_l2_norm": float(np.linalg.norm(sample, axis=1).mean()),
    }


def main() -> None:
    """主入口：读取数据、采样、训练 codebook、写出产物。"""
    args = parse_args()
    parquet_path = Path(args.embeddings_parquet)
    output_npz = Path(args.output_npz)
    output_meta = Path(args.output_meta)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    output_meta.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] reservoir sampling from {parquet_path}")
    sample, sample_info = reservoir_sample_embeddings(
        parquet_path=parquet_path,
        column=args.embedding_column,
        sample_size=args.sample_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    sample_stats = compute_sample_stats(sample)
    print(
        "[sample] rows_seen={rows_seen:,}, sample_size={sample_size_effective:,}, dim={embedding_dim}".format(
            **sample_info
        )
    )
    print(f"[sample] stats={sample_stats}")

    print("[2/3] training residual balanced k-means codebook")
    generator = SemanticIDGeneratorTorch(
        n_levels=args.n_levels,
        codebook_size=args.codebook_size,
        device=args.device,
        max_iter=args.max_iter,
    )
    generator.fit(sample)

    codebooks = []
    for idx, cb in enumerate(generator.codebooks, start=1):
        if cb is None:
            break
        codebooks.append(cb.numpy())
        print(f"[codebook] level={idx}, shape={tuple(cb.shape)}")

    codebook_arr = np.stack(codebooks, axis=0)

    print(f"[3/3] saving codebooks to {output_npz}")
    np.savez(output_npz, codebooks=codebook_arr)

    meta = {
        "embeddings_parquet": str(parquet_path),
        "embedding_column": args.embedding_column,
        "sample_size_requested": args.sample_size,
        "sample_info": sample_info,
        "sample_stats": sample_stats,
        "n_levels_requested": args.n_levels,
        "n_levels_trained": len(codebooks),
        "codebook_size": args.codebook_size,
        "max_iter": args.max_iter,
        "device_used": str(generator.device),
        "output_npz": str(output_npz),
    }
    output_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] metadata saved to {output_meta}")


if __name__ == "__main__":
    main()
