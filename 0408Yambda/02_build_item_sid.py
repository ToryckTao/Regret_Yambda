#!/usr/bin/env python3
"""
把 Yambda 的 item 向量编码成 SID，并同时完成 item_id 重映射。

这个脚本参考原 HSRL 的 build_item2sid 思路。

Yambda 版的核心差异：
  1. 输入改为 embeddings.parquet
  2. 原始 item_id 是稀疏的，所以必须先做：
       orig_item_id -> dense_item_id
  3. 最终 HSRL 训练更适合吃 dense id 和对齐数组

默认输出：
  - dense_item2sid.npy
  - orig2dense_item_id.npy
  - dense2orig_item_id.npy

可选输出：
  - dense_item2sid.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from tqdm import tqdm

from RQVAE.encoder import load_rqvae_encoder


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_YAMBA_DATA_DIR = Path(os.environ.get("YAMBA_DATA_DIR", "/Users/Toryck/Coding/DATASET/Yambda"))


def parse_args() -> argparse.Namespace:
    """输入：命令行参数。输出：参数对象 Namespace。"""
    parser = argparse.ArgumentParser(description="Build dense item remapping and SID assignments for Yambda")
    parser.add_argument(
        "--embeddings_parquet",
        type=str,
        default=str(DEFAULT_YAMBA_DATA_DIR / "embeddings.parquet"),
        help="Path to Yambda embeddings.parquet",
    )
    parser.add_argument(
        "--codebook_npz",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/codebook/yambda_rq_codebook.npz"),
        help="Codebook file produced by 01_build_codebook.py (RQKMeans npz or RQVAE .pth)",
    )
    parser.add_argument(
        "--rqvae_ckpt",
        type=str,
        default=None,
        help="Path to RQVAE checkpoint (.pth). If provided, uses RQVAE encoder + codebooks instead of raw codebooks.",
    )
    parser.add_argument(
        "--embedding_column",
        type=str,
        default="normalized_embed",
        choices=["embed", "normalized_embed"],
        help="Embedding column used for SID encoding",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        help="Parquet streaming batch size",
    )
    parser.add_argument(
        "--max_rows",
        type=int,
        default=0,
        help="Optional row cap. 0 means use all rows.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PROJECT_ROOT / "artifacts/mappings"),
        help="Directory to write mapping artifacts",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="yambda",
        help="Output file prefix",
    )
    parser.add_argument(
        "--write_pickle_dict",
        action="store_true",
        help="Also write dense_item2sid.pkl. Disabled by default for Yambda scale.",
    )
    return parser.parse_args()


def load_codebooks(npz_path: Path) -> list[np.ndarray]:
    """
    输入：codebook 的 npz 文件路径。
    输出：list[np.ndarray]，每一层一个 codebook，shape 近似为 (V_l, D)。
    """
    z = np.load(npz_path, allow_pickle=True)
    cbs = z["codebooks"]
    if isinstance(cbs, np.ndarray) and cbs.dtype == object:
        return [np.asarray(cb, dtype=np.float32) for cb in cbs.tolist()]
    if cbs.ndim == 3:
        return [cbs[i].astype(np.float32) for i in range(cbs.shape[0])]
    return [np.asarray(cb, dtype=np.float32) for cb in cbs]


def encode_sid_batch(emb_batch: np.ndarray, codebooks: list[np.ndarray]) -> np.ndarray:
    """
    输入：
      - emb_batch: 一批 item 向量，shape=(B, D)
      - codebooks: 多层 codebook
    输出：
      - sid_batch: shape=(B, L)，每行是一个 item 的 SID 路径

    说明：
      - 这个函数与 encode_sid 的语义完全一致
      - 只是把逐 item 的最近邻量化改成 batch 向量化，便于全量跑 Yambda
    """
    residual = emb_batch.astype(np.float32, copy=True)
    sid = np.empty((emb_batch.shape[0], len(codebooks)), dtype=np.int32)

    for level, cb in enumerate(codebooks):
        xc = residual @ cb.T                           # (B, V)
        x2 = (residual * residual).sum(axis=1, keepdims=True)  # (B, 1)
        c2 = (cb * cb).sum(axis=1, keepdims=True).T   # (1, V)
        d2 = x2 + c2 - 2.0 * xc                       # (B, V)
        z = np.argmin(d2, axis=1).astype(np.int32)    # (B,)
        sid[:, level] = z
        residual = residual - cb[z]

    return sid


def iter_item_batches(
    parquet_path: Path,
    embedding_column: str,
    batch_size: int,
) -> Iterable[tuple[list[int], list[list[float]]]]:
    """
    输入：
      - parquet_path: embeddings.parquet 路径
      - embedding_column: embed 或 normalized_embed
      - batch_size: 流式读取 batch 大小
    输出：
      - (item_ids, embeddings) 批次
        item_ids: list[int]
        embeddings: list[list[float]]
    """
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=["item_id", embedding_column]):
        pyd = batch.to_pydict()
        yield pyd["item_id"], pyd[embedding_column]


def scan_item_id_range(parquet_path: Path) -> tuple[int, int, int]:
    """
    输入：embeddings.parquet 路径。
    输出：
      - num_rows: 总 item 数
      - min_item_id: 最小原始 item_id
      - max_item_id: 最大原始 item_id
    """
    pf = pq.ParquetFile(parquet_path)
    num_rows = pf.metadata.num_rows
    min_item_id = None
    max_item_id = None

    for batch in pf.iter_batches(batch_size=100000, columns=["item_id"]):
        pyd = batch.to_pydict()
        ids = pyd["item_id"]
        if min_item_id is None:
            min_item_id = ids[0]
            max_item_id = ids[0]
        batch_min = min(ids)
        batch_max = max(ids)
        if batch_min < min_item_id:
            min_item_id = batch_min
        if batch_max > max_item_id:
            max_item_id = batch_max

    if min_item_id is None or max_item_id is None:
        raise ValueError(f"Failed to scan item_id range from {parquet_path}")

    return num_rows, int(min_item_id), int(max_item_id)


def main() -> None:
    """主入口：加载 codebook、扫描 item_id、编码 SID、保存映射产物。"""
    args = parse_args()
    parquet_path = Path(args.embeddings_parquet)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_rqvae = args.rqvae_ckpt is not None
    if use_rqvae:
        print(f"[1/4] loading RQVAE checkpoint from {args.rqvae_ckpt}")
        encoder = load_rqvae_encoder(args.rqvae_ckpt)
        codebooks = encoder.codebooks
        n_levels = encoder.n_levels
        vocab_sizes = [encoder.codebook_size] * n_levels
        emb_dim = int(encoder.enc_w0.shape[1])  # 128 (input dim)
        print(f"[codebook] RQVAE mode: levels={n_levels}, vocab_sizes={vocab_sizes}, input_dim={emb_dim}")
    else:
        print(f"[1/4] loading codebooks from {args.codebook_npz}")
        codebooks = load_codebooks(Path(args.codebook_npz))
        n_levels = len(codebooks)
        vocab_sizes = [int(cb.shape[0]) for cb in codebooks]
        emb_dim = int(codebooks[0].shape[1])
        print(f"[codebook] levels={n_levels}, vocab_sizes={vocab_sizes}, dim={emb_dim}")
        encoder = None

    print(f"[2/4] scanning item_id range in {parquet_path}")
    num_rows, min_item_id, max_item_id = scan_item_id_range(parquet_path)
    if args.max_rows and args.max_rows > 0:
        num_rows_effective = min(num_rows, args.max_rows)
    else:
        num_rows_effective = num_rows
    print(
        f"[items] rows={num_rows:,}, effective_rows={num_rows_effective:,}, "
        f"min_item_id={min_item_id}, max_item_id={max_item_id}"
    )

    print("[3/4] encoding dense item ids and SID paths")
    dense_item2sid = -np.ones((num_rows_effective + 1, n_levels), dtype=np.int32)
    dense2orig = -np.ones(num_rows_effective + 1, dtype=np.int64)
    orig2dense = -np.ones(max_item_id + 1, dtype=np.int32)

    dense_id = 0
    prev_item_id = None
    monotonic_non_decreasing = True

    pbar = tqdm(total=num_rows_effective, unit="items", desc="[encode]", ncols=80)
    for batch_idx, (item_ids, embeddings) in enumerate(
        iter_item_batches(parquet_path, args.embedding_column, args.batch_size),
        start=1,
    ):
        batch_orig_ids: list[int] = []
        batch_embs: list[np.ndarray] = []

        for orig_item_id, emb in zip(item_ids, embeddings):
            if dense_id + len(batch_orig_ids) >= num_rows_effective:
                break

            orig_item_id = int(orig_item_id)
            if prev_item_id is not None and orig_item_id < prev_item_id:
                monotonic_non_decreasing = False
            prev_item_id = orig_item_id

            if orig2dense[orig_item_id] != -1:
                raise ValueError(f"Duplicate item_id detected: {orig_item_id}")

            emb_arr = np.asarray(emb, dtype=np.float32)
            if emb_arr.shape[0] != emb_dim:
                raise ValueError(
                    f"Inconsistent embedding dim for item_id={orig_item_id}: "
                    f"expected {emb_dim}, got {emb_arr.shape[0]}"
                )
            batch_orig_ids.append(orig_item_id)
            batch_embs.append(emb_arr)

        if not batch_embs:
            break

        emb_batch = np.stack(batch_embs, axis=0)

        if use_rqvae:
            sid_batch = encoder.encode_numpy(emb_batch).astype(np.int32)
        else:
            sid_batch = encode_sid_batch(emb_batch, codebooks)

        start_dense = dense_id + 1
        end_dense = dense_id + len(batch_orig_ids)
        dense_ids = np.arange(start_dense, end_dense + 1, dtype=np.int32)

        dense2orig[start_dense:end_dense + 1] = np.asarray(batch_orig_ids, dtype=np.int64)
        orig2dense[np.asarray(batch_orig_ids, dtype=np.int64)] = dense_ids
        dense_item2sid[start_dense:end_dense + 1] = sid_batch
        dense_id = end_dense
        pbar.update(len(batch_orig_ids))
        if args.max_rows and dense_id >= num_rows_effective:
            break
    pbar.close()

    if dense_id != num_rows_effective:
        raise RuntimeError(f"Expected {num_rows_effective} items, but encoded {dense_id}")

    prefix = args.output_prefix
    sid_npy_path = output_dir / f"{prefix}_dense_item2sid.npy"
    dense2orig_path = output_dir / f"{prefix}_dense2orig_item_id.npy"
    orig2dense_path = output_dir / f"{prefix}_orig2dense_item_id.npy"
    meta_path = output_dir / f"{prefix}_item_id_mapping.meta.json"

    print("[4/4] saving mapping artifacts")
    np.save(sid_npy_path, dense_item2sid)
    np.save(dense2orig_path, dense2orig)
    np.save(orig2dense_path, orig2dense)

    meta = {
        "embeddings_parquet": str(parquet_path),
        "embedding_column": args.embedding_column,
        "codebook_npz": str(args.codebook_npz),
        "rqvae_ckpt": str(args.rqvae_ckpt) if use_rqvae else None,
        "encoding_mode": "rqvae" if use_rqvae else "rqkmeans",
        "num_rows_total": int(num_rows),
        "num_rows_effective": int(num_rows_effective),
        "min_item_id": int(min_item_id),
        "max_item_id": int(max_item_id),
        "monotonic_non_decreasing_item_ids": bool(monotonic_non_decreasing),
        "n_levels": int(n_levels),
        "vocab_sizes": vocab_sizes,
        "embedding_dim": int(emb_dim),
        "outputs": {
            "dense_item2sid_npy": str(sid_npy_path),
            "dense2orig_item_id_npy": str(dense2orig_path),
            "orig2dense_item_id_npy": str(orig2dense_path),
        },
    }

    if args.write_pickle_dict:
        pkl_path = output_dir / f"{prefix}_dense_item2sid.pkl"
        print("[pickle] building dense_item2sid dict; this may use substantial memory")
        sid_dict = {
            int(i): tuple(int(x) for x in dense_item2sid[i])
            for i in range(1, dense_item2sid.shape[0])
        }
        with open(pkl_path, "wb") as f:
            pickle.dump(sid_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        meta["outputs"]["dense_item2sid_pkl"] = str(pkl_path)

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] saved metadata to {meta_path}")
    print("[example]")
    print(
        f"  dense_id=1 -> orig_item_id={int(dense2orig[1])} -> SID={dense_item2sid[1].tolist()}"
    )


if __name__ == "__main__":
    main()
