#!/usr/bin/env python3
"""准备 item 映射：把 Repersentation 里的语义 ID 产物整理成 Regret 训练可用的 dense item / SID 映射。

一般只在重新生成 RQKmeans/RQVAE 语义 ID 时运行；日常主线训练不需要反复运行。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent


VARIANT_PATHS = {
    "raw_rqkmeans": ("artifacts_full_train", "rawRQKmeans"),
    "rqvae": ("artifacts_full_train", "RQVAE"),
    "pca32_nowhiten": ("artifacts_full_train", "PCA_RQKmeans", "pca32_nowhiten"),
    "pca64_nowhiten": ("artifacts_full_train", "PCA_RQKmeans", "pca64_nowhiten"),
    "pca64_whiten": ("artifacts_full_train", "PCA_RQKmeans", "pca64_whiten"),
    "pca96_nowhiten": ("artifacts_full_train", "PCA_RQKmeans", "pca96_nowhiten"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Regret dense mappings from Repersentation SID artifacts")
    parser.add_argument("--representation_dir", default=str(REPO_ROOT / "Repersentation"))
    parser.add_argument("--variant", default="raw_rqkmeans", choices=sorted(VARIANT_PATHS))
    parser.add_argument("--out_dir", default=str(PROJECT_ROOT / "artifacts/mappings/raw_rqkmeans"))
    parser.add_argument("--embeddings_parquet", default=str(REPO_ROOT / "../0330Yambda/data/embeddings.parquet"))
    parser.add_argument("--embedding_column", default="normalized_embed", choices=["embed", "normalized_embed"])
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--write_item_features", action="store_true")
    parser.add_argument("--feature_batch_size", type=int, default=8192)
    return parser.parse_args()


def resolve_variant_dir(representation_dir: Path, variant: str) -> Path:
    return representation_dir.joinpath(*VARIANT_PATHS[variant])


def write_feature_matrix(
    embeddings_parquet: Path,
    embedding_column: str,
    embedding_dim: int,
    orig2dense: np.ndarray,
    n_dense: int,
    out_path: Path,
    batch_size: int,
) -> dict:
    features = np.lib.format.open_memmap(
        out_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_dense + 1, embedding_dim),
    )
    features[0, :] = 0.0
    pf = pq.ParquetFile(embeddings_parquet)
    filled = 0
    skipped = 0
    for batch_idx, batch in enumerate(pf.iter_batches(batch_size=batch_size, columns=["item_id", embedding_column]), start=1):
        pyd = batch.to_pydict()
        orig_ids = np.asarray(pyd["item_id"], dtype=np.int64)
        valid = orig_ids < len(orig2dense)
        dense_ids = np.zeros_like(orig_ids, dtype=np.int64)
        dense_ids[valid] = orig2dense[orig_ids[valid]]
        valid &= dense_ids > 0
        if valid.any():
            vecs = np.asarray(pyd[embedding_column], dtype=np.float32)
            features[dense_ids[valid]] = vecs[valid]
            filled += int(valid.sum())
        skipped += int((~valid).sum())
        if batch_idx % 50 == 0:
            print(f"[features] batch={batch_idx} filled={filled:,} skipped={skipped:,}")
    features.flush()
    return {"filled": int(filled), "skipped": int(skipped), "path": str(out_path)}


def main() -> None:
    args = parse_args()
    representation_dir = Path(args.representation_dir)
    variant_dir = resolve_variant_dir(representation_dir, args.variant)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    item_ids_path = variant_dir / "item_ids.npy"
    sid_path = variant_dir / "sid.npy"
    if not item_ids_path.exists() or not sid_path.exists():
        raise FileNotFoundError(f"Missing item_ids.npy or sid.npy under {variant_dir}")

    item_ids = np.load(item_ids_path, mmap_mode="r")
    sid = np.load(sid_path, mmap_mode="r")
    if item_ids.shape[0] != sid.shape[0]:
        raise ValueError(f"item_ids and sid row mismatch: {item_ids.shape} vs {sid.shape}")

    n_item = int(item_ids.shape[0])
    sid_levels = int(sid.shape[1])
    max_orig = int(item_ids.max())
    dense2orig = np.empty(n_item + 1, dtype=np.int64)
    dense2orig[0] = -1
    dense2orig[1:] = np.asarray(item_ids, dtype=np.int64)

    orig2dense = np.zeros(max_orig + 1, dtype=np.int32)
    orig2dense[np.asarray(item_ids, dtype=np.int64)] = np.arange(1, n_item + 1, dtype=np.int32)

    dense_item2sid = np.zeros((n_item + 1, sid_levels), dtype=np.int16)
    dense_item2sid[1:, :] = np.asarray(sid, dtype=np.int16)

    dense2orig_path = out_dir / "dense2orig_item_id.npy"
    orig2dense_path = out_dir / "orig2dense_item_id.npy"
    dense_item2sid_path = out_dir / "dense_item2sid.npy"
    np.save(dense2orig_path, dense2orig)
    np.save(orig2dense_path, orig2dense)
    np.save(dense_item2sid_path, dense_item2sid)

    feature_stats = None
    feature_path = out_dir / "dense_item_features.npy"
    if args.write_item_features:
        feature_stats = write_feature_matrix(
            embeddings_parquet=Path(args.embeddings_parquet),
            embedding_column=args.embedding_column,
            embedding_dim=args.embedding_dim,
            orig2dense=orig2dense,
            n_dense=n_item,
            out_path=feature_path,
            batch_size=args.feature_batch_size,
        )

    meta = {
        "variant": args.variant,
        "variant_dir": str(variant_dir),
        "n_item": n_item,
        "max_orig_item_id": max_orig,
        "sid_levels": sid_levels,
        "sid_vocab_size": int(dense_item2sid[1:].max()) + 1,
        "outputs": {
            "dense2orig_item_id": str(dense2orig_path),
            "orig2dense_item_id": str(orig2dense_path),
            "dense_item2sid": str(dense_item2sid_path),
            "dense_item_features": str(feature_path) if args.write_item_features else "",
        },
        "feature_stats": feature_stats,
    }
    (out_dir / "mapping.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[done] mapping saved to {out_dir}")


if __name__ == "__main__":
    main()
