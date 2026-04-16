#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.io import ensure_dir, iter_embedding_batches, load_embedding_matrix, parquet_metadata, reservoir_sample_embeddings, write_json
from common.metrics import LocalityReservoir, ReconstructionAccumulator, build_metric_report, sampled_locality_metrics
from common.rqkmeans import ResidualKMeans


DEFAULT_EMBEDDINGS = "/Users/Toryck/Coding/DATASET/Yambda/embeddings.parquet"
DEFAULT_OUT = ROOT / "artifacts" / "rawRQKmeans"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Raw RQKMeans baseline for Yambda semantic IDs")
    parser.add_argument("--embeddings", default=DEFAULT_EMBEDDINGS, help="Path to embeddings.parquet")
    parser.add_argument("--embedding_column", default="normalized_embed", choices=["embed", "normalized_embed"])
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--max_rows", type=int, default=0, help="0 means all rows")
    parser.add_argument("--train_rows", type=int, default=0, help="0 means train on all effective rows; otherwise reservoir sample")
    parser.add_argument("--batch_size", type=int, default=65536)
    parser.add_argument("--n_levels", type=int, default=4)
    parser.add_argument("--codebook_size", type=int, default=256)
    parser.add_argument("--kmeans_iter", type=int, default=50)
    parser.add_argument("--nredo", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_points_per_centroid", type=int, default=100000)
    parser.add_argument("--locality_sample_size", type=int, default=200000)
    parser.add_argument("--locality_query_size", type=int, default=5000)
    parser.add_argument("--locality_topk", type=int, default=10)
    parser.add_argument("--write_dense_mapping", action="store_true", help="Also write HSRL-style dense arrays with padding row 0")
    parser.add_argument("--quiet_faiss", action="store_true")
    return parser.parse_args()


def load_training_data(args: argparse.Namespace) -> tuple[np.ndarray, dict]:
    if args.train_rows == 0:
        print("[train-data] loading all effective rows into memory")
        _, x_train = load_embedding_matrix(
            args.embeddings,
            args.embedding_column,
            batch_size=args.batch_size,
            max_rows=args.max_rows,
            include_item_ids=False,
        )
        info = {
            "mode": "full_effective_rows",
            "rows": int(x_train.shape[0]),
            "dim": int(x_train.shape[1]),
        }
        return x_train, info

    print(f"[train-data] reservoir sampling {args.train_rows:,} rows")
    _, x_train, info = reservoir_sample_embeddings(
        args.embeddings,
        sample_size=args.train_rows,
        embedding_column=args.embedding_column,
        batch_size=args.batch_size,
        max_rows=args.max_rows,
        seed=args.seed,
        include_item_ids=False,
    )
    return x_train, info


def main() -> None:
    args = parse_args()
    started = time.time()
    out_dir = ensure_dir(args.out_dir)

    data_meta = parquet_metadata(args.embeddings, args.embedding_column)
    total_rows = int(data_meta["num_rows"])
    effective_rows = min(total_rows, args.max_rows) if args.max_rows else total_rows
    print(f"[data] rows={total_rows:,}, effective_rows={effective_rows:,}, dim={data_meta['embedding_dim']}")

    x_train, train_info = load_training_data(args)
    model = ResidualKMeans(
        n_levels=args.n_levels,
        codebook_size=args.codebook_size,
        niter=args.kmeans_iter,
        nredo=args.nredo,
        seed=args.seed,
        verbose=not args.quiet_faiss,
        max_points_per_centroid=args.max_points_per_centroid,
    )
    model.fit(x_train)
    codebook_path = out_dir / "codebook.npz"
    model.save(codebook_path)
    del x_train

    item_ids = np.empty(effective_rows, dtype=np.int64)
    sid = np.empty((effective_rows, args.n_levels), dtype=np.int32)
    recon_acc = ReconstructionAccumulator()
    locality_sampler = LocalityReservoir(
        sample_size=args.locality_sample_size,
        embedding_dim=int(data_meta["embedding_dim"]),
        sid_levels=args.n_levels,
        seed=args.seed,
    )

    cursor = 0
    for batch_idx, (batch_ids, x) in enumerate(
        iter_embedding_batches(
            args.embeddings,
            args.embedding_column,
            batch_size=args.batch_size,
            max_rows=args.max_rows,
            include_item_ids=True,
        ),
        start=1,
    ):
        codes = model.encode(x)
        recon = model.reconstruct(codes)
        recon_acc.update(x, recon)
        locality_sampler.update(x, codes)

        end = cursor + x.shape[0]
        item_ids[cursor:end] = batch_ids
        sid[cursor:end] = codes
        cursor = end
        if batch_idx % 10 == 0:
            print(f"[encode] {cursor:,}/{effective_rows:,}")

    if cursor != effective_rows:
        raise RuntimeError(f"Encoded {cursor} rows, expected {effective_rows}")

    item_path = out_dir / "item_ids.npy"
    sid_path = out_dir / "sid.npy"
    np.save(item_path, item_ids)
    np.save(sid_path, sid)

    if args.write_dense_mapping:
        dense_sid = -np.ones((effective_rows + 1, args.n_levels), dtype=np.int32)
        dense2orig = -np.ones(effective_rows + 1, dtype=np.int64)
        dense_sid[1:] = sid
        dense2orig[1:] = item_ids
        np.save(out_dir / "dense_item2sid.npy", dense_sid)
        np.save(out_dir / "dense2orig_item_id.npy", dense2orig)

    loc_x, loc_sid = locality_sampler.arrays()
    np.save(out_dir / "locality_sample_embeddings.npy", loc_x)
    np.save(out_dir / "locality_sample_sid.npy", loc_sid)
    locality = sampled_locality_metrics(
        loc_x,
        loc_sid,
        query_size=args.locality_query_size,
        pool_size=args.locality_sample_size,
        topk=args.locality_topk,
        seed=args.seed,
    )

    metrics = build_metric_report(
        sid=sid,
        codebook_size=args.codebook_size,
        reconstruction=recon_acc.compute(dim=int(data_meta["embedding_dim"])),
        locality=locality,
        extra={
            "variant": "raw_rqkmeans",
            "codebook_path": str(codebook_path),
            "item_ids_path": str(item_path),
            "sid_path": str(sid_path),
        },
    )
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "meta.json", {
        "variant": "raw_rqkmeans",
        "args": vars(args),
        "data": data_meta,
        "effective_rows": int(effective_rows),
        "train_info": train_info,
        "runtime_sec": float(time.time() - started),
        "outputs": {
            "codebook": str(codebook_path),
            "item_ids": str(item_path),
            "sid": str(sid_path),
            "metrics": str(out_dir / "metrics.json"),
        },
    })
    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()

