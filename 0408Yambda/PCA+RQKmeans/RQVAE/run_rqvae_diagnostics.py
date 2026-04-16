#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from RQVAE.encoder import load_rqvae_encoder
from common.io import ensure_dir, iter_embedding_batches, parquet_metadata, write_json
from common.metrics import LocalityReservoir, ReconstructionAccumulator, build_metric_report, sampled_locality_metrics


DEFAULT_EMBEDDINGS = "/Users/Toryck/Coding/DATASET/Yambda/embeddings.parquet"
DEFAULT_OUT = ROOT / "artifacts" / "RQVAE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode and diagnose an RQVAE semantic-ID checkpoint")
    parser.add_argument("--embeddings", default=DEFAULT_EMBEDDINGS, help="Path to embeddings.parquet")
    parser.add_argument("--embedding_column", default="normalized_embed", choices=["embed", "normalized_embed"])
    parser.add_argument("--rqvae_ckpt", required=True, help="Path to RQVAE checkpoint")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--max_rows", type=int, default=0, help="0 means all rows")
    parser.add_argument("--batch_size", type=int, default=65536)
    parser.add_argument("--device", default=None, help="cuda, mps, cpu, or empty for auto")
    parser.add_argument("--locality_sample_size", type=int, default=200000)
    parser.add_argument("--locality_query_size", type=int, default=5000)
    parser.add_argument("--locality_topk", type=int, default=10)
    parser.add_argument("--write_dense_mapping", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def decode_numpy(encoder, codes: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(codes.astype(np.int64, copy=False)).to(encoder.device)
    recon = encoder.model.decode_from_codes(tensor)
    return recon.detach().cpu().numpy().astype(np.float32, copy=False)


def main() -> None:
    args = parse_args()
    started = time.time()
    out_dir = ensure_dir(args.out_dir)

    data_meta = parquet_metadata(args.embeddings, args.embedding_column)
    total_rows = int(data_meta["num_rows"])
    effective_rows = min(total_rows, args.max_rows) if args.max_rows else total_rows
    print(f"[data] rows={total_rows:,}, effective_rows={effective_rows:,}, dim={data_meta['embedding_dim']}")

    encoder = load_rqvae_encoder(args.rqvae_ckpt, device=args.device)
    n_levels = int(encoder.n_levels)
    codebook_size = int(encoder.codebook_size)
    use_recon = bool(getattr(encoder.model, "use_recon", False)) and hasattr(encoder.model, "decoder")
    np.savez(out_dir / "codebooks_latent_space.npz", codebooks=np.stack(encoder.codebooks, axis=0).astype(np.float32))

    item_ids = np.empty(effective_rows, dtype=np.int64)
    sid = np.empty((effective_rows, n_levels), dtype=np.int32)
    recon_acc = ReconstructionAccumulator() if use_recon else None
    locality_sampler = LocalityReservoir(
        sample_size=args.locality_sample_size,
        embedding_dim=int(data_meta["embedding_dim"]),
        sid_levels=n_levels,
        seed=42,
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
        codes = encoder.encode_numpy(x).astype(np.int32)
        if recon_acc is not None:
            recon = decode_numpy(encoder, codes)
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
        dense_sid = -np.ones((effective_rows + 1, n_levels), dtype=np.int32)
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
        seed=42,
    )

    reconstruction = recon_acc.compute(dim=int(data_meta["embedding_dim"])) if recon_acc is not None else {
        "skipped": True,
        "reason": "RQVAE checkpoint has use_recon=False or no decoder",
    }
    metrics = build_metric_report(
        sid=sid,
        codebook_size=codebook_size,
        reconstruction=reconstruction,
        locality=locality,
        extra={
            "variant": "rqvae",
            "rqvae_ckpt": str(args.rqvae_ckpt),
            "item_ids_path": str(item_path),
            "sid_path": str(sid_path),
            "use_recon": use_recon,
        },
    )
    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "meta.json", {
        "variant": "rqvae",
        "args": vars(args),
        "data": data_meta,
        "effective_rows": int(effective_rows),
        "runtime_sec": float(time.time() - started),
        "model": {
            "n_levels": n_levels,
            "codebook_size": codebook_size,
            "code_dim": int(encoder.code_dim),
            "input_dim": int(encoder.input_dim),
            "use_recon": use_recon,
        },
        "outputs": {
            "item_ids": str(item_path),
            "sid": str(sid_path),
            "metrics": str(out_dir / "metrics.json"),
        },
    })
    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()

