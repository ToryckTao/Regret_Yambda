from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parquet_metadata(parquet_path: str | Path, embedding_column: str = "normalized_embed") -> dict:
    parquet_path = Path(parquet_path)
    pf = pq.ParquetFile(parquet_path)
    first = next(pf.iter_batches(batch_size=1, columns=["item_id", embedding_column]))
    dim = len(first.column(embedding_column)[0].as_py())
    return {
        "path": str(parquet_path),
        "num_rows": int(pf.metadata.num_rows),
        "num_row_groups": int(pf.metadata.num_row_groups),
        "embedding_column": embedding_column,
        "embedding_dim": int(dim),
        "schema": str(pf.schema),
    }


def _list_array_to_numpy(column) -> np.ndarray:
    """Convert a pyarrow List/LargeList vector column into a 2D float32 matrix."""
    offsets = np.asarray(column.offsets.to_numpy(zero_copy_only=False), dtype=np.int64)
    values = np.asarray(column.values.to_numpy(zero_copy_only=False), dtype=np.float32)
    lengths = np.diff(offsets)
    if lengths.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    if not np.all(lengths == lengths[0]):
        return np.asarray(column.to_pylist(), dtype=np.float32)
    start = int(offsets[0])
    stop = int(offsets[-1])
    dim = int(lengths[0])
    return values[start:stop].reshape(lengths.size, dim).astype(np.float32, copy=False)


def iter_embedding_batches(
    parquet_path: str | Path,
    embedding_column: str = "normalized_embed",
    batch_size: int = 65536,
    max_rows: int = 0,
    include_item_ids: bool = True,
) -> Iterator[tuple[np.ndarray | None, np.ndarray]]:
    parquet_path = Path(parquet_path)
    columns = [embedding_column]
    if include_item_ids:
        columns = ["item_id", embedding_column]

    rows_yielded = 0
    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(batch_size=batch_size, columns=columns):
        if include_item_ids:
            item_ids = np.asarray(batch.column("item_id").to_numpy(zero_copy_only=False), dtype=np.int64)
            emb_col = batch.column(embedding_column)
        else:
            item_ids = None
            emb_col = batch.column(embedding_column)
        emb = _list_array_to_numpy(emb_col)

        if max_rows and rows_yielded + emb.shape[0] > max_rows:
            keep = max_rows - rows_yielded
            if keep <= 0:
                break
            emb = emb[:keep]
            if item_ids is not None:
                item_ids = item_ids[:keep]

        rows_yielded += emb.shape[0]
        yield item_ids, emb
        if max_rows and rows_yielded >= max_rows:
            break


def load_embedding_matrix(
    parquet_path: str | Path,
    embedding_column: str = "normalized_embed",
    batch_size: int = 65536,
    max_rows: int = 0,
    include_item_ids: bool = True,
) -> tuple[np.ndarray | None, np.ndarray]:
    item_chunks: list[np.ndarray] = []
    emb_chunks: list[np.ndarray] = []
    for item_ids, emb in iter_embedding_batches(
        parquet_path=parquet_path,
        embedding_column=embedding_column,
        batch_size=batch_size,
        max_rows=max_rows,
        include_item_ids=include_item_ids,
    ):
        if include_item_ids and item_ids is not None:
            item_chunks.append(item_ids)
        emb_chunks.append(emb)
    if not emb_chunks:
        raise RuntimeError(f"No embeddings loaded from {parquet_path}")
    items = np.concatenate(item_chunks, axis=0) if include_item_ids else None
    embeddings = np.concatenate(emb_chunks, axis=0).astype(np.float32, copy=False)
    return items, embeddings


def reservoir_sample_embeddings(
    parquet_path: str | Path,
    sample_size: int,
    embedding_column: str = "normalized_embed",
    batch_size: int = 65536,
    max_rows: int = 0,
    seed: int = 42,
    include_item_ids: bool = True,
) -> tuple[np.ndarray | None, np.ndarray, dict]:
    if sample_size <= 0:
        return (*load_embedding_matrix(parquet_path, embedding_column, batch_size, max_rows, include_item_ids), {
            "mode": "full_load",
            "sample_size_requested": int(sample_size),
        })

    rng = np.random.default_rng(seed)
    sample_x: np.ndarray | None = None
    sample_ids: np.ndarray | None = None
    seen = 0

    for item_ids, emb in iter_embedding_batches(
        parquet_path=parquet_path,
        embedding_column=embedding_column,
        batch_size=batch_size,
        max_rows=max_rows,
        include_item_ids=include_item_ids,
    ):
        if sample_x is None:
            take = min(sample_size, emb.shape[0])
            sample_x = np.empty((sample_size, emb.shape[1]), dtype=np.float32)
            if include_item_ids:
                sample_ids = np.empty(sample_size, dtype=np.int64)
            sample_x[:take] = emb[:take]
            if include_item_ids and sample_ids is not None and item_ids is not None:
                sample_ids[:take] = item_ids[:take]
            seen += take
            emb = emb[take:]
            if item_ids is not None:
                item_ids = item_ids[take:]

        for row_idx in range(emb.shape[0]):
            if seen < sample_size:
                sample_x[seen] = emb[row_idx]
                if include_item_ids and sample_ids is not None and item_ids is not None:
                    sample_ids[seen] = item_ids[row_idx]
            else:
                j = int(rng.integers(0, seen + 1))
                if j < sample_size:
                    sample_x[j] = emb[row_idx]
                    if include_item_ids and sample_ids is not None and item_ids is not None:
                        sample_ids[j] = item_ids[row_idx]
            seen += 1

    if sample_x is None:
        raise RuntimeError(f"No embeddings sampled from {parquet_path}")

    effective = min(seen, sample_size)
    sample_x = sample_x[:effective]
    if include_item_ids and sample_ids is not None:
        sample_ids = sample_ids[:effective]

    info = {
        "mode": "reservoir",
        "rows_seen": int(seen),
        "sample_size_requested": int(sample_size),
        "sample_size_effective": int(effective),
        "embedding_dim": int(sample_x.shape[1]),
        "seed": int(seed),
    }
    return sample_ids, sample_x, info

