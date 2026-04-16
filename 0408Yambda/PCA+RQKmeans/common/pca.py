from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import iter_embedding_batches, write_json


def fit_streaming_pca(
    parquet_path: str | Path,
    n_components: int,
    embedding_column: str = "normalized_embed",
    batch_size: int = 65536,
    max_rows: int = 0,
) -> dict:
    """Fit exact PCA for low-dimensional embeddings by streaming mean and covariance."""
    total = 0
    sum_x: np.ndarray | None = None

    for _, x in iter_embedding_batches(parquet_path, embedding_column, batch_size, max_rows, include_item_ids=False):
        xb = x.astype(np.float64, copy=False)
        if sum_x is None:
            sum_x = np.zeros(xb.shape[1], dtype=np.float64)
        sum_x += xb.sum(axis=0)
        total += xb.shape[0]

    if sum_x is None or total < 2:
        raise RuntimeError("PCA needs at least two rows")

    mean = sum_x / total
    xtx = np.zeros((mean.shape[0], mean.shape[0]), dtype=np.float64)

    for _, x in iter_embedding_batches(parquet_path, embedding_column, batch_size, max_rows, include_item_ids=False):
        xc = x.astype(np.float64, copy=False) - mean
        xtx += xc.T @ xc

    cov = xtx / max(total - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    n_components = min(int(n_components), eigvecs.shape[1])

    explained = eigvals[:n_components]
    components = eigvecs[:, :n_components].T.astype(np.float32)
    total_var = float(np.maximum(eigvals, 0.0).sum())
    ratio = explained / total_var if total_var > 0 else np.zeros_like(explained)
    return {
        "mean": mean.astype(np.float32),
        "components": components,
        "explained_variance": explained.astype(np.float32),
        "explained_variance_ratio": ratio.astype(np.float32),
        "n_samples": int(total),
        "input_dim": int(mean.shape[0]),
        "n_components": int(n_components),
    }


def transform(x: np.ndarray, pca: dict, whiten: bool = False, eps: float = 1e-12) -> np.ndarray:
    z = (x.astype(np.float32, copy=False) - pca["mean"]) @ pca["components"].T
    if whiten:
        z = z / np.sqrt(pca["explained_variance"] + eps)
    return z.astype(np.float32, copy=False)


def inverse_transform(z: np.ndarray, pca: dict, whiten: bool = False, eps: float = 1e-12) -> np.ndarray:
    y = z.astype(np.float32, copy=False)
    if whiten:
        y = y * np.sqrt(pca["explained_variance"] + eps)
    return (y @ pca["components"] + pca["mean"]).astype(np.float32, copy=False)


def save_pca(path: str | Path, pca: dict, extra_meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        mean=pca["mean"],
        components=pca["components"],
        explained_variance=pca["explained_variance"],
        explained_variance_ratio=pca["explained_variance_ratio"],
    )
    meta = {
        "n_samples": int(pca["n_samples"]),
        "input_dim": int(pca["input_dim"]),
        "n_components": int(pca["n_components"]),
        "explained_variance_ratio_sum": float(np.sum(pca["explained_variance_ratio"])),
    }
    if extra_meta:
        meta.update(extra_meta)
    write_json(path.with_suffix(".meta.json"), meta)


def load_pca(path: str | Path) -> dict:
    z = np.load(path)
    return {
        "mean": z["mean"].astype(np.float32),
        "components": z["components"].astype(np.float32),
        "explained_variance": z["explained_variance"].astype(np.float32),
        "explained_variance_ratio": z["explained_variance_ratio"].astype(np.float32),
        "n_samples": -1,
        "input_dim": int(z["mean"].shape[0]),
        "n_components": int(z["components"].shape[0]),
    }

