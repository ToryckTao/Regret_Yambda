from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np


def nearest_code(x: np.ndarray, centers: np.ndarray, metric: str = "l2") -> np.ndarray:
    x = np.ascontiguousarray(x.astype(np.float32, copy=False))
    centers = np.ascontiguousarray(centers.astype(np.float32, copy=False))
    if metric != "l2":
        raise ValueError(f"Unsupported metric: {metric}")
    index = faiss.IndexFlatL2(centers.shape[1])
    index.add(centers)
    _, labels = index.search(x, 1)
    return labels[:, 0].astype(np.int32)


class ResidualKMeans:
    def __init__(
        self,
        n_levels: int = 4,
        codebook_size: int = 256,
        niter: int = 50,
        nredo: int = 1,
        seed: int = 42,
        verbose: bool = True,
        max_points_per_centroid: int = 256,
    ) -> None:
        self.n_levels = int(n_levels)
        self.codebook_size = int(codebook_size)
        self.niter = int(niter)
        self.nredo = int(nredo)
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self.max_points_per_centroid = int(max_points_per_centroid)
        self.codebooks: list[np.ndarray] = []

    def fit(self, x: np.ndarray) -> "ResidualKMeans":
        residual = np.ascontiguousarray(x.astype(np.float32, copy=True))
        self.codebooks = []
        for level in range(self.n_levels):
            if self.verbose:
                print(f"[rqkmeans] fitting level {level + 1}/{self.n_levels} on {residual.shape}")
            km = faiss.Kmeans(
                d=residual.shape[1],
                k=self.codebook_size,
                niter=self.niter,
                nredo=self.nredo,
                seed=self.seed + level,
                verbose=self.verbose,
                max_points_per_centroid=self.max_points_per_centroid,
            )
            km.train(np.ascontiguousarray(residual))
            centers = np.ascontiguousarray(km.centroids.astype(np.float32))
            labels = nearest_code(residual, centers)
            self.codebooks.append(centers)
            residual -= centers[labels]
            if self.verbose:
                print(f"[rqkmeans] residual mean l2={float(np.linalg.norm(residual, axis=1).mean()):.6f}")
        return self

    def encode(self, x: np.ndarray) -> np.ndarray:
        if not self.codebooks:
            raise RuntimeError("ResidualKMeans has no codebooks; call fit() or load() first")
        residual = np.ascontiguousarray(x.astype(np.float32, copy=True))
        codes = np.empty((residual.shape[0], len(self.codebooks)), dtype=np.int32)
        for level, centers in enumerate(self.codebooks):
            labels = nearest_code(residual, centers)
            codes[:, level] = labels
            residual -= centers[labels]
        return codes

    def reconstruct(self, codes: np.ndarray) -> np.ndarray:
        if not self.codebooks:
            raise RuntimeError("ResidualKMeans has no codebooks; call fit() or load() first")
        codes = codes.astype(np.int64, copy=False)
        out = np.zeros((codes.shape[0], self.codebooks[0].shape[1]), dtype=np.float32)
        for level, centers in enumerate(self.codebooks):
            out += centers[codes[:, level]]
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, codebooks=np.stack(self.codebooks, axis=0).astype(np.float32))

    @classmethod
    def load(cls, path: str | Path) -> "ResidualKMeans":
        z = np.load(path)
        obj = cls(n_levels=int(z["codebooks"].shape[0]), codebook_size=int(z["codebooks"].shape[1]))
        obj.codebooks = [z["codebooks"][i].astype(np.float32) for i in range(z["codebooks"].shape[0])]
        return obj

