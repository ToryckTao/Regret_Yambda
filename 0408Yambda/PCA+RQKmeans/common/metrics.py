from __future__ import annotations

import math

import faiss
import numpy as np


class LocalityReservoir:
    def __init__(self, sample_size: int, embedding_dim: int, sid_levels: int, seed: int = 42) -> None:
        self.sample_size = int(sample_size)
        self.rng = np.random.default_rng(seed)
        self.seen = 0
        self.embeddings = np.empty((self.sample_size, embedding_dim), dtype=np.float32)
        self.sid = np.empty((self.sample_size, sid_levels), dtype=np.int32)

    def update(self, embeddings: np.ndarray, sid: np.ndarray) -> None:
        if self.sample_size <= 0:
            self.seen += int(embeddings.shape[0])
            return
        embeddings = embeddings.astype(np.float32, copy=False)
        sid = sid.astype(np.int32, copy=False)
        for i in range(embeddings.shape[0]):
            if self.seen < self.sample_size:
                self.embeddings[self.seen] = embeddings[i]
                self.sid[self.seen] = sid[i]
            else:
                j = int(self.rng.integers(0, self.seen + 1))
                if j < self.sample_size:
                    self.embeddings[j] = embeddings[i]
                    self.sid[j] = sid[i]
            self.seen += 1

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        keep = min(self.seen, self.sample_size)
        return self.embeddings[:keep], self.sid[:keep]


def _gini(counts: np.ndarray) -> float:
    counts = counts.astype(np.float64)
    if counts.size == 0 or counts.sum() == 0:
        return 0.0
    sorted_counts = np.sort(counts)
    n = counts.size
    return float((2.0 * np.arange(1, n + 1) @ sorted_counts) / (n * sorted_counts.sum()) - (n + 1) / n)


def sid_distribution_metrics(sid: np.ndarray, codebook_size: int = 256) -> dict:
    sid = sid.astype(np.int64, copy=False)
    n, levels = sid.shape
    per_level = []
    for level in range(levels):
        counts = np.bincount(sid[:, level], minlength=codebook_size)
        used = int(np.count_nonzero(counts))
        probs = counts[counts > 0] / max(n, 1)
        entropy = float(-(probs * np.log2(probs)).sum()) if probs.size else 0.0
        per_level.append({
            "level": int(level + 1),
            "used_codes": used,
            "codebook_size": int(codebook_size),
            "utilization": float(used / codebook_size),
            "entropy_bits": entropy,
            "entropy_normalized": float(entropy / math.log2(codebook_size)) if codebook_size > 1 else 0.0,
            "gini": _gini(counts),
            "max_token_share": float(counts.max() / max(n, 1)),
            "min_nonzero_count": int(counts[counts > 0].min()) if used else 0,
            "max_count": int(counts.max()) if counts.size else 0,
        })
    return {"num_items": int(n), "n_levels": int(levels), "per_level": per_level}


def _pack_sid(sid: np.ndarray, prefix_len: int) -> np.ndarray:
    sid = sid[:, :prefix_len].astype(np.uint64, copy=False)
    key = np.zeros(sid.shape[0], dtype=np.uint64)
    for i in range(prefix_len):
        key |= sid[:, i] << np.uint64(16 * i)
    return key


def collision_metrics(sid: np.ndarray) -> dict:
    sid = sid.astype(np.int64, copy=False)
    n, levels = sid.shape
    out = {"num_items": int(n), "prefix": []}
    for prefix_len in range(1, levels + 1):
        keys = _pack_sid(sid, prefix_len)
        _, counts = np.unique(keys, return_counts=True)
        unique = int(counts.size)
        collided_items = int(counts[counts > 1].sum())
        excess = int(n - unique)
        out["prefix"].append({
            "prefix_len": int(prefix_len),
            "unique_paths": unique,
            "collision_groups": int(np.count_nonzero(counts > 1)),
            "items_in_collision_groups": collided_items,
            "items_in_collision_groups_rate": float(collided_items / max(n, 1)),
            "excess_collision_items": excess,
            "excess_collision_rate": float(excess / max(n, 1)),
            "max_bucket_size": int(counts.max()) if counts.size else 0,
        })
    out["full_sid"] = out["prefix"][-1] if out["prefix"] else {}
    return out


class ReconstructionAccumulator:
    def __init__(self) -> None:
        self.n = 0
        self.sse = 0.0
        self.sae = 0.0
        self.cos_err_sum = 0.0
        self.orig_norm_sum = 0.0
        self.recon_norm_sum = 0.0

    def update(self, original: np.ndarray, reconstructed: np.ndarray) -> None:
        original = original.astype(np.float32, copy=False)
        reconstructed = reconstructed.astype(np.float32, copy=False)
        diff = original - reconstructed
        self.n += int(original.shape[0])
        self.sse += float(np.square(diff, dtype=np.float32).sum())
        self.sae += float(np.abs(diff).sum())
        dot = np.sum(original * reconstructed, axis=1)
        on = np.linalg.norm(original, axis=1)
        rn = np.linalg.norm(reconstructed, axis=1)
        cos = dot / np.maximum(on * rn, 1e-12)
        self.cos_err_sum += float((1.0 - cos).sum())
        self.orig_norm_sum += float(on.sum())
        self.recon_norm_sum += float(rn.sum())

    def compute(self, dim: int) -> dict:
        denom = max(self.n * dim, 1)
        return {
            "num_items": int(self.n),
            "dim": int(dim),
            "mse": float(self.sse / denom),
            "mae": float(self.sae / denom),
            "mean_cosine_error": float(self.cos_err_sum / max(self.n, 1)),
            "mean_original_l2": float(self.orig_norm_sum / max(self.n, 1)),
            "mean_reconstructed_l2": float(self.recon_norm_sum / max(self.n, 1)),
        }


def common_prefix_len(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    eq = a == b
    out = np.zeros(eq.shape[0], dtype=np.int32)
    still = np.ones(eq.shape[0], dtype=bool)
    for level in range(eq.shape[1]):
        still &= eq[:, level]
        out += still.astype(np.int32)
    return out


def sampled_locality_metrics(
    embeddings: np.ndarray,
    sid: np.ndarray,
    query_size: int = 5000,
    pool_size: int = 200000,
    topk: int = 10,
    seed: int = 42,
) -> dict:
    n = embeddings.shape[0]
    if n < 2:
        return {"skipped": True, "reason": "need at least two items"}

    rng = np.random.default_rng(seed)
    pool_n = min(int(pool_size), n)
    query_n = min(int(query_size), n)
    pool_idx = rng.choice(n, size=pool_n, replace=False)
    query_idx = rng.choice(n, size=query_n, replace=False)

    xb = np.ascontiguousarray(embeddings[pool_idx].astype(np.float32, copy=False))
    xq = np.ascontiguousarray(embeddings[query_idx].astype(np.float32, copy=False))
    faiss.normalize_L2(xb)
    faiss.normalize_L2(xq)

    index = faiss.IndexFlatIP(xb.shape[1])
    index.add(xb)
    search_k = min(topk + 1, pool_n)
    sims, neigh = index.search(xq, search_k)

    hamming_vals = []
    prefix_vals = []
    sim_vals = []
    exact_sid = 0
    same_l1 = 0
    same_l2 = 0
    pairs = 0

    pool_pos_to_global = pool_idx
    for qi in range(query_n):
        q_global = query_idx[qi]
        taken = 0
        for rank in range(search_k):
            p_global = pool_pos_to_global[neigh[qi, rank]]
            if p_global == q_global:
                continue
            a = sid[q_global]
            b = sid[p_global]
            hamming_vals.append(int(np.count_nonzero(a != b)))
            pref = int(common_prefix_len(a[None, :], b[None, :])[0])
            prefix_vals.append(pref)
            sim_vals.append(float(sims[qi, rank]))
            exact_sid += int(pref == sid.shape[1])
            same_l1 += int(pref >= 1)
            same_l2 += int(pref >= 2)
            pairs += 1
            taken += 1
            if taken >= topk:
                break

    if pairs == 0:
        return {"skipped": True, "reason": "no non-self neighbors found"}

    return {
        "skipped": False,
        "query_size": int(query_n),
        "pool_size": int(pool_n),
        "topk": int(topk),
        "pairs": int(pairs),
        "mean_neighbor_cosine": float(np.mean(sim_vals)),
        "mean_sid_hamming": float(np.mean(hamming_vals)),
        "mean_common_prefix_len": float(np.mean(prefix_vals)),
        "same_level1_rate": float(same_l1 / pairs),
        "same_level12_rate": float(same_l2 / pairs),
        "same_full_sid_rate": float(exact_sid / pairs),
    }


def build_metric_report(
    sid: np.ndarray,
    codebook_size: int,
    reconstruction: dict | None = None,
    locality: dict | None = None,
    extra: dict | None = None,
) -> dict:
    report = {
        "sid_distribution": sid_distribution_metrics(sid, codebook_size),
        "collision": collision_metrics(sid),
    }
    if reconstruction is not None:
        report["reconstruction"] = reconstruction
    if locality is not None:
        report["sampled_locality"] = locality
    if extra:
        report["extra"] = extra
    return report
