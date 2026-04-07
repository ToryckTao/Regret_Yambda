#!/usr/bin/env python3
"""
RQVAE Tokenization 质量评估脚本

评估维度：
  1. 重构质量   - MSE / Cosine Sim / Pearson Corr（全量数据）
  2. Token 分布 - Zipf 图 / Gini 系数 / Per-level 熵
  3. 码本利用率 - 每层 active codes / 均衡程度
  4. 近邻召回   - Top-k 近邻在 token 空间 vs embedding 空间的一致率
  5. Token 序列分析 - 多样性 / 平均 unique token 数 / 碰撞矩阵

用法：
    python evaluate.py \
        --ckpt ./checkpoints/Mar-30-2026_23-12-08/best_entropy_e20.pth \
        --data ../data/embeddings.npz \
        --output ./eval_results.json
"""
import argparse
import json
import time
import os
import warnings
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import RQVAE
from dataset import EmbDataset


# ─────────────────────────────────────────────────────────────────────────────
#  Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _shannon_entropy(counts):
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-np.sum(probs * np.log2(probs)))


def gini_coefficient(freqs):
    """越小越均匀，0 = 完全均匀，1 = 极度不均。"""
    freqs = np.sort(freqs)
    n = len(freqs)
    total = freqs.sum()
    if total == 0 or n == 0:
        return 0.0
    # 归一化到 [0, 1]
    gini_numer = 2 * np.sum(np.arange(1, n + 1) * freqs)
    gini_denom = n * total
    return float(gini_numer / gini_denom) - (n + 1) / n


def cosine_similarity_matrix(a, b):
    """(N, D) vs (M, D) → (N, M) cosine sim"""
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_norm @ b_norm.T


def topk_recall(emb_sim, tok_sim, k):
    """两个 (N, N) 相似度矩阵，返回 top-k 近邻召回率"""
    n = emb_sim.shape[0]
    k = min(k, n - 1)
    emb_topk = set() if False else {
        frozenset([i, j]) for i in range(n)
        for j in np.argsort(-emb_sim[i])[1:k + 1]
    }
    tok_topk = {
        frozenset([i, j]) for i in range(n)
        for j in np.argsort(-tok_sim[i])[1:k + 1]
    }
    if not emb_topk:
        return 0.0
    return len(emb_topk & tok_topk) / len(emb_topk)


def token_to_onehot(codes, codebook_size):
    """codes: (N, num_levels) → (N, num_levels * codebook_size) one-hot"""
    N, L = codes.shape
    out = np.zeros((N, L * codebook_size), dtype=np.float32)
    for l in range(L):
        for n in range(N):
            out[n, l * codebook_size + codes[n, l]] = 1.0
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Main evaluator
# ─────────────────────────────────────────────────────────────────────────────

class Evaluator:
    def __init__(self, ckpt_path, npz_path, device='cuda:0', batch_size=4096,
                 num_workers=8, sample_size=None, nn_sample=5000, nn_k=20):
        self.device = torch.device(device)
        self.nn_sample = nn_sample
        self.nn_k = nn_k
        self.sample_size = sample_size

        # ── Load dataset ──────────────────────────────────────────────────
        self.dataset = EmbDataset(npz_path=npz_path)
        self.all_embs = self.dataset.embeddings  # (N, D) numpy
        self.N = self.all_embs.shape[0]

        if sample_size and sample_size < self.N:
            rng = np.random.default_rng(42)
            idx = rng.choice(self.N, sample_size, replace=False)
            self.all_embs = self.all_embs[idx]
            self.N = sample_size
            print(f"  Sampled {self.N} items for evaluation")

        loader = DataLoader(
            self.dataset if not sample_size else torch.utils.data.TensorDataset(
                torch.from_numpy(self.all_embs).float(),
                torch.zeros(len(self.all_embs), dtype=torch.long)
            ),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

        # ── Load model ───────────────────────────────────────────────────
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)
        model = RQVAE(
            input_dim=self.dataset.dim,
            code_dim=32,
            num_levels=4,
            codebook_size=256,
            use_recon=True,
        )
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        self.model = model
        self.codebook_size = 256
        self.num_levels = 4
        print(f"  Model loaded: {ckpt_path}")

        # ── Encode all ────────────────────────────────────────────────────
        print("  Encoding all items...")
        t0 = time.time()
        all_codes = []
        all_recon = []
        with torch.no_grad():
            for (x, _) in loader:
                x = x.to(self.device)
                codes = model.get_codes(x)
                all_codes.append(codes.cpu().numpy())
                if model.use_recon:
                    recon_x, _, _, _ = model(x)
                    all_recon.append(recon_x.cpu().numpy())

        self.all_codes = np.concatenate(all_codes, axis=0)  # (N, 4)
        self.all_recon = np.concatenate(all_recon, axis=0) if all_recon else None
        print(f"  Done encoding in {time.time() - t0:.1f}s, shape: {self.all_codes.shape}")

    # ── 1. Reconstruction quality ─────────────────────────────────────────
    def eval_reconstruction(self):
        if self.all_recon is None:
            return {}
        embs = self.all_embs
        recon = self.all_recon

        mse = float(np.mean((embs - recon) ** 2))

        cos_sim = np.diag(cosine_similarity_matrix(embs, recon))
        mean_cos = float(np.mean(cos_sim))
        std_cos = float(np.std(cos_sim))

        pearson_r = float(stats.pearsonr(embs.ravel(), recon.ravel())[0])

        # Per-sample cosine sim distribution
        pct_90 = float(np.percentile(cos_sim, 90))
        pct_50 = float(np.percentile(cos_sim, 50))
        pct_10 = float(np.percentile(cos_sim, 10))

        return {
            'mse': mse,
            'cosine_sim_mean': mean_cos,
            'cosine_sim_std': std_cos,
            'cosine_sim_p90': pct_90,
            'cosine_sim_p50': pct_50,
            'cosine_sim_p10': pct_10,
            'pearson_r': pearson_r,
        }

    # ── 2. Token distribution ─────────────────────────────────────────────
    def eval_token_distribution(self):
        codes = self.all_codes
        N, L = codes.shape
        K = self.codebook_size

        # Per-level stats
        per_level = {}
        for l in range(L):
            lvl_codes = codes[:, l]
            cnt = np.bincount(lvl_codes, minlength=K)
            active = int(np.sum(cnt > 0))
            entropy = _shannon_entropy(cnt)
            max_ent = np.log2(K)
            gini = gini_coefficient(cnt.astype(float))
            freq_top1 = float(cnt.max() / cnt.sum()) if cnt.sum() > 0 else 0

            per_level[f'level_{l}'] = {
                'active_codes': active,
                'utilization': active / K,
                'entropy': entropy,
                'norm_entropy': entropy / max_ent,
                'gini': gini,
                'freq_top1': freq_top1,
            }

        # Global (4-tuple as one token)
        tuples = [str(c.tolist()) for c in codes]
        unique_tokens = len(set(tuples))
        tuple_cnt = Counter(tuples)
        tuple_freqs = np.array(list(tuple_cnt.values()), dtype=float)
        tuple_gini = gini_coefficient(tuple_freqs)

        # Cross-level correlation (how independent are levels?)
        corr_matrix = np.corrcoef(codes.T)  # (L, L)

        # Average unique tokens per item (always 1 per item since each has 4-tuple)
        # But in cross-level: how many unique tokens per level
        avg_unique_per_level = np.mean([per_level[f'level_{l}']['active_codes'] for l in range(L)])

        return {
            'unique_4tuples': unique_tokens,
            'total_items': N,
            'tuple_collision_rate': 1 - unique_tokens / N,
            'tuple_gini': float(tuple_gini),
            'avg_unique_per_level': float(avg_unique_per_level),
            'cross_level_corr': corr_matrix.tolist(),
            'per_level': per_level,
        }

    # ── 3. Codebook utilization ───────────────────────────────────────────
    def eval_codebook_utilization(self):
        codes = self.all_codes
        N, L = codes.shape
        K = self.codebook_size

        flat_all = codes.ravel()
        global_active = int(np.sum(np.bincount(flat_all, minlength=K) > 0))
        counts = np.bincount(flat_all, minlength=K)
        entropy = _shannon_entropy(counts)
        max_ent = np.log2(K)

        # How many items share the most popular token?
        tuple_counter = Counter(str(c.tolist()) for c in codes)
        most_common_count = tuple_counter.most_common(1)[0][1]
        top10_pct = sum(v for _, v in tuple_counter.most_common(N // 10)) / N if N > 0 else 0

        return {
            'global_active_codes': global_active,
            'global_utilization': global_active / K,
            'global_entropy': entropy,
            'global_norm_entropy': entropy / max_ent,
            'max_item_share': most_common_count,
            'top10_item_fraction': top10_pct,
        }

    # ── 4. Nearest neighbor recall ────────────────────────────────────────
    def eval_nn_recall(self, sample_size=None, k=20):
        sample_size = self.nn_sample if sample_size is None else sample_size
        k = min(self.nn_k, sample_size - 1)
        n = min(sample_size, self.N)
        if n <= k + 1:
            return {}

        rng = np.random.default_rng(123)
        idx = rng.choice(self.N, n, replace=False)

        embs = self.all_embs[idx].astype(np.float64)
        codes = self.all_codes[idx]

        # Embedding space similarity  (chunked for progress visibility)
        pbar = tqdm(total=100, ncols=60, desc="  Emb sim")
        emb_chunks = np.array_split(embs, 10)
        emb_sim_rows = []
        for chunk in emb_chunks:
            rows = cosine_similarity_matrix(chunk, embs)
            np.fill_diagonal(rows, -1)
            emb_sim_rows.append(rows)
            pbar.update(10)
        emb_sim = np.vstack(emb_sim_rows)
        pbar.close()

        # Token space similarity (one-hot → cosine)
        tok_onehot = token_to_onehot(codes, self.codebook_size).astype(np.float64)
        pbar = tqdm(total=100, ncols=60, desc="  Tok sim")
        tok_chunks = np.array_split(tok_onehot, 10)
        tok_sim_rows = []
        for chunk in tok_chunks:
            rows = cosine_similarity_matrix(chunk, tok_onehot)
            np.fill_diagonal(rows, -1)
            tok_sim_rows.append(rows)
            pbar.update(10)
        tok_sim = np.vstack(tok_sim_rows)
        pbar.close()

        # Top-k recall
        recall = topk_recall(emb_sim, tok_sim, k)

        # Spearman correlation between two similarity matrices (subsample)
        n_sub = min(5000, n * n)
        iu = np.triu_indices(n, k=1)
        # sample a subset for speed
        if n_sub > 500000:
            si = rng.choice(len(iu[0]), 500000, replace=False)
            e_flat = emb_sim[(iu[0][si], iu[1][si])]
            t_flat = tok_sim[(iu[0][si], iu[1][si])]
        else:
            e_flat = emb_sim[iu]
            t_flat = tok_sim[iu]
        spearman_r, _ = stats.spearmanr(e_flat, t_flat)

        # MRR (Mean Reciprocal Rank)
        mr_score = 0.0
        step = max(1, n // 500)
        for i in range(0, n, step):
            emb_rank = np.argsort(-emb_sim[i])
            tok_rank = np.argsort(-tok_sim[i])
            emb_rr = np.where(emb_rank == i)[0]
            tok_rr = np.where(tok_rank == i)[0]
            if len(tok_rr) > 0:
                mr_score += 1.0 / (tok_rr[0] + 1)
        mr_score /= (n // step)

        return {
            'nn_recall': float(recall),
            'spearman_r': float(spearman_r),
            'mrr': float(mr_score),
            'nn_k': k,
            'nn_sample_size': n,
        }

    # ── 5. Clustering quality ───────────────────────────────────────────────
    def eval_clustering(self, n_clusters=128):
        codes = self.all_codes
        N, L = codes.shape
        n_clusters = min(n_clusters, N)

        # Token-space clustering (4-tuple as label)
        tuple_to_id = {}
        for i, t in enumerate(codes):
            key = str(t.tolist())
            if key not in tuple_to_id:
                tuple_to_id[key] = len(tuple_to_id)
        token_labels = np.array([tuple_to_id[str(t.tolist())] for t in codes])

        # KMeans on embedding space
        pbar = tqdm(ncols=60, desc="  KMeans")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        emb_cluster_labels = kmeans.fit_predict(self.all_embs)
        pbar.update(100)
        pbar.close()

        # Silhouette on embedding space
        sil_emb_pbar = tqdm(total=None, ncols=60, desc="  Silhouette Emb")
        sil_emb = silhouette_score(self.all_embs[:min(50000, N)], emb_cluster_labels[:min(50000, N)])
        sil_emb_pbar.close()

        # Silhouette on token one-hot space
        sil_tok_pbar = tqdm(total=None, ncols=60, desc="  Silhouette Tok")
        tok_onehot = token_to_onehot(codes, self.codebook_size)
        sil_tok = silhouette_score(tok_onehot[:min(50000, N)], token_labels[:min(50000, N)])
        sil_tok_pbar.close()

        # NMI between token labels and embedding kmeans
        nmi = normalized_mutual_info_score(token_labels, emb_cluster_labels)

        return {
            'n_clusters': n_clusters,
            'silhouette_emb': float(sil_emb),
            'silhouette_tok': float(sil_tok),
            'nmi_emb_tok': float(nmi),
        }

    # ── Run all ─────────────────────────────────────────────────────────────
    def run(self):
        print("\n" + "=" * 70)
        print("RQVAE Tokenization Evaluation")
        print("=" * 70)
        results = {'n_items': self.N}

        t0 = time.time()

        print("\n[1/5] Reconstruction quality...")
        r_recon = self.eval_reconstruction()
        results['reconstruction'] = r_recon
        print(f"      MSE={r_recon.get('mse', 'N/A'):.6f}  "
              f"CosSim={r_recon.get('cosine_sim_mean', 'N/A'):.4f}  "
              f"Pearson={r_recon.get('pearson_r', 'N/A'):.4f}")

        print("\n[2/5] Token distribution...")
        r_dist = self.eval_token_distribution()
        results['token_distribution'] = r_dist
        print(f"      Unique 4tuples={r_dist['unique_4tuples']}  "
              f"Collision={r_dist['tuple_collision_rate']:.4f}  "
              f"Gini={r_dist['tuple_gini']:.4f}")
        for l in range(self.num_levels):
            m = r_dist['per_level'][f'level_{l}']
            print(f"      Level {l}: active={m['active_codes']}/{self.codebook_size}  "
                  f"norm_H={m['norm_entropy']:.4f}  gini={m['gini']:.4f}")

        print("\n[3/5] Codebook utilization...")
        r_util = self.eval_codebook_utilization()
        results['codebook_utilization'] = r_util
        print(f"      Global active={r_util['global_active_codes']}/{self.codebook_size}  "
              f"util={r_util['global_utilization']:.4f}  "
              f"norm_H={r_util['global_norm_entropy']:.4f}")

        print(f"\n[4/5] Nearest neighbor recall (sample={self.nn_sample})...")
        r_nn = self.eval_nn_recall()
        results['nn_recall'] = r_nn
        print(f"      Recall@{r_nn.get('nn_k','?')}={r_nn.get('nn_recall',0):.4f}  "
              f"Spearman={r_nn.get('spearman_r',0):.4f}  "
              f"MRR={r_nn.get('mrr',0):.4f}")

        print(f"\n[5/5] Clustering quality (k=128)...")
        r_clust = self.eval_clustering()
        results['clustering'] = r_clust
        print(f"      Silhouette emb={r_clust['silhouette_emb']:.4f}  "
              f"silhouette tok={r_clust['silhouette_tok']:.4f}  "
              f"NMI={r_clust['nmi_emb_tok']:.4f}")

        elapsed = time.time() - t0
        results['eval_time_seconds'] = round(elapsed, 1)
        print(f"\n{'=' * 70}")
        print(f"Evaluation done in {elapsed:.1f}s")
        print("=" * 70)

        return results


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--data', type=str, default='../data/embeddings.npz')
    parser.add_argument('--output', type=str, default='../data/eval_results.json')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--sample_size', type=int, default=None,
                        help='采样评估的 item 数，默认全量')
    parser.add_argument('--nn_sample', type=int, default=5000,
                        help='近邻召回采样数')
    parser.add_argument('--nn_k', type=int, default=20,
                        help='近邻召回的 k 值')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    evaluator = Evaluator(
        ckpt_path=args.ckpt,
        npz_path=args.data,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_size=args.sample_size,
        nn_sample=args.nn_sample,
        nn_k=args.nn_k,
    )

    results = evaluator.run()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {args.output}")
