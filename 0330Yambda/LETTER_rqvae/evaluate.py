#!/usr/bin/env python3
"""
LETTER Tokenization 质量评估脚本

评估维度（基于 LETTER 论文）：
  1. 重构质量        - MSE / Cosine Sim / Pearson Corr
  2. Token 分布      - Zipf / Gini / Per-level 熵
  3. 码本利用率      - 每层 active codes / 均衡程度
  4. 近邻召回        - Top-k 近邻在 token 空间 vs embedding 空间的一致率
  5. Token 序列分析  - 多样性 / 碰撞率
  6. CF 对齐质量     - 量化向量与 CF embedding 的对齐程度（新增）
  7. Diversity 质量  - 码字簇分离度（新增）

用法：
    python evaluate.py \
        --ckpt ./checkpoints/.../best_entropy_e20.pth \
        --data ../data/embeddings.npz \
        --output ./eval_results.json

    # 有 CF embeddings 时（额外评估 CF 对齐）
    python evaluate.py \
        --ckpt ./checkpoints/.../best_entropy_e20.pth \
        --data ../data/embeddings.npz \
        --cf_embeddings ../data/cf_embeddings.npz \
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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import LETTER
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
    freqs = np.sort(freqs)
    n = len(freqs)
    total = freqs.sum()
    if total == 0 or n == 0:
        return 0.0
    gini_numer = 2 * np.sum(np.arange(1, n + 1) * freqs)
    gini_denom = n * total
    return float(gini_numer / gini_denom) - (n + 1) / n


def cosine_similarity_matrix(a, b):
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_norm @ b_norm.T


def topk_recall(emb_sim, tok_sim, k):
    n = emb_sim.shape[0]
    k = min(k, n - 1)
    emb_topk = {
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
    N, L = codes.shape
    out = np.zeros((N, L * codebook_size), dtype=np.float32)
    for l in range(L):
        for n in range(N):
            out[n, l * codebook_size + codes[n, l]] = 1.0
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class LETTEREvaluator:
    def __init__(self, ckpt_path, npz_path, cf_npz_path=None,
                 device='cuda:0', batch_size=4096, num_workers=8,
                 sample_size=None, nn_sample=5000, nn_k=20,
                 cf_proj_dim=32):
        self.device = torch.device(device)
        self.nn_sample = nn_sample
        self.nn_k = nn_k
        self.sample_size = sample_size
        self.cf_proj_dim = cf_proj_dim
        self.cf_npz_path = cf_npz_path

        # ── Load dataset ──────────────────────────────────────────────────
        self.dataset = EmbDataset(npz_path=npz_path)
        self.all_embs = self.dataset.embeddings
        self.N = self.all_embs.shape[0]

        if sample_size and sample_size < self.N:
            rng = np.random.default_rng(42)
            idx = rng.choice(self.N, sample_size, replace=False)
            self.all_embs = self.all_embs[idx]
            self.N = sample_size
            print(f"  Sampled {self.N} items for evaluation")

        loader = DataLoader(
            self.dataset if not sample_size else
            torch.utils.data.TensorDataset(
                torch.from_numpy(self.all_embs).float(),
                torch.zeros(len(self.all_embs), dtype=torch.long)
            ),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

        # ── Load model ───────────────────────────────────────────────────
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)
        args_c = ckpt.get('args', {})

        model = LETTER(
            input_dim=self.dataset.dim,
            code_dim=args_c.get('code_dim', 32),
            num_levels=args_c.get('num_levels', 4),
            codebook_size=args_c.get('codebook_size', 256),
            commitment_mu=args_c.get('commitment_mu', 0.25),
            use_recon=args_c.get('use_recon', True),
            cf_dim=args_c.get('cf_dim', 64),
            cf_proj_dim=args_c.get('cf_proj_dim', 32),
            div_cluster_k=args_c.get('div_cluster_k', 8),
        )
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        self.model = model
        self.codebook_size = model.codebook_size
        self.num_levels = model.num_levels
        self.code_dim = model.code_dim
        print(f"  Model loaded: {ckpt_path}")

        # ── Load CF embeddings ───────────────────────────────────────────
        self.cf_embs = None
        if cf_npz_path and os.path.exists(cf_npz_path):
            cf_data = np.load(cf_npz_path)
            cf_all = cf_data['cf_embeddings'].astype(np.float32)
            if sample_size and sample_size < len(cf_all):
                rng = np.random.default_rng(42)
                idx = rng.choice(len(cf_all), sample_size, replace=False)
                cf_all = cf_all[idx]
            self.cf_embs = cf_all
            print(f"  CF embeddings loaded: {self.cf_embs.shape}")

        # ── Encode all items ─────────────────────────────────────────────
        print("  Encoding all items ...")
        t0 = time.time()
        all_codes = []
        all_z_q = []
        all_recon = []

        with torch.no_grad():
            for (x, _) in tqdm(loader, ncols=60, desc="  Encoding"):
                x = x.to(self.device)
                codes = model.get_codes(x)
                all_codes.append(codes.cpu().numpy())
                if model.use_recon:
                    recon_x, z_q, _, _ = model(x, alpha=0.0, beta=0.0)
                    all_recon.append(recon_x.cpu().numpy())
                    all_z_q.append(z_q.cpu().numpy())

        self.all_codes = np.concatenate(all_codes, axis=0)
        self.all_z_q = np.concatenate(all_z_q, axis=0) if all_z_q else None
        self.all_recon = np.concatenate(all_recon, axis=0) if all_recon else None
        print(f"  Done in {time.time() - t0:.1f}s, codes: {self.all_codes.shape}")

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

        return {
            'mse': mse,
            'cosine_sim_mean': mean_cos,
            'cosine_sim_std': std_cos,
            'cosine_sim_p90': float(np.percentile(cos_sim, 90)),
            'cosine_sim_p50': float(np.percentile(cos_sim, 50)),
            'cosine_sim_p10': float(np.percentile(cos_sim, 10)),
            'pearson_r': pearson_r,
        }

    # ── 2. Token distribution ─────────────────────────────────────────────
    def eval_token_distribution(self):
        codes = self.all_codes
        N, L = codes.shape
        K = self.codebook_size

        per_level = {}
        for l in range(L):
            lvl_codes = codes[:, l]
            cnt = np.bincount(lvl_codes, minlength=K)
            active = int(np.sum(cnt > 0))
            entropy = _shannon_entropy(cnt)
            max_ent = np.log2(K)
            gini = gini_coefficient(cnt.astype(float))

            per_level[f'level_{l}'] = {
                'active_codes': active,
                'utilization': active / K,
                'entropy': entropy,
                'norm_entropy': entropy / max_ent,
                'gini': gini,
            }

        tuples = [str(c.tolist()) for c in codes]
        unique_tokens = len(set(tuples))
        tuple_gini = gini_coefficient(
            np.array(list(Counter(tuples).values()), dtype=float)
        )

        corr_matrix = np.corrcoef(codes.T)

        return {
            'unique_4tuples': unique_tokens,
            'total_items': N,
            'tuple_collision_rate': 1 - unique_tokens / N,
            'tuple_gini': float(tuple_gini),
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

        tuple_counter = Counter(str(c.tolist()) for c in codes)
        most_common_count = tuple_counter.most_common(1)[0][1]

        return {
            'global_active_codes': global_active,
            'global_utilization': global_active / K,
            'global_entropy': entropy,
            'global_norm_entropy': entropy / max_ent,
            'max_item_share': most_common_count,
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

        pbar = tqdm(total=100, ncols=60, desc="  Tok sim")
        tok_onehot = token_to_onehot(codes, self.codebook_size).astype(np.float64)
        tok_chunks = np.array_split(tok_onehot, 10)
        tok_sim_rows = []
        for chunk in tok_chunks:
            rows = cosine_similarity_matrix(chunk, tok_onehot)
            np.fill_diagonal(rows, -1)
            tok_sim_rows.append(rows)
            pbar.update(10)
        tok_sim = np.vstack(tok_sim_rows)
        pbar.close()

        recall = topk_recall(emb_sim, tok_sim, k)

        n_sub = min(5000, n * n)
        iu = np.triu_indices(n, k=1)
        si = rng.choice(len(iu[0]), min(500000, len(iu[0])), replace=False)
        e_flat = emb_sim[(iu[0][si], iu[1][si])]
        t_flat = tok_sim[(iu[0][si], iu[1][si])]
        spearman_r, _ = stats.spearmanr(e_flat, t_flat)

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

        # z_q space similarity (uses decoded z_q)
        if self.all_z_q is not None:
            z_q = self.all_z_q[idx].astype(np.float64)
            pbar = tqdm(total=100, ncols=60, desc="  z_q sim")
            zq_chunks = np.array_split(z_q, 10)
            zq_sim_rows = []
            for chunk in zq_chunks:
                rows = cosine_similarity_matrix(chunk, z_q)
                np.fill_diagonal(rows, -1)
                zq_sim_rows.append(rows)
                pbar.update(10)
            zq_sim = np.vstack(zq_sim_rows)
            pbar.close()
            recall_zq = topk_recall(emb_sim, zq_sim, k)
            mr_zq = 0.0
            for i in range(0, n, step):
                zq_rank = np.argsort(-zq_sim[i])
                zq_rr = np.where(zq_rank == i)[0]
                if len(zq_rr) > 0:
                    mr_zq += 1.0 / (zq_rr[0] + 1)
            mr_zq /= (n // step)
        else:
            recall_zq = None
            mr_zq = None

        return {
            'nn_recall': float(recall),
            'spearman_r': float(spearman_r),
            'mrr': float(mr_score),
            'nn_recall_z_q': float(recall_zq) if recall_zq is not None else None,
            'mrr_z_q': float(mr_zq) if mr_zq is not None else None,
            'nn_k': k,
            'nn_sample_size': n,
        }

    # ── 5. Clustering quality ───────────────────────────────────────────────
    def eval_clustering(self, n_clusters=128):
        codes = self.all_codes
        N, L = codes.shape
        n_clusters = min(n_clusters, N)

        tuple_to_id = {}
        for i, t in enumerate(codes):
            key = str(t.tolist())
            if key not in tuple_to_id:
                tuple_to_id[key] = len(tuple_to_id)
        token_labels = np.array([tuple_to_id[str(t.tolist())] for t in codes])

        pbar = tqdm(ncols=60, desc="  KMeans")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        emb_cluster_labels = kmeans.fit_predict(self.all_embs)
        pbar.update(100)

        sil_emb = silhouette_score(
            self.all_embs[:min(50000, N)], emb_cluster_labels[:min(50000, N)]
        )
        tok_onehot = token_to_onehot(codes, self.codebook_size)
        sil_tok = silhouette_score(
            tok_onehot[:min(50000, N)], token_labels[:min(50000, N)]
        )
        nmi = normalized_mutual_info_score(token_labels, emb_cluster_labels)

        return {
            'n_clusters': n_clusters,
            'silhouette_emb': float(sil_emb),
            'silhouette_tok': float(sil_tok),
            'nmi_emb_tok': float(nmi),
        }

    # ── 6. CF Alignment Quality（LETTER 特色）──────────────────────────────
    def eval_cf_alignment(self, sample_size=10000):
        if self.cf_embs is None:
            return {}

        n = min(sample_size, self.N)
        rng = np.random.default_rng(42)
        idx = rng.choice(self.N, n, replace=False)

        z_q = self.all_z_q[idx].astype(np.float64)   # (n, code_dim)
        cf_embs = self.cf_embs[idx].astype(np.float64)  # (n, cf_dim)

        # Project both to same dimension for comparison
        min_dim = min(z_q.shape[1], cf_embs.shape[1])
        z_q_s = z_q[:, :min_dim]
        cf_s = cf_embs[:, :min_dim]

        # Cosine sim between z_q and cf_embs
        sim = cosine_similarity_matrix(z_q_s, cf_s)
        diag_sim = np.diag(sim)
        mean_sim = float(np.mean(diag_sim))

        # All-pair correlation
        iu = np.triu_indices(n, k=1)
        si = rng.choice(len(iu[0]), min(100000, len(iu[0])), replace=False)
        flat_sim = sim[(iu[0][si], iu[1][si])]

        return {
            'mean_alignment_cosim': mean_sim,
            'std_alignment_cosim': float(np.std(diag_sim)),
            'p90_alignment_cosim': float(np.percentile(diag_sim, 90)),
            'p50_alignment_cosim': float(np.percentile(diag_sim, 50)),
            'p10_alignment_cosim': float(np.percentile(diag_sim, 10)),
        }

    # ── 7. Diversity Quality（LETTER 特色）────────────────────────────────
    def eval_diversity_quality(self):
        codes = self.all_codes
        K = self.codebook_size
        L = self.num_levels

        per_level = {}
        for l in range(L):
            lvl_codes = codes[:, l]
            cnt = np.bincount(lvl_codes, minlength=K)

            # Entropy = higher is more diverse
            entropy = _shannon_entropy(cnt)
            max_ent = np.log2(K)
            norm_entropy = entropy / max_ent if max_ent > 0 else 0.0

            # Gini = lower is more uniform (inverse of diversity)
            gini = gini_coefficient(cnt.astype(float))

            per_level[f'level_{l}'] = {
                'norm_entropy': float(norm_entropy),
                'gini': float(gini),
                'top1_freq': float(cnt.max() / cnt.sum()) if cnt.sum() > 0 else 0,
            }

        # Global across all levels
        flat = codes.ravel()
        flat_cnt = np.bincount(flat, minlength=K)
        global_norm_ent = _shannon_entropy(flat_cnt) / np.log2(K)

        return {
            'global_norm_entropy': float(global_norm_ent),
            'per_level': per_level,
        }

    # ── Run all ─────────────────────────────────────────────────────────────
    def run(self):
        print("\n" + "=" * 70)
        print("LETTER Tokenization Evaluation")
        print("=" * 70)
        results = {'n_items': self.N}

        t0 = time.time()

        print("\n[1/7] Reconstruction quality...")
        r_recon = self.eval_reconstruction()
        results['reconstruction'] = r_recon
        print(f"      MSE={r_recon.get('mse', 'N/A'):.6f}  "
              f"CosSim={r_recon.get('cosine_sim_mean', 'N/A'):.4f}  "
              f"Pearson={r_recon.get('pearson_r', 'N/A'):.4f}")

        print("\n[2/7] Token distribution...")
        r_dist = self.eval_token_distribution()
        results['token_distribution'] = r_dist
        print(f"      Unique={r_dist['unique_4tuples']}  "
              f"Collision={r_dist['tuple_collision_rate']:.4f}")
        for l in range(self.num_levels):
            m = r_dist['per_level'][f'level_{l}']
            print(f"      Level {l}: active={m['active_codes']}/{self.codebook_size}  "
                  f"norm_H={m['norm_entropy']:.4f}  gini={m['gini']:.4f}")

        print("\n[3/7] Codebook utilization...")
        r_util = self.eval_codebook_utilization()
        results['codebook_utilization'] = r_util
        print(f"      Global active={r_util['global_active_codes']}/{self.codebook_size}  "
              f"norm_H={r_util['global_norm_entropy']:.4f}")

        print(f"\n[4/7] Nearest neighbor recall (sample={self.nn_sample})...")
        r_nn = self.eval_nn_recall()
        results['nn_recall'] = r_nn
        print(f"      Token Recall@{r_nn.get('nn_k','?')}={r_nn.get('nn_recall',0):.4f}  "
              f"Spearman={r_nn.get('spearman_r',0):.4f}  "
              f"MRR={r_nn.get('mrr',0):.4f}")
        if r_nn.get('nn_recall_z_q') is not None:
            print(f"      z_q   Recall@{r_nn.get('nn_k','?')}={r_nn.get('nn_recall_z_q',0):.4f}  "
                  f"MRR={r_nn.get('mrr_z_q',0):.4f}")

        print(f"\n[5/7] Clustering quality (k=128)...")
        r_clust = self.eval_clustering()
        results['clustering'] = r_clust
        print(f"      Silhouette emb={r_clust['silhouette_emb']:.4f}  "
              f"silhouette tok={r_clust['silhouette_tok']:.4f}  "
              f"NMI={r_clust['nmi_emb_tok']:.4f}")

        if self.cf_embs is not None:
            print(f"\n[6/7] CF Alignment quality...")
            r_cf = self.eval_cf_alignment()
            results['cf_alignment'] = r_cf
            print(f"      Mean cosim={r_cf.get('mean_alignment_cosim', 'N/A'):.4f}  "
                  f"p50={r_cf.get('p50_alignment_cosim', 'N/A'):.4f}  "
                  f"p10={r_cf.get('p10_alignment_cosim', 'N/A'):.4f}")

        print(f"\n[7/7] Diversity quality...")
        r_div = self.eval_diversity_quality()
        results['diversity'] = r_div
        print(f"      Global norm_H={r_div['global_norm_entropy']:.4f}")
        for l in range(self.num_levels):
            m = r_div['per_level'][f'level_{l}']
            print(f"      Level {l}: norm_H={m['norm_entropy']:.4f}  "
                  f"gini={m['gini']:.4f}  top1={m['top1_freq']:.4f}")

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
    parser.add_argument('--cf_embeddings', type=str, default=None,
                        help='CF embeddings npz path')
    parser.add_argument('--output', type=str, default='../data/letter_eval_results.json')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--sample_size', type=int, default=None)
    parser.add_argument('--nn_sample', type=int, default=5000)
    parser.add_argument('--nn_k', type=int, default=20)
    parser.add_argument('--cf_proj_dim', type=int, default=32)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    evaluator = LETTEREvaluator(
        ckpt_path=args.ckpt,
        npz_path=args.data,
        cf_npz_path=args.cf_embeddings,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_size=args.sample_size,
        nn_sample=args.nn_sample,
        nn_k=args.nn_k,
        cf_proj_dim=args.cf_proj_dim,
    )

    results = evaluator.run()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {args.output}")
