"""
LETTER 训练器

损失组成（论文 Eq.(5)）：
  L_LETTER = L_Sem + α·L_CF + β·L_Div

其中：
  L_Sem  = L_Recon + L_RQVAE
  L_RQVAE = Σ_l ‖sg[r_l-1] − e_cl‖² + μ·‖r_l-1 − sg[e_cl]‖²
  L_CF   = In-batch NCE（对齐量化向量与 CF embedding）
  L_Div  = Diversity NCE（每层码字簇内拉近，簇间推开）
"""
import os
import time
import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import get_local_time, ensure_dir


# ─────────────────────────────────────────────────────────────────────────────
#  码本质量评估工具（与 pure_rqvae 兼容）
# ─────────────────────────────────────────────────────────────────────────────

def _shannon_entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-np.sum(probs * np.log2(probs)))


def compute_codebook_metrics(all_codes: torch.Tensor, codebook_size: int):
    """
    计算码本质量指标，按层 + 整体输出。

    all_codes: (N, num_levels)
    codebook_size: 每层码本大小 K=256
    """
    num_levels = all_codes.size(1)
    N = all_codes.size(0)

    code_tuples = [str(c.tolist()) for c in all_codes]
    unique_tokens = len(set(code_tuples))
    collision_rate = (N - unique_tokens) / N if N > 0 else 0.0

    overall = {
        'unique_tokens': unique_tokens,
        'collision_rate': collision_rate,
    }

    per_level = []
    for lvl in range(num_levels):
        lvl_codes = all_codes[:, lvl].numpy()
        unique_lvl = len(set(lvl_codes))
        counts = np.bincount(lvl_codes, minlength=codebook_size)
        entropy = _shannon_entropy(counts)
        max_entropy = float(np.log2(codebook_size))

        per_level.append({
            'active_codes': unique_lvl,
            'utilization': unique_lvl / codebook_size,
            'entropy': entropy,
            'norm_entropy': entropy / max_entropy if max_entropy > 0 else 0.0,
        })

    all_flat = all_codes.view(-1).numpy()
    global_counts = np.bincount(all_flat, minlength=codebook_size)
    global_unique = int(np.sum(global_counts > 0))
    global_entropy = _shannon_entropy(global_counts)
    global_max_ent = float(np.log2(codebook_size))

    overall.update({
        'global_active_codes': global_unique,
        'global_utilization': global_unique / codebook_size,
        'global_entropy': global_entropy,
        'global_norm_entropy': global_entropy / global_max_ent,
    })

    return overall, per_level


def _fmt_metrics(overall, per_level, K):
    lines = []
    lines.append(
        f"  Overall   unique_tokens={overall['unique_tokens']}  "
        f"collision={overall['collision_rate']:.6f}  "
        f"global_active={overall['global_active_codes']}/{K}  "
        f"global_H_norm={overall['global_norm_entropy']:.4f}"
    )
    for lvl, m in enumerate(per_level):
        lines.append(
            f"  Level {lvl}   active={m['active_codes']}/{K}  "
            f"util={m['utilization']:.4f}  "
            f"H={m['entropy']:.4f}  H_norm={m['norm_entropy']:.4f}"
        )
    return lines


# ─────────────────────────────────────────────────────────────────────────────
#  Diversity Cluster Cache
# ─────────────────────────────────────────────────────────────────────────────

class DiversityClusterCache:
    """
    缓存每层码本的簇分配，避免每步都做 K-Means（节省计算）。
    每隔 N 个 epoch 更新一次。
    """

    def __init__(self, model, K=8, update_every=5, device='cuda:0'):
        self.model = model
        self.K = K
        self.update_every = update_every
        self.device = torch.device(device)
        self.last_update_epoch = -1
        # {level: (N,) int64 簇标签}
        self.cluster_labels = {}

    def update(self, epoch):
        if epoch - self.last_update_epoch < self.update_every:
            return
        self.last_update_epoch = epoch

        print(f"  [DiversityCache] Updating clusters (epoch {epoch}) ...")
        t0 = time.time()

        for level in range(self.model.num_levels):
            cb_weight = self.model.codebooks[level].weight.detach()
            labels = _constrained_kmeans_cpu(cb_weight.cpu().float(), self.K)
            self.cluster_labels[level] = labels.to(self.device)

        print(f"  [DiversityCache] Done in {time.time() - t0:.1f}s")

    def get_labels(self, level):
        return self.cluster_labels.get(level, None)


def _constrained_kmeans_cpu(embeddings: torch.Tensor, K: int, max_iter: int = 20):
    """
    CPU 上的 K-Means，聚类码字 embeddings。
    返回 (K_or_N,) 簇标签。K ≤ 码本大小。
    """
    from sklearn.cluster import KMeans
    K_actual = min(K, len(embeddings))
    kmeans = KMeans(n_clusters=K_actual, n_init='auto',
                    max_iter=max_iter, random_state=42)
    return torch.from_numpy(kmeans.fit_predict(embeddings.numpy())).long()


# ─────────────────────────────────────────────────────────────────────────────
#  LETTER Trainer
# ─────────────────────────────────────────────────────────────────────────────

class LETTERTrainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        args,
        cf_embeddings=None,   # (N_items, cf_dim) tensor，或 None
        logger=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.logger = logger or logging.getLogger()
        self.device = torch.device(args.device)
        self.model.to(self.device)

        # ── CF Embeddings ────────────────────────────────────────────────
        if cf_embeddings is not None:
            cf_embs_device = cf_embeddings.to(self.device)
            self.model.register_buffer(
                '_cf_embeddings_buffer',
                cf_embs_device,
                persistent=False
            )
            self.model._cf_embeddings = None  # 使用 buffer
            self.logger.info(
                f"CF embeddings registered as buffer: {cf_embs_device.shape}"
            )
        else:
            self.logger.info("CF embeddings not provided — L_CF will be disabled")

        # ── Optimizer & Scheduler ─────────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )

        # ── Diversity Cluster Cache ──────────────────────────────────────
        if args.beta > 0:
            self.div_cache = DiversityClusterCache(
                model, K=args.div_cluster_k,
                update_every=args.div_update_every,
                device=args.device,
            )
        else:
            self.div_cache = None

        # ── Checkpoint ────────────────────────────────────────────────────
        self.ckpt_dir = os.path.join(args.ckpt_dir, get_local_time())
        ensure_dir(self.ckpt_dir)
        self.logger.info(f"Checkpoint dir: {self.ckpt_dir}")

        self.best_recon_loss = float('inf')
        self.best_collision_rate = float('inf')
        self.best_global_entropy = -1.0
        self.global_step = 0

    # ─────────────────────────────────────────────────────────────────────
    def train_step(self, x, cf_x=None):
        self.model.train()
        self.optimizer.zero_grad()

        alpha = getattr(self.args, 'alpha', 0.0)
        beta = getattr(self.args, 'beta', 0.0)

        recon_x, z_q, codes, losses = self.model(x, cf_x, alpha=alpha, beta=beta)
        losses['total'].backward()
        self.optimizer.step()

        return {k: v.item() if torch.is_tensor(v) else v
                for k, v in losses.items()}

    # ─────────────────────────────────────────────────────────────────────
    def train_epoch(self, epoch):
        self.model.train()
        total_metrics = {
            'total': 0.0, 'sem': 0.0, 'rq_vae': 0.0,
            'recon': 0.0, 'cf': 0.0, 'div': 0.0
        }
        n_batches = len(self.train_loader)

        # 更新 diversity cluster cache
        if self.div_cache is not None:
            self.div_cache.update(epoch)

        pbar = tqdm(self.train_loader, ncols=130, desc=f"Epoch {epoch}")
        for x, cf_x_or_idx in pbar:
            x = x.to(self.device, non_blocking=True)

            # DualEmbDataset 返回 (sem, cf, idx)，EmbDataset 返回 (emb, idx)
            if isinstance(cf_x_or_idx, tuple):
                cf_x = cf_x_or_idx[1].to(self.device, non_blocking=True)
            else:
                cf_x = None

            metrics = self.train_step(x, cf_x)

            for k in total_metrics:
                total_metrics[k] += metrics.get(k, 0.0)
            self.global_step += 1

            alpha = getattr(self.args, 'alpha', 0.0)
            beta = getattr(self.args, 'beta', 0.0)
            pbar.set_postfix_str(
                f"sem={metrics.get('sem', 0):.4f} "
                f"rq={metrics.get('rq_vae', 0):.4f} "
                f"recon={metrics.get('recon', 0):.4f} "
                f"cf={metrics.get('cf', 0):.4f} "
                f"div={metrics.get('div', 0):.4f} "
                f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
            )

        avg = {k: v / n_batches for k, v in total_metrics.items()}
        return avg

    # ─────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def eval(self):
        """评估码本质量（分层 + 整体）。"""
        self.model.eval()
        all_codes = []

        for x, _ in tqdm(self.val_loader, ncols=80, desc="Eval codes"):
            x = x.to(self.device)
            codes = self.model.get_codes(x)
            all_codes.append(codes.cpu())

        all_codes = torch.cat(all_codes, dim=0)
        overall, per_level = compute_codebook_metrics(
            all_codes, codebook_size=self.model.codebook_size
        )
        return overall, per_level

    # ─────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def compute_recon_loss(self):
        self.model.eval()
        total = 0.0
        for x, _ in tqdm(self.val_loader, ncols=80, desc="Eval recon"):
            x = x.to(self.device)
            recon_x, _, _, losses = self.model(
                x, alpha=0.0, beta=0.0
            )
            total += losses.get('recon', torch.tensor(0.0)).item()
        return total / len(self.val_loader)

    # ─────────────────────────────────────────────────────────────────────
    def fit(self, epochs):
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            avg = self.train_epoch(epoch)
            self.scheduler.step()
            elapsed = time.time() - t0

            self.logger.info(
                f"Epoch {epoch:3d} | {elapsed:5.1f}s | "
                f"total={avg['total']:.4f} "
                f"sem={avg['sem']:.4f} "
                f"rq={avg['rq_vae']:.4f} "
                f"recon={avg['recon']:.4f} "
                f"cf={avg['cf']:.4f} "
                f"div={avg['div']:.4f}"
            )

            if epoch % self.args.eval_every == 0:
                overall, per_level = self.eval()
                val_recon = self.compute_recon_loss()

                for line in _fmt_metrics(overall, per_level, self.model.codebook_size):
                    self.logger.info(line)

                self.logger.info(
                    f"  Val  recon={val_recon:.4f} | "
                    f"collision={overall['collision_rate']:.6f} "
                    f"unique_tokens={overall['unique_tokens']}"
                )

                # 保存最优 checkpoint
                if val_recon < self.best_recon_loss:
                    self.best_recon_loss = val_recon
                    self._save_ckpt(epoch, 'best_recon')

                if overall['collision_rate'] < self.best_collision_rate:
                    self.best_collision_rate = overall['collision_rate']
                    self._save_ckpt(epoch, 'best_collision')

                if overall['global_norm_entropy'] > self.best_global_entropy:
                    self.best_global_entropy = overall['global_norm_entropy']
                    self._save_ckpt(epoch, 'best_entropy')

            if epoch % self.args.save_every == 0:
                self._save_ckpt(epoch, f'epoch{epoch}')

        self.logger.info(
            f"\nTraining done."
            f"  Best recon loss:    {self.best_recon_loss:.6f}\n"
            f"  Best collision rate:{self.best_collision_rate:.6f}\n"
            f"  Best global norm H: {self.best_global_entropy:.6f}"
        )

    # ─────────────────────────────────────────────────────────────────────
    def _save_ckpt(self, epoch, tag):
        path = os.path.join(self.ckpt_dir, f"{tag}_e{epoch}.pth")
        torch.save(
            {
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'args': vars(self.args),
                'best_recon_loss': self.best_recon_loss,
                'best_collision_rate': self.best_collision_rate,
                'best_global_entropy': self.best_global_entropy,
            },
            path,
            pickle_protocol=4,
        )
        self.logger.info(f"  [Saved] {path}")

    def load_ckpt(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt['state_dict'], strict=True)
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])
        self.best_recon_loss = ckpt.get('best_recon_loss', float('inf'))
        self.best_collision_rate = ckpt.get('best_collision_rate', float('inf'))
        self.best_global_entropy = ckpt.get('best_global_entropy', -1.0)
        self.logger.info(f"Loaded: {ckpt_path} (epoch={ckpt['epoch']})")
