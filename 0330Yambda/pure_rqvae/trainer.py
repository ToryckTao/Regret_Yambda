"""
RQVAE 训练器

Loss 组成（三项加起来）：
    loss_total = cb_loss + commit_loss + recon_loss
               = Σ_l  ‖quantized_l − residual_l.detach()‖²        ← 码本损失
                 + β * Σ_l  ‖quantized_l.detach() − residual_l‖²  ← 承诺损失
                 + MSE(recon_x, x)                                 ← 重构损失（use_recon=True 时）

use_recon=False 时不加重构损失，训练更快，但 codes 的语义质量无保证。
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
#  码本质量评估工具（模块级函数，可直接 import 测试）
# ─────────────────────────────────────────────────────────────────────────────

def _shannon_entropy(counts: np.ndarray) -> float:
    """
    给定各符号出现次数，返回香农熵（以 bit 为单位）。
    counts: shape (K,)，各码字被选中的次数。
    """
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def compute_codebook_metrics(all_codes: torch.Tensor, codebook_size: int):
    """
    计算码本质量指标，按层 + 整体输出。

    all_codes: (N, num_levels)  验证集所有样本的离散 Token
    codebook_size: 每层码本大小 K=256

    返回:
        overall: dict，跨所有层的整体指标
        per_level: list[dict]，每层各指标，长度 = num_levels
    """
    num_levels = all_codes.size(1)
    N = all_codes.size(0)

    # ── 全局（4-tuple 层面）─────────────────────────────────────────────
    code_tuples = [str(c.tolist()) for c in all_codes]
    unique_tokens = len(set(code_tuples))
    collision_rate = (N - unique_tokens) / N if N > 0 else 0.0
    overall = {
        'unique_tokens': unique_tokens,
        'collision_rate': collision_rate,
    }

    # ── 分层统计 ─────────────────────────────────────────────────────────
    per_level = []
    for lvl in range(num_levels):
        lvl_codes = all_codes[:, lvl].numpy()              # (N,)

        # ① Active codes：被使用过的不同码字数量
        unique_lvl = len(set(lvl_codes))
        active_codes = unique_lvl

        # ② Utilization rate：被使用的码字数 / 总码字数
        utilization = active_codes / codebook_size

        # ③ Shannon Entropy & Normalized Entropy
        counts = np.bincount(lvl_codes, minlength=codebook_size)
        entropy = _shannon_entropy(counts)
        max_entropy = float(np.log2(codebook_size))
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        per_level.append({
            'active_codes':  active_codes,
            'utilization':   utilization,
            'entropy':       entropy,
            'norm_entropy':  norm_entropy,
        })

    # ── 全局：把所有层的码字拼在一起统计 ────────────────────────────────
    # 反映"整个码本体系"被使用的均衡程度
    all_flat = all_codes.view(-1).numpy()                  # (N * num_levels,)
    global_counts = np.bincount(all_flat, minlength=codebook_size)
    global_unique = int(np.sum(global_counts > 0))
    global_entropy = _shannon_entropy(global_counts)
    global_norm_ent = global_entropy / max_entropy if max_entropy > 0 else 0.0
    overall.update({
        'global_active_codes':  global_unique,
        'global_utilization':   global_unique / codebook_size,
        'global_entropy':       global_entropy,
        'global_norm_entropy':  global_norm_ent,
    })

    return overall, per_level


def _fmt_metrics(overall, per_level, K):
    """把指标格式化成一行行可读字符串，用于 logger.info。"""
    lines = []
    # 整体
    lines.append(
        f"  Overall   unique_tokens={overall['unique_tokens']}  "
        f"collision={overall['collision_rate']:.6f}  "
        f"global_active={overall['global_active_codes']}/{K}  "
        f"global_util={overall['global_utilization']:.4f}  "
        f"global_H={overall['global_entropy']:.4f}  "
        f"global_H_norm={overall['global_norm_entropy']:.4f}"
    )
    # 分层
    for lvl, m in enumerate(per_level):
        lines.append(
            f"  Level {lvl}   active={m['active_codes']}/{K}  "
            f"util={m['utilization']:.4f}  "
            f"H={m['entropy']:.4f}  "
            f"H_norm={m['norm_entropy']:.4f}"
        )
    return lines


# ─────────────────────────────────────────────────────────────────────────────
#  Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:

    def __init__(self, model, train_loader, val_loader, args, logger=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.logger = logger or logging.getLogger()
        self.device = torch.device(args.device)
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )

        self.ckpt_dir = os.path.join(args.ckpt_dir, get_local_time())
        ensure_dir(self.ckpt_dir)
        self.logger.info(f"Checkpoint dir: {self.ckpt_dir}")

        self.best_recon_loss = float('inf')
        self.best_collision_rate = float('inf')
        self.best_global_entropy = -1.0      # 熵越大越好（越均匀）
        self.global_step = 0

    def train_step(self, x):
        self.model.train()
        self.optimizer.zero_grad()

        recon_x, z_q, codes, losses = self.model(x)
        losses['total'].backward()
        self.optimizer.step()

        return {k: v.item() for k, v in losses.items()}

    def train_epoch(self, epoch):
        self.model.train()
        total_metrics = {'total': 0.0, 'cb': 0.0, 'commit': 0.0, 'recon': 0.0}
        n_batches = len(self.train_loader)

        pbar = tqdm(self.train_loader, ncols=130, desc=f"Epoch {epoch}")
        for x, _ in pbar:
            x = x.to(self.device, non_blocking=True)
            metrics = self.train_step(x)
            for k in total_metrics:
                total_metrics[k] += metrics[k]
            self.global_step += 1

            pbar.set_postfix_str(
                f"cb={metrics['cb']:.4f} "
                f"commit={metrics['commit']:.4f} "
                f"recon={metrics['recon']:.4f} "
                f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
            )

        avg = {k: v / n_batches for k, v in total_metrics.items()}
        return avg

    @torch.no_grad()
    def eval(self):
        """
        计算码本质量指标（分层 + 整体）：
          collision_rate  : 4-tuple 层面，相同序列比例
          active_codes    : 每层被使用过的不同码字数量
          utilization     : active_codes / codebook_size
          entropy         : 每层/全局码字分布的香农熵
          norm_entropy    : 归一化熵（越接近 1 越均匀，越接近 0 越倾斜）
        """
        self.model.eval()
        all_codes = []

        for x, _ in tqdm(self.val_loader, ncols=80, desc="Eval codes"):
            x = x.to(self.device)
            codes = self.model.get_codes(x)
            all_codes.append(codes.cpu())

        all_codes = torch.cat(all_codes, dim=0)          # (N, num_levels)
        overall, per_level = compute_codebook_metrics(
            all_codes, codebook_size=self.model.codebook_size
        )
        return overall, per_level

    @torch.no_grad()
    def compute_recon_loss(self):
        self.model.eval()
        total = 0.0
        for x, _ in tqdm(self.val_loader, ncols=80, desc="Eval recon"):
            x = x.to(self.device)
            recon_x, _, _, losses = self.model(x)
            total += losses['recon'].item()
        return total / len(self.val_loader)

    def fit(self, epochs):
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            avg = self.train_epoch(epoch)
            self.scheduler.step()
            elapsed = time.time() - t0

            self.logger.info(
                f"Epoch {epoch:3d} | {elapsed:5.1f}s | "
                f"total={avg['total']:.4f} "
                f"cb={avg['cb']:.4f} "
                f"commit={avg['commit']:.4f} "
                f"recon={avg['recon']:.4f}"
            )

            if epoch % self.args.eval_every == 0:
                overall, per_level = self.eval()
                val_recon = self.compute_recon_loss()

                # 打印码本质量指标
                for line in _fmt_metrics(overall, per_level, self.model.codebook_size):
                    self.logger.info(line)

                self.logger.info(
                    f"  Val  recon={val_recon:.4f} | "
                    f"collision={overall['collision_rate']:.6f} "
                    f"unique_tokens={overall['unique_tokens']}"
                )

                # 保存最优
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
