#!/usr/bin/env python3
"""
LETTER-RQVAE 训练入口

用法:
    # 完整训练（Semantic + Collaborative + Diversity）
    python main.py \
        --data ../data/embeddings.npz \
        --cf_embeddings ../data/cf_embeddings.npz \
        --epochs 200 \
        --batch_size 1024 \
        --device cuda:0 \
        --alpha 0.1 \
        --beta 0.01 \
        --eval_every 5 \
        --save_every 10

    # 纯语义（不用 CF embeddings）
    python main.py \
        --data ../data/embeddings.npz \
        --epochs 200 \
        --batch_size 1024 \
        --device cuda:0 \
        --alpha 0.0 \
        --beta 0.0

    # 断点续训
    python main.py \
        --data ../data/embeddings.npz \
        --cf_embeddings ../data/cf_embeddings.npz \
        --epochs 400 \
        --resume ./checkpoints/.../best_entropy_e100.pth \
        --batch_size 1024 \
        --device cuda:0
"""
import argparse
import random
import os
import logging
import torch
import numpy as np

from model import LETTER
from dataset import DualEmbDataset, EmbDataset
from trainer import LETTERTrainer
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="LETTER-RQVAE Training")

    # ── 数据 ────────────────────────────────────────────────────────────
    parser.add_argument('--data', type=str,
        default='../data/embeddings.npz',
        help='Semantic embeddings .npz 路径')
    parser.add_argument('--cf_embeddings', type=str, default=None,
        help='CF embeddings .npz 路径（来自 cf_trainer.py）')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=8)

    # ── 训练 ────────────────────────────────────────────────────────────
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=50)

    # ── LETTER 模型参数 ─────────────────────────────────────────────────
    parser.add_argument('--input_dim', type=int, default=128,
        help='Semantic embedding 维度（自动检测）')
    parser.add_argument('--code_dim', type=int, default=32,
        help='量化隐向量维度')
    parser.add_argument('--num_levels', type=int, default=4,
        help='RQ 量化层数（identifier 长度）')
    parser.add_argument('--codebook_size', type=int, default=256,
        help='每层码本大小')
    parser.add_argument('--commitment_mu', type=float, default=0.25,
        help='论文 Eq.(2) 中 μ，控制 encoder commitment 强度')

    # ── 重构开关 ────────────────────────────────────────────────────────
    parser.add_argument('--use_recon', type=bool, default=True,
        help='是否包含 decoder（启用 L_Recon）')

    # ── LETTER 损失参数（论文 Eq.(5)）───────────────────────────────────
    parser.add_argument('--alpha', type=float, default=0.1,
        help='Collaborative Regularization 权重')
    parser.add_argument('--beta', type=float, default=0.01,
        help='Diversity Regularization 权重')
    parser.add_argument('--cf_dim', type=int, default=64,
        help='CF embedding 维度（来自 SASRec 等 CF 模型）')
    parser.add_argument('--cf_proj_dim', type=int, default=32,
        help='CF 映射层输出维度（建议 ≤ cf_dim）')

    # ── Diversity Regularization ────────────────────────────────────────
    parser.add_argument('--div_cluster_k', type=int, default=8,
        help='Diversity loss 每层簇数 K（论文建议 ≤ 16）')
    parser.add_argument('--div_update_every', type=int, default=5,
        help='簇分配更新频率（每 N 个 epoch）')

    # ── 断点续训 ────────────────────────────────────────────────────────
    parser.add_argument('--resume', type=str, default=None,
        help='.pth checkpoint 路径')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints')

    return parser.parse_args()


def seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    seed(42)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )
    logger = logging.getLogger()

    # ── 数据 ────────────────────────────────────────────────────────────
    logger.info(f"Loading data: {args.data}")
    if args.cf_embeddings:
        logger.info(f"Loading CF embeddings: {args.cf_embeddings}")
        dataset = DualEmbDataset(
            npz_path=args.data,
            cf_npz_path=args.cf_embeddings,
            cf_dim=args.cf_dim,
        )
        has_cf = True
    else:
        logger.info("No CF embeddings — running LETTER without Collaborative Regularization")
        dataset = EmbDataset(npz_path=args.data)
        has_cf = False

    actual_dim = dataset.dim
    if args.input_dim != actual_dim:
        logger.info(f"input_dim {args.input_dim} -> auto corrected to {actual_dim}")
        args.input_dim = actual_dim

    n = len(dataset)
    train_size = int(n * 0.95)
    val_size = n - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    logger.info(f"Dataset: {n} items, train={train_size}, val={val_size}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # ── 加载 CF embeddings 为 tensor ────────────────────────────────────
    cf_embeddings = None
    if has_cf:
        cf_npz = np.load(args.cf_embeddings)
        cf_embeddings = torch.from_numpy(
            cf_npz['cf_embeddings'].astype(np.float32)
        )
        logger.info(f"CF embeddings tensor: {cf_embeddings.shape}")

    # ── 模型 ────────────────────────────────────────────────────────────
    model = LETTER(
        input_dim=args.input_dim,
        code_dim=args.code_dim,
        num_levels=args.num_levels,
        codebook_size=args.codebook_size,
        commitment_mu=args.commitment_mu,
        use_recon=args.use_recon,
        cf_dim=args.cf_dim,
        cf_proj_dim=args.cf_proj_dim,
        div_cluster_k=args.div_cluster_k,
    )
    n_params = sum(p.numel() for p in model.parameters())
    cf_active = has_cf and args.alpha > 0
    div_active = args.beta > 0

    logger.info("=" * 60)
    logger.info("LETTER-RQVAE")
    logger.info(f"  input_dim={args.input_dim}, code_dim={args.code_dim}")
    logger.info(f"  num_levels={args.num_levels}, codebook_size={args.codebook_size}")
    logger.info(f"  commitment_mu={args.commitment_mu}, use_recon={args.use_recon}")
    logger.info(f"  α(CF)={args.alpha} {'[ACTIVE]' if cf_active else '[DISABLED]'}")
    logger.info(f"  β(Div)={args.beta}  {'[ACTIVE]' if div_active else '[DISABLED]'}")
    logger.info(f"  cf_dim={args.cf_dim}, cf_proj_dim={args.cf_proj_dim}")
    logger.info(f"  div_cluster_k={args.div_cluster_k}, div_update_every={args.div_update_every}")
    logger.info(f"  Total params: {n_params:,}")
    logger.info("=" * 60)

    loss_desc = "L_LETTER = L_Sem"
    if cf_active:
        loss_desc += f" + {args.alpha}·L_CF"
    if div_active:
        loss_desc += f" + {args.beta}·L_Div"
    logger.info(f"  Loss: {loss_desc}")

    # ── 训练器 ──────────────────────────────────────────────────────────
    trainer = LETTERTrainer(
        model, train_loader, val_loader, args,
        cf_embeddings=cf_embeddings, logger=logger,
    )

    if args.resume:
        trainer.load_ckpt(args.resume)

    trainer.fit(args.epochs)
    logger.info("Done.")


if __name__ == "__main__":
    main()
