#!/usr/bin/env python3
"""
RQVAE 训练入口

用法:
    # 重构模式（要生成音频）：加重构损失
    python main.py --use_recon True --epochs 200

    # 纯量化模式（只用离散 codes 做 token）：不加重构损失
    python main.py --use_recon False --epochs 200
"""
import argparse
import random
import os
import logging

import torch
import numpy as np

from model import RQVAE
from dataset import EmbDataset
from trainer import Trainer
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="RQVAE Training")

    # 数据
    parser.add_argument('--data', type=str,
        default='../data/embeddings.npz',
        help='预处理后的 .npz 文件路径')
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--num_workers', type=int, default=8)

    # 训练
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--save_every', type=int, default=50)

    # 模型
    # input_dim 程序自动从数据中检测，此处写 128 仅作默认值
    parser.add_argument('--input_dim', type=int, default=128)
    parser.add_argument('--code_dim', type=int, default=32)
    parser.add_argument('--num_levels', type=int, default=4)
    parser.add_argument('--codebook_size', type=int, default=256)
    parser.add_argument('--commitment_beta', type=float, default=0.25)

    # 重构开关
    # True  → 加重构损失 loss = cb + commit + recon（训练慢，但 codes 语义好）
    # False → 不加重构损失 loss = cb + commit（训练快，但 codes 语义无保证）
    parser.add_argument('--use_recon', type=bool, default=True)

    # 断点续训
    parser.add_argument('--resume', type=str, default=None)

    # 输出
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
    logger.info(f"Loading: {args.data}")
    dataset = EmbDataset(npz_path=args.data)

    # 自动检测 embedding 维度
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
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ── 模型 ────────────────────────────────────────────────────────────
    model = RQVAE(
        input_dim=args.input_dim,
        code_dim=args.code_dim,
        num_levels=args.num_levels,
        codebook_size=args.codebook_size,
        commitment_beta=args.commitment_beta,
        use_recon=args.use_recon,
    )
    n_params = sum(p.numel() for p in model.parameters())

    logger.info("=" * 60)
    logger.info("RQVAE")
    logger.info(f"  input_dim={args.input_dim}, code_dim={args.code_dim}")
    logger.info(f"  num_levels={args.num_levels}, codebook_size={args.codebook_size}")
    logger.info(f"  commitment_beta={args.commitment_beta}")
    logger.info(f"  use_recon={args.use_recon}")
    if args.use_recon:
        logger.info("  Loss = cb_loss + commit_loss + recon_loss  （含重建）")
    else:
        logger.info("  Loss = cb_loss + commit_loss            （纯量化）")
    logger.info(f"  Total params: {n_params:,}")
    logger.info("=" * 60)

    trainer = Trainer(model, train_loader, val_loader, args, logger)

    if args.resume:
        trainer.load_ckpt(args.resume)

    trainer.fit(args.epochs)
    logger.info("Done.")


if __name__ == "__main__":
    main()
