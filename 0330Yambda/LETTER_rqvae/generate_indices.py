#!/usr/bin/env python3
"""
从训练好的 LETTER ckpt 导出所有 item 的离散 Token 序列，保存为 .json。

用法:
    python generate_indices.py \
        --ckpt ./checkpoints/.../best_entropy_e100.pth \
        --data ../data/embeddings.npz \
        --output ../data/item_tokens_letter.json
"""
import argparse
import json
import time

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from model import LETTER
from dataset import EmbDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True,
                        help='.pth checkpoint path')
    parser.add_argument('--data', type=str,
                        default='../data/embeddings.npz',
                        help='embeddings.npz path')
    parser.add_argument('--output', type=str,
                        default='../data/item_tokens_letter.json',
                        help='output .json path')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--num_workers', type=int, default=8)
    return parser.parse_args()


def check_collision(all_indices_str):
    unique = len(set(all_indices_str))
    total = len(all_indices_str)
    return unique == total, (total - unique) / total


def main():
    args = parse_args()
    t0 = time.time()

    # 加载数据
    dataset = EmbDataset(npz_path=args.data)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 加载模型
    ckpt = torch.load(args.ckpt, map_location='cpu')
    state_dict = ckpt.get('state_dict', ckpt)
    args_c = ckpt.get('args', {})
    print(f"[generate_indices] Loaded ckpt from {args.ckpt}")
    print(f"  state_dict keys sample: {list(state_dict.keys())[:5]}")

    device = torch.device(args.device)

    model = LETTER(
        input_dim=dataset.dim,
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
    model.to(device)
    model.eval()

    # Token 前缀（供下游大模型使用）
    # 与 pure_rqvae 一致，支持 L=4
    prefix = ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>"]

    all_tokens = []
    all_strs = []

    with torch.no_grad():
        for x, indices in tqdm(loader, ncols=80, desc="Encoding"):
            x = x.to(device)
            codes = model.get_codes(x)                # (B, num_levels)
            codes_np = codes.cpu().numpy()             # (B, num_levels)

            for code in codes_np:
                token_seq = [p.format(int(idx)) for p, idx in zip(prefix, code)]
                all_tokens.append(token_seq)
                all_strs.append(str(token_seq))

    # 碰撞率统计
    unique_count, collision_rate = check_collision(all_strs)
    elapsed = time.time() - t0

    print(f"\n[generate_indices] Done in {elapsed:.1f}s")
    print(f"  Total items:     {len(all_tokens)}")
    print(f"  Unique tokens:   {unique_count}")
    print(f"  Collision rate:   {collision_rate:.6f}")

    # 保存: index → token_seq
    tokens_dict = {int(idx): tokens for idx, tokens in enumerate(all_tokens)}
    with open(args.output, 'w') as f:
        json.dump(tokens_dict, f, indent=2)
    print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    main()
