#!/usr/bin/env python3
import sys
sys.path.insert(0, "/root/autodl-tmp/0408Yambda")
from RQVAE.encoder import load_rqvae_encoder
import numpy as np

ckpt = "/root/autodl-tmp/0330Yambda/pure_rqvae/checkpoints/Mar-30-2026_23-12-08/best_entropy_e20.pth"
enc = load_rqvae_encoder(ckpt)
emb = np.random.randn(128, 128).astype(np.float32)
codes = enc.encode_numpy(emb)
print(f"OK shape={codes.shape} dtype={codes.dtype}")
assert codes.shape == (128, 4), f"bad shape {codes.shape}"
print("assert OK")