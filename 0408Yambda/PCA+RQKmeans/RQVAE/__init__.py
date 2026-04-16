"""
RQVAE — 自包含残差量化变分自编码器。

Stage 02 (build_item_sid.py) 通过 from RQVAE.encoder import load_rqvae_encoder 加载。
Checkpoint 来源：
  /root/autodl-tmp/0330Yambda/pure_rqvae/checkpoints/Mar-30-2026_23-12-08/best_entropy_e20.pth
"""
from .model import RQVAE
from .encoder import RQVAEEncoder, load_rqvae_encoder

__all__ = ["RQVAE", "RQVAEEncoder", "load_rqvae_encoder"]
