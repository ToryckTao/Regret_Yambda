"""
RQVAE encoder 加载器。

从 .pth checkpoint 加载 RQVAE 模型，提取编码器 + 码本，返回轻量包装对象，
供 Stage 02 (build_item_sid.py) 直接调用。

Usage:
    from RQVAE.encoder import load_rqvae_encoder
    encoder = load_rqvae_encoder("/path/to/best_entropy_e20.pth")
    codes = encoder.encode(torch.randn(32, 128))   # (32, 4)
"""
import torch
import numpy as np
from pathlib import Path
from typing import Optional

from .model import RQVAE


class RQVAEEncoder:
    """
    Stage 02 使用的 RQVAE 编码器封装。

    Attributes:
        n_levels:       层数（L=4）
        codebook_size:  每层码本大小（K=256）
        code_dim:       量化向量维度（32）
        input_dim:      输入 embedding 维度（128）
        codebooks:      list[ndarray]，每层码本权重，shape=(codebook_size, code_dim)
        enc_w0:         encoder 第一层权重 shape=(256, input_dim)，供 Stage 02 推断 dim
        device:         计算设备
    """

    def __init__(self, model: RQVAE, device: Optional[torch.device] = None):
        self.model = model
        self.n_levels = model.num_levels
        self.codebook_size = model.codebook_size
        self.code_dim = model.code_dim
        self.input_dim = model.input_dim
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self._codebook_weights = [
            cb.weight.detach().cpu().numpy()
            for cb in model.codebooks
        ]

        enc_w0 = next(m for n, m in model.named_parameters() if "encoder.0.weight" in n)
        self.enc_w0 = enc_w0.detach().cpu()

    @property
    def codebooks(self) -> list:
        """返回 CPU numpy 码本列表，供 encode_sid_batch() 使用。"""
        return self._codebook_weights

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        批量编码。

        Args:
            x: (B, input_dim) tensor，float32

        Returns:
            codes: (B, n_levels) tensor，int64，每行是一个 item 的离散 SID
        """
        x = x.to(self.device)
        return self.model.get_codes(x)

    def encode_numpy(self, x: np.ndarray) -> np.ndarray:
        """
        NumPy 接口。

        Args:
            x: (B, input_dim) ndarray，float32

        Returns:
            codes: (B, n_levels) ndarray，int32
        """
        tensor = torch.from_numpy(x.astype(np.float32))
        codes = self.encode(tensor)
        return codes.cpu().numpy()

    def encode_from_parquet_batch(self, emb_list: list) -> np.ndarray:
        """
        从 parquet 读取的一批 embedding（list of arrays）直接编码。

        Args:
            emb_list: list[array]，每项 shape=(input_dim,)

        Returns:
            codes: (len(emb_list), n_levels) ndarray，int32
        """
        batch = np.stack(emb_list, axis=0).astype(np.float32)
        return self.encode_numpy(batch)


def load_rqvae_encoder(
    ckpt_path: str,
    device: Optional[str] = None,
) -> RQVAEEncoder:
    """
    从 checkpoint 加载 RQVAE，返回编码器封装对象。

    Args:
        ckpt_path: .pth 文件路径（best_entropy_e*.pth 等）
        device:    计算设备，默认 auto

    Returns:
        RQVAEEncoder 实例

    Raises:
        FileNotFoundError: checkpoint 不存在
        RuntimeError:     state_dict 格式不匹配
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    saved_args = ckpt.get("args", {})

    input_dim = saved_args.get("input_dim", 128)
    code_dim = saved_args.get("code_dim", 32)
    num_levels = saved_args.get("num_levels", 4)
    codebook_size = saved_args.get("codebook_size", 256)
    use_recon = saved_args.get("use_recon", True)

    model = RQVAE(
        input_dim=input_dim,
        code_dim=code_dim,
        num_levels=num_levels,
        codebook_size=codebook_size,
        use_recon=use_recon,
    )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load_rqvae_encoder] missing keys: {missing}")
    if unexpected:
        print(f"[load_rqvae_encoder] unexpected keys: {unexpected}")

    dev = torch.device(device) if device else None
    encoder = RQVAEEncoder(model, device=dev)
    print(
        f"[load_rqvae_encoder] loaded from {ckpt_path.name}  "
        f"(levels={num_levels}, code_dim={code_dim}, input_dim={input_dim})"
    )
    return encoder
