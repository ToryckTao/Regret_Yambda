"""
RQVAE 模型（pure_rqvae 独立版本，不依赖外部项目）

三个优化目标：
  1. cb_loss      — 梯度流向 codebook，让码字靠近 encoder 输出
  2. commit_loss  — 梯度流向 encoder，让 z_e 靠近码字
  3. recon_loss   — 梯度流向 encoder + decoder + codebook，保持邻居结构（use_recon=True 时）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RQVAE(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        code_dim: int = 32,
        num_levels: int = 4,
        codebook_size: int = 256,
        commitment_beta: float = 0.25,
        use_recon: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.code_dim = code_dim
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.commitment_beta = commitment_beta
        self.use_recon = use_recon

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, code_dim),
        )

        self.codebooks = nn.ModuleList([
            nn.Embedding(codebook_size, code_dim) for _ in range(num_levels)
        ])
        for cb in self.codebooks:
            nn.init.xavier_normal_(cb.weight)

        if use_recon:
            self.decoder = nn.Sequential(
                nn.Linear(code_dim, 256),
                nn.ReLU(),
                nn.Linear(256, input_dim),
            )

    def _quantize_layer(self, residual, cb_weight):
        dist = (
            residual.pow(2).sum(1, keepdim=True)
            + cb_weight.pow(2).sum(1)
            - 2 * torch.matmul(residual, cb_weight.t())
        )
        indices = torch.argmin(dist, dim=-1)
        quantized = F.embedding(indices, cb_weight)

        cb_loss = F.mse_loss(quantized, residual.detach())
        commit_loss = F.mse_loss(quantized.detach(), residual)

        return quantized, indices, cb_loss, commit_loss

    def forward(self, x):
        z_e = self.encoder(x)
        residual = z_e
        z_q = torch.zeros_like(z_e)
        codes = []
        cb_total = 0.0
        commit_total = 0.0

        for level in range(self.num_levels):
            cb_weight = self.codebooks[level].weight
            quantized, indices, cb_loss, commit_loss = self._quantize_layer(
                residual, cb_weight
            )
            codes.append(indices)
            cb_total += cb_loss
            commit_total += commit_loss

            quantized = residual + (quantized - residual).detach()
            z_q = z_q + quantized
            residual = residual - quantized.detach()

        losses = {
            "cb": cb_total,
            "commit": self.commitment_beta * commit_total,
            "recon": 0.0,
        }
        recon_x = None
        if self.use_recon:
            recon_x = self.decoder(z_q)
            losses["recon"] = F.mse_loss(recon_x, x)

        losses["total"] = losses["cb"] + losses["commit"] + losses["recon"]
        codes = torch.stack(codes, dim=1)

        return recon_x, z_q, codes, losses

    @torch.no_grad()
    def get_codes(self, x):
        z_e = self.encoder(x)
        residual = z_e
        codes = []

        for level in range(self.num_levels):
            cb_weight = self.codebooks[level].weight
            quantized, indices, _, _ = self._quantize_layer(residual, cb_weight)
            codes.append(indices)
            residual = residual - quantized.detach()

        return torch.stack(codes, dim=1)

    @torch.no_grad()
    def decode_from_codes(self, codes):
        assert self.use_recon, "use_recon=False 时无法 decode"
        z_q = torch.zeros(codes.size(0), self.code_dim, device=codes.device)
        for level in range(self.num_levels):
            z_q = z_q + self.codebooks[level](codes[:, level])
        return self.decoder(z_q)
