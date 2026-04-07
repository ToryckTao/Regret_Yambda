"""
RQVAE 模型

三个优化目标（对应三个损失）：
  1. cb_loss   （码本损失）  → 梯度流向 codebook 权重
     ‖z_e − e_i.detach()‖²
     目标：让码字 e_i 学会靠近 encoder 输出的残差

  2. commit_loss（承诺损失） → 梯度流向 encoder 权重
     ‖z_e.detach() − e_i‖²
     目标：让 encoder 输出 z_e 学会靠近被选中的码字

  3. recon_loss（重构损失） → 梯度流向 encoder + decoder + codebook
     ‖decoder(z_q) − x‖²
     目标：让量化结果 z_q 包含足够信息来重建原始 embedding
     【如果不用 decoder 生成，设为 use_recon=False 即可去掉此项】
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RQVAE(nn.Module):

    def __init__(
        self,
        input_dim: int = 128,       # yambda embedding 维度
        code_dim: int = 32,         # encoder 输出维度，也是码字维度
        num_levels: int = 4,        # 残差量化层数
        codebook_size: int = 256,   # 每层码本大小 K
        commitment_beta: float = 0.25,  # 承诺损失权重
        use_recon: bool = True,     # True=加重构损失（训练+生成）
                                     # False=只加码本+承诺损失（纯量化）
    ):
        super().__init__()
        self.input_dim = input_dim
        self.code_dim = code_dim
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.commitment_beta = commitment_beta
        self.use_recon = use_recon

        # Encoder: (B, input_dim) → (B, code_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, code_dim),
        )

        # 4 个独立码本, 每个大小 (K, code_dim)
        self.codebooks = nn.ModuleList([
            nn.Embedding(codebook_size, code_dim) for _ in range(num_levels)
        ])
        for cb in self.codebooks:
            nn.init.xavier_normal_(cb.weight)

        # Decoder: (B, code_dim) → (B, input_dim)
        if use_recon:
            self.decoder = nn.Sequential(
                nn.Linear(code_dim, 256),
                nn.ReLU(),
                nn.Linear(256, input_dim),
            )

    # ────────────────────────────────────────────────────────────────────
    def _quantize_layer(self, residual, cb_weight):
        """
        单层量化：找最近邻码字

        Args:
            residual:   (B, code_dim)    当前层残差
            cb_weight:  (K, code_dim)    码本矩阵

        Returns:
            quantized:  (B, code_dim)    量化后的向量（来自码本）
            indices:    (B,)              被选中的码字下标
            cb_loss:    标量               码本损失
            commit_loss: 标量              承诺损失
        """
        # L2 距离展开: ‖r - e‖² = ‖r‖² + ‖e‖² - 2⟨r, e⟩
        # (B, 1) + (K,) - (B, K)  → (B, K)
        dist = (
            residual.pow(2).sum(1, keepdim=True)
            + cb_weight.pow(2).sum(1)
            - 2 * torch.matmul(residual, cb_weight.t())
        )
        indices = torch.argmin(dist, dim=-1)          # (B,)
        quantized = F.embedding(indices, cb_weight)   # (B, code_dim)

        # ① 码本损失: 梯度流向码本，让码字靠近残差（detach 残差）
        cb_loss = F.mse_loss(quantized, residual.detach())

        # ② 承诺损失: 梯度流向 encoder，让 z_e 靠近码字（detach 量化值）
        commit_loss = F.mse_loss(quantized.detach(), residual)

        return quantized, indices, cb_loss, commit_loss

    # ────────────────────────────────────────────────────────────────────
    def forward(self, x):
        """
        完整前向

        Returns:
            recon_x:    (B, input_dim)  重建结果（use_recon=True 时有效）
            z_q:        (B, code_dim)   量化后的隐向量
            codes:      (B, num_levels) 每层的离散下标
            losses:     dict            三项损失的标量字典
        """
        z_e = self.encoder(x)                   # (B, code_dim)

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

            # STE (Straight-Through Estimator)
            # forward 阶段用量化值；backward 阶段跳过 argmin 直通
            quantized = residual + (quantized - residual).detach()

            z_q = z_q + quantized
            # 残差 detach 更新，隔断跨层梯度传播
            residual = residual - quantized.detach()

        # ③ 重构损失: 梯度流向 encoder + decoder + 所有 codebook
        losses = {
            'cb': cb_total,                                   # 码本损失
            'commit': self.commitment_beta * commit_total,    # 承诺损失
            'recon': 0.0,
        }
        recon_x = None
        if self.use_recon:
            recon_x = self.decoder(z_q)                        # (B, input_dim)
            losses['recon'] = F.mse_loss(recon_x, x)           # 重构损失

        losses['total'] = losses['cb'] + losses['commit'] + losses['recon']
        codes = torch.stack(codes, dim=1)                     # (B, num_levels)

        return recon_x, z_q, codes, losses

    # ────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def get_codes(self, x):
        """纯推理：只返回离散 codes，不做重建"""
        z_e = self.encoder(x)
        residual = z_e
        codes = []

        for level in range(self.num_levels):
            cb_weight = self.codebooks[level].weight
            quantized, indices, _, _ = self._quantize_layer(residual, cb_weight)
            codes.append(indices)
            residual = residual - quantized.detach()

        return torch.stack(codes, dim=1)   # (B, num_levels)

    # ────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def decode_from_codes(self, codes):
        """给定 (B, num_levels) 离散下标，解码回 (B, input_dim)"""
        assert self.use_recon, "use_recon=False 时无法 decode"
        z_q = torch.zeros(codes.size(0), self.code_dim, device=codes.device)
        for level in range(self.num_levels):
            z_q = z_q + self.codebooks[level](codes[:, level])
        return self.decoder(z_q)
