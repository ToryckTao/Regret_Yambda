"""
LETTER-RQVAE 模型

基于论文 "LETTER: LLM-based Generative Recommendation with
Hierarchical Semantic Tokenization" 实现。

三个损失组件（对应论文 Eq.(5)）：
  L_Sem  = L_Recon + L_RQVAE          → 层次语义
  L_CF   = in-batch NCE 对比损失        → 协同信号
  L_Div  = diversity NCE 对比损失       → 码字分配均匀性

最终: L_LETTER = L_Sem + α·L_CF + β·L_Div
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_


# ─────────────────────────────────────────────────────────────────────────────
#  Constrained K-Means（用于 Diversity Regularization）
# ─────────────────────────────────────────────────────────────────────────────

def constrained_kmeans(embeddings: torch.Tensor, K: int, max_iter: int = 10):
    """
    在 embeddings 上运行一次 K-Means（CPU 风格），返回每个样本的簇标签。

    Args:
        embeddings: (N, D) float32
        K:          簇数量（建议 <= 16）

    Returns:
        labels: (N,) int64，值域 [0, K-1]
    """
    embeddings_np = embeddings.detach().cpu().numpy()
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=min(K, len(embeddings_np)),
                    n_init="auto", max_iter=max_iter, random_state=42)
    return torch.from_numpy(kmeans.fit_predict(embeddings_np)).long().to(embeddings.device)


# ─────────────────────────────────────────────────────────────────────────────
#  CF Embedding Head（Collaborative Regularization 用）
# ─────────────────────────────────────────────────────────────────────────────

class CFEmbeddingHead(nn.Module):
    """
    将 RQ-VAE 量化向量 z_q 映射到 CF embedding 空间，供 L_CF 对比损失使用。

    论文 Eq.(3): L_CF = - 1/B · Σ log exp(<z_hat_i, h_i>) / Σ_j exp(<z_hat_i, h_j>)
    其中 h_i 是预训练 CF 模型给出的 CF embedding。
    """

    def __init__(self, code_dim: int, cf_dim: int):
        super().__init__()
        self.proj = nn.Linear(code_dim, cf_dim, bias=False)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(z_q), dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
#  LETTER 主模型
# ─────────────────────────────────────────────────────────────────────────────

class LETTER(nn.Module):
    """
    LETTER Tokenizer = RQ-VAE + Collaborative Regularization + Diversity Regularization

    论文 Fig.4 描述的两个步骤：
      1. Semantic Embedding Extraction  →  z = Encoder(s)
      2. Semantic Embedding Quantization →  [c1, c2, ..., cL] via L-level codebooks

    论文 Eq.(5):
      L_LETTER = L_Sem + α·L_CF + β·L_Div

    其中 L_Sem = L_Recon + L_RQVAE（Eq.(2)）
          L_RQVAE = Σ_l ‖sg[r_l-1] − e_cl‖² + μ·‖r_l-1 − sg[e_cl]‖²
    """

    def __init__(
        self,
        input_dim: int = 128,        # semantic embedding 维度（yambda LLM 输出）
        code_dim: int = 32,           # encoder 隐向量 & 码字维度 d
        num_levels: int = 4,          # identifier 长度 L
        codebook_size: int = 256,     # 每层码本大小 N
        commitment_mu: float = 0.25,  # 论文 Eq.(2) 中 μ，控制 encoder 靠近码本的强度
        use_recon: bool = True,       # 是否包含 decoder（L_Recon）
        cf_dim: int = 64,             # CF embedding 维度（来自 SASRec 等 CF 模型）
        cf_proj_dim: int = 32,       # CF 映射层输出维度（建议 ≤ cf_dim）
        div_cluster_k: int = 8,       # Diversity loss 每层簇数 K
    ):
        super().__init__()
        self.input_dim = input_dim
        self.code_dim = code_dim
        self.num_levels = num_levels
        self.codebook_size = codebook_size
        self.commitment_mu = commitment_mu
        self.use_recon = use_recon
        self.cf_dim = cf_dim
        self.div_cluster_k = div_cluster_k

        # ── Semantic Embedding Extraction ────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, code_dim),
        )

        # ── L-level Codebooks（RQ-VAE 核心）─────────────────────────────────
        self.codebooks = nn.ModuleList([
            nn.Embedding(codebook_size, code_dim) for _ in range(num_levels)
        ])
        for cb in self.codebooks:
            xavier_normal_(cb.weight)

        # ── Decoder（Semantic Embedding Reconstruction）────────────────────────
        if use_recon:
            self.decoder = nn.Sequential(
                nn.Linear(code_dim, 256),
                nn.ReLU(),
                nn.Linear(256, input_dim),
            )

        # ── CF Embedding Head ────────────────────────────────────────────────
        self.cf_head = CFEmbeddingHead(code_dim, cf_proj_dim)

        # CF embeddings 由外部注入（训练时通过 register_buffer）
        self._cf_embeddings = None

    # ─────────────────────────────────────────────────────────────────────────
    def set_cf_embeddings(self, cf_embeddings: torch.Tensor):
        """
        注入预训练 CF 模型生成的 item CF embeddings。

        Args:
            cf_embeddings: (N_items, cf_dim) 预训练 CF 模型的 item embeddings
        """
        self._cf_embeddings = cf_embeddings
        self.to(cf_embeddings.device)

    # ─────────────────────────────────────────────────────────────────────────
    def _quantize_layer(
        self,
        residual: torch.Tensor,
        cb_weight: torch.Tensor,
    ):
        """
        单层 RQ 量化（论文 Eq.(1)）

        Args:
            residual: (B, code_dim)   当前层残差 r_{l-1}
            cb_weight: (K, code_dim) 当前层码本权重

        Returns:
            quantized:  (B, code_dim)  量化后向量 e_{c_l}
            indices:    (B,)           码字下标 c_l
            cb_loss:    标量            sg[r_{l-1}] − e_{c_l}  (梯度流向码本)
            commit_loss: 标量            r_{l-1} − sg[e_{c_l}]  (梯度流向 encoder)
        """
        # L2 距离展开：‖r - e‖² = ‖r‖² + ‖e‖² - 2⟨r, e⟩
        dist = (
            residual.pow(2).sum(1, keepdim=True)
            + cb_weight.pow(2).sum(1)
            - 2 * torch.matmul(residual, cb_weight.t())
        )  # (B, K)
        indices = torch.argmin(dist, dim=-1)       # (B,)
        quantized = F.embedding(indices, cb_weight)  # (B, code_dim)

        # 论文 Eq.(2) L_RQVAE 第一项：梯度流向码本
        cb_loss = F.mse_loss(quantized, residual.detach())

        # 论文 Eq.(2) L_RQVAE 第二项：梯度流向 encoder
        commit_loss = F.mse_loss(quantized.detach(), residual)

        return quantized, indices, cb_loss, commit_loss

    # ─────────────────────────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,
        cf_embeds: torch.Tensor = None,
        alpha: float = 0.0,
        beta: float = 0.0,
    ):
        """
        完整前向传播。

        Args:
            x:         (B, input_dim) 原始 semantic embedding（来自 LLM）
            cf_embeds: (B, cf_dim)   该 batch 对应的 CF embeddings（可省略，优先用 buffer）
            alpha:     CF loss 权重
            beta:      Diversity loss 权重

        Returns:
            recon_x:  (B, input_dim) 重构 embedding（use_recon=True 时有效）
            z_q:     (B, code_dim)  量化后隐向量 Σ_l e_{c_l}
            codes:   (B, num_levels) 每层码字下标
            losses:  dict 各损失项的标量值
        """
        B = x.size(0)

        # ── 1. Semantic Embedding Extraction ────────────────────────────────
        z_e = self.encoder(x)          # (B, code_dim)

        # ── 2. RQ Quantization（论文 Eq.(1)）───────────────────────────────
        residual = z_e
        z_q = torch.zeros_like(z_e)
        codes = []
        cb_total = 0.0
        commit_total = 0.0

        for level in range(self.num_levels):
            cb_weight = self.codebooks[level].weight
            q, idx, cb_l, commit_l = self._quantize_layer(residual, cb_weight)
            codes.append(idx)
            cb_total += cb_l
            commit_total += commit_l

            # STE（Straight-Through Estimator）
            q = residual + (q - residual).detach()
            z_q = z_q + q
            # 残差 detach，隔断跨层梯度
            residual = residual - q.detach()

        codes = torch.stack(codes, dim=1)   # (B, num_levels)

        # ── 3. Semantic Regularization（论文 Eq.(2)）──────────────────────
        # L_RQVAE = Σ_l ‖sg[r_l-1] − e_cl‖² + μ·‖r_l-1 − sg[e_cl]‖²
        L_RQVAE = cb_total + self.commitment_mu * commit_total

        L_Recon = 0.0
        recon_x = None
        if self.use_recon:
            recon_x = self.decoder(z_q)         # (B, input_dim)
            L_Recon = F.mse_loss(recon_x, x)    # ‖s − ŝ‖²

        L_Sem = L_RQVAE + L_Recon

        # ── 4. Collaborative Regularization（论文 Eq.(3)）──────────────────
        L_CF = torch.tensor(0.0, device=x.device)
        if alpha > 0 and self._cf_embeddings is not None:
            z_q_norm = self.cf_head(z_q)           # (B, cf_proj_dim), normalized

            # 从 buffer 中取该 batch 的 CF embeddings
            if cf_embeds is None:
                cf_batch = self._cf_embeddings[:B]  # fallback
            else:
                cf_batch = cf_embeds

            cf_batch = F.normalize(cf_batch[:, :self.cf_dim], dim=-1)
            # cf_proj_dim ≤ cf_dim，确保 dim 兼容
            if cf_batch.size(-1) > self.cf_head.proj.out_features:
                cf_batch = cf_batch[:, :self.cf_head.proj.out_features]
            elif cf_batch.size(-1) < self.cf_head.proj.out_features:
                pad = torch.zeros(B, self.cf_head.proj.out_features - cf_batch.size(-1),
                                  device=cf_batch.device)
                cf_batch = torch.cat([cf_batch, pad], dim=-1)

            # In-batch softmax NCE（论文 Eq.(3)）
            logits = torch.matmul(z_q_norm, cf_batch.T) / 0.1  # temperature=0.1
            labels = torch.arange(B, device=x.device)
            L_CF = F.cross_entropy(logits, labels)

        # ── 5. Diversity Regularization（论文 Eq.(4)）──────────────────────
        L_Div = torch.tensor(0.0, device=x.device)
        if beta > 0:
            # 对每层分别计算 diversity loss
            for level in range(self.num_levels):
                cb_weight = self.codebooks[level].weight   # (K, code_dim)

                # 该 batch 中每条样本被分配的码字
                assigned = codes[:, level]               # (B,)

                # 找出分配了相同码字的正样本对（同码字 → 同簇）
                # 构建 (B, K) one-hot，取出每条样本被分配的码字 embedding
                e_cl = F.embedding(assigned, cb_weight)   # (B, code_dim)

                # 同簇正样本：在同 batch 中随机选一个同码字的样本
                # 找同码字的其他样本索引
                pos_idx = torch.zeros(B, dtype=torch.long, device=x.device)
                for b in range(B):
                    same_code_mask = (codes[:, level] == assigned[b])
                    same_code_idx = torch.where(same_code_mask)[0]
                    same_code_idx = same_code_idx[same_code_idx != b]
                    if len(same_code_idx) > 0:
                        pos_idx[b] = same_code_idx[torch.randint(len(same_code_idx), (1,))]
                    else:
                        pos_idx[b] = b  # 没有正样本时选自己

                e_pos = F.embedding(pos_idx, cb_weight)    # (B, code_dim) 正样本

                # 负样本：同 batch 中所有其他码字
                # 计算 (B, B) 相似度矩阵，排除自己
                all_cb = cb_weight                            # (K, code_dim)
                # 同 batch 内：e_cl[i] 与所有码字的相似度
                # 为节省显存，用采样策略：取当前码字和非当前码字
                e_cl_exp = e_cl.unsqueeze(1)                  # (B, 1, code_dim)
                all_cb_exp = all_cb.unsqueeze(0)              # (1, K, code_dim)
                sim_all = F.cosine_similarity(e_cl_exp, all_cb_exp, dim=-1)  # (B, K)

                # 正样本 logits：e_cl 与 e_pos 的相似度
                sim_pos = F.cosine_similarity(e_cl, e_pos, dim=-1, eps=1e-8)  # (B,)

                # Diversity loss：让同簇拉近，非同簇推开
                # 用一个简化的 NCE：同一簇 vs 随机负样本
                # 论文 Eq.(4) 的简化版
                neg_logits = sim_all  # (B, K)
                pos_logits = sim_pos.unsqueeze(-1)  # (B, 1)

                # 构造正负样本标签：pos 位置 = 1，其他 = 0
                labels_div = torch.zeros(B, self.codebook_size, device=x.device)
                labels_div.scatter_(1, assigned.unsqueeze(-1), 1.0)
                # 但这样 K 个负样本太多，改用采样负样本

                # 简化版 diversity loss：让同码字样本拉近，不同码字推开
                # L_div = -log exp(sim_pos) / (exp(sim_pos) + Σ_neg exp(sim_neg))
                neg_logits = logits = (
                    e_cl @ all_cb.T
                ) / 0.1  # temperature=0.1

                # 正样本 logits
                pos_score = (e_cl * e_pos).sum(dim=-1) / 0.1  # (B,)

                # 取前 N_neg 个最难负样本（相似度最高的非正样本）
                mask_pos = torch.zeros_like(neg_logits).scatter_(
                    1, assigned.unsqueeze(-1), float('-inf')
                )
                neg_logits_safe = neg_logits + mask_pos

                # 每条样本随机选 N_neg=8 个最难负样本
                N_neg = min(8, self.codebook_size - 1)
                topk_neg, _ = neg_logits_safe.topk(N_neg, dim=1, largest=True)  # (B, N_neg)

                # In-batch NCE
                denom = torch.exp(pos_score).unsqueeze(-1) + torch.exp(topk_neg).sum(dim=1, keepdim=True)
                L_Div_layer = -torch.log(torch.exp(pos_score) / (denom.squeeze() + 1e-8) + 1e-8).mean()
                L_Div = L_Div + L_Div_layer

            L_Div = L_Div / self.num_levels  # 平均到每层

        # ── 6. 总损失（论文 Eq.(5)）────────────────────────────────────────
        total = L_Sem + alpha * L_CF + beta * L_Div

        losses = {
            'total': total,
            'sem': L_Sem,
            'rq_vae': L_RQVAE,
            'recon': L_Recon,
            'cf': alpha * L_CF,
            'div': beta * L_Div,
        }

        return recon_x, z_q, codes, losses

    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def get_codes(self, x: torch.Tensor):
        """纯推理：只返回离散 code sequence，不做重建。"""
        z_e = self.encoder(x)
        residual = z_e
        codes = []

        for level in range(self.num_levels):
            cb_weight = self.codebooks[level].weight
            q, indices, _, _ = self._quantize_layer(residual, cb_weight)
            codes.append(indices)
            residual = residual - q.detach()

        return torch.stack(codes, dim=1)   # (B, num_levels)

    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def decode_from_codes(self, codes: torch.Tensor):
        """给定 (B, num_levels) 码字下标，解码回 (B, input_dim) semantic embedding。"""
        assert self.use_recon, "use_recon=False 时无法 decode"
        z_q = torch.zeros(codes.size(0), self.code_dim, device=codes.device)
        for level in range(self.num_levels):
            z_q = z_q + self.codebooks[level](codes[:, level])
        return self.decoder(z_q)

    # ─────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def get_z_q(self, x: torch.Tensor):
        """返回量化后的隐向量 z_q（用于 CF alignment 分析）。"""
        z_e = self.encoder(x)
        residual = z_e
        z_q = torch.zeros_like(z_e)

        for level in range(self.num_levels):
            cb_weight = self.codebooks[level].weight
            q, _, _, _ = self._quantize_layer(residual, cb_weight)
            z_q = z_q + q.detach()
            residual = residual - q.detach()

        return z_q
