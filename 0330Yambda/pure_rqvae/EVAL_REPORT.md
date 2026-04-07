# RQVAE Tokenization 评估报告

> 评估对象：`best_entropy_e20.pth`（Mar-30-2026_23-12-08）
> 数据规模：7,721,749 items / 10 万采样评估 / dim=128
> 参数量：code_dim=32, num_levels=4, codebook_size=256
> 训练策略：loss = cb_loss + 0.25 * commit_loss + recon_loss

---

## 一、评估维度总览

| # | 维度 | 核心问题 | 主要指标 |
|---|---|---|---|
| 1 | 重构质量 | decoder 能还原原始 embedding 吗？ | MSE / CosSim / Pearson R |
| 2 | Token 分布 | 各层码字分配是否均匀？ | norm_entropy / Gini / collision_rate |
| 3 | 码本利用率 | 码本有没有被充分利用？ | global_active / top10_pct |
| 4 | 近邻召回 | token 近邻关系保留了多少语义？ | Recall@K / Spearman r / MRR |
| 5 | 聚类质量 | token 能否捕捉 embedding 的自然结构？ | Silhouette / NMI |

---

## 二、训练指标（从 checkpoint 读取）

| Checkpoint | Val Recon Loss | Collision Rate | Global Norm Entropy |
|---|---|---|---|
| e5 | 0.000981 | 3.91% | 0.9968 |
| e10 | 0.000961 | 3.88% | 0.9975 |
| e15 | 0.000948 | **3.88%** | 0.9978 |
| e20 | **0.000941** | 3.88% | **0.9980** ← 评估对象 |

- **recon loss** 在 20 epoch 内持续下降，说明 decoder 仍在学习
- **collision rate** 在 e10 之后收敛，e15 ~ e20 不再变化
- **global norm entropy** 持续上升接近 1，说明码本分配越来越均匀

---

## 三、评估结果（采样 10 万）

### 3.1 重构质量（Reconstruction）

| 指标 | 数值 | 解读 |
|---|---|---|
| MSE | **0.000940** | 非常低，embedding 能量被很好保留 |
| Cosine Sim（均值） | **0.938** | 方向一致性很高 |
| Cosine Sim（p90） | 0.970 | 90% 的样本 cosine sim > 0.97 |
| Cosine Sim（p50） | 0.945 | 中位数表现稳健 |
| Cosine Sim（p10） | 0.898 | 最差 10% 仍有 ~0.90 |
| Pearson R | **0.938** | 线性相关性强 |
| Cosine Sim（std） | 0.030 | 各样本重构质量波动小 |

**小结**：重构质量优秀，MSE 仅 0.00094，cosine sim 均值 0.938，说明 decoder 能很好地还原原始 embedding。p10 仍有 0.898，整体分布健康。

---

### 3.2 Token 分布（Per-level）

| 层 | Active Codes | Utilization | norm_entropy | Gini | Top-1 Freq |
|---|---|---|---|---|---|
| Level 0 | 256/256 | 100% | 0.9910 | 0.171 | 1.07% |
| Level 1 | 255/256 | 99.6% | 0.9939 | 0.120 | 1.09% |
| Level 2 | 250/256 | 97.7% | 0.9918 | 0.119 | 1.23% |
| Level 3 | 231/256 | 90.2% | 0.9805 | 0.148 | 0.65% |

- **Level 3 明显弱于前 3 层**：只有 231/256 个码字被使用，熵也最低（0.9805），说明高层残差更稀疏
- **Gini 系数都偏低**（0.12~0.17），说明各层码字分配相对均匀，没有严重倾斜
- **Level 3 Top-1 freq 只有 0.65%**（最不倾斜），验证了它虽然 active codes 少但分布均匀

**4-tuple 整体碰撞率：1.73%（98,271 / 100,000 unique）**

---

### 3.3 码本利用率（Codebook Utilization）

| 指标 | 数值 |
|---|---|
| Global Active Codes | 256/256（100%） |
| Global norm_entropy | **0.9980**（极均匀） |
| 最多被共享的 Token | **37 items**（共 10 万） |
| Top-10% Token 占比 | **11.73%** |

**小结**：码本利用率接近完美，所有 256 个码字都被使用，norm_entropy 0.998 说明分配极均匀。最热门的 4-tuple 也只被 37/100,000 items 共享，说明碰撞率低且分布均匀。

---

### 3.4 近邻召回（Nearest Neighbor Recall）⚠️ 核心短板

| 指标 | 数值 | 解读 |
|---|---|---|
| Recall@20 | **0.174** | 语义保留较差 |
| Spearman r | **0.088** | embedding 和 token 空间排序相关性弱 |
| MRR | **0.0002** | token 近邻几乎找不到真实近邻 |

**这是最重要的发现**：

- **Recall@20 = 17.4%**：在 embedding 空间里互为 top-20 近邻的 pair 中，只有 17.4% 在 token 空间里也保持近邻关系
- **Spearman r ≈ 0.09**：两种相似度排序的相关性接近随机
- **原因分析**：4-tuple 的 one-hot 编码天然把相似度变成了 token 重叠率的度量（两个序列有相同位置 token 才加分），而 cosine 相似度是连续空间的度量，两者本质不同
- **更关键的问题**：Level 0 的码字和原始 embedding 空间强耦合（norm_entropy 0.99），但跨层之间是残差量化，每层只管自己的残差部分，叠加后才能重构 full embedding，所以单看每层码字的 one-hot overlap 不足以反映连续空间的相似性

**改进方向**：
- 用 **weighted token overlap** 替代 uniform one-hot cosine（越低层的码字越重要）
- 或者直接算 `decode_from_codes` 后的重建向量 cosine similarity

---

### 3.5 聚类质量（Clustering, k=128）

| 指标 | 数值 | 解读 |
|---|---|---|
| Silhouette（Emb 空间） | **0.108** | embedding 聚类边界较模糊（数据特性） |
| Silhouette（Tok 空间） | **0.017** | token 序列相似性更分散 |
| NMI（Emb vs Tok） | **0.592** | token 和 embedding 的聚类结构有中等相关性 |

**小结**：embedding 本身的 silhouette 就只有 0.108（底层数据可能没有清晰聚类边界），NMI 0.592 说明 token 序列确实捕捉了部分 embedding 聚类结构，但仍有提升空间。

---

## 四、综合评价

| 维度 | 评分 | 说明 |
|---|---|---|
| 重构质量 | ⭐⭐⭐⭐⭐ | MSE 0.00094，cosine sim 0.938，p10 也有 0.898 |
| Token 分布 | ⭐⭐⭐⭐ | norm_entropy 0.998，Gini 0.017，极均匀；Level 3 是唯一短板 |
| 码本利用率 | ⭐⭐⭐⭐⭐ | 256/256 码字全被用，最热 token 仅占 0.037% |
| 近邻召回 | ⭐⭐ | Recall@20=0.174，Spearman r=0.088，核心短板 |
| 聚类质量 | ⭐⭐⭐ | NMI=0.592，silhouette 低可能源于数据本身特性 |

### 下一步建议

1. **优先改进近邻召回**：RQVAE 的残差量化机制决定了 token 重叠率和连续余弦相似度天然不等价。可以尝试：
   - 在 loss 里加一个"近邻一致性"正则项（用采样 hard negative pairs）
   - 用加权 token overlap 作为 token 空间相似度替代 uniform cosine
2. **Level 3 是瓶颈**：考虑增加 Level 3 的 commitment_beta 或增大 codebook_size
3. **训练还没饱和**：recon loss 在 e10~e20 仍在下降，建议继续训练到 e50~e100，看近邻召回是否同步提升

---

## 五、评估脚本说明

评估逻辑和训练 `trainer.eval()` 的关系：

| 指标 | 训练时（验证集） | 评估脚本（全量/采样） |
|---|---|---|
| recon loss | MSE 单值 | MSE + CosSim + Pearson R + 分位数 |
| token 分布 | active / entropy / collision | + Gini + Top-1 freq（新增） |
| 码本利用率 | global active / norm_H | + max_item_share / top10_pct（新增） |
| 近邻召回 | 无 | Recall@K / Spearman / MRR（新增） |
| 聚类质量 | 无 | Silhouette / NMI（新增） |

评估脚本 `evaluate.py` 使用方法：

```bash
# 快速采样（推荐先跑这个）
python evaluate.py \
    --ckpt ./checkpoints/Mar-30-2026_23-12-08/best_entropy_e20.pth \
    --data ../data/embeddings.npz \
    --output ./eval_results_sample.json \
    --sample_size 100000 \
    --nn_sample 5000 \
    --nn_k 20

# 全量评估（较慢）
python evaluate.py \
    --ckpt ./checkpoints/Mar-30-2026_23-12-08/best_entropy_e20.pth \
    --data ../data/embeddings.npz \
    --output ./eval_results_full.json
```
