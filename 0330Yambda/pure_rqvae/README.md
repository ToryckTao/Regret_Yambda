# RQVAE 训练

## 完整数据流程与维度变化

```
STEP 1: 预处理（只跑一次）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
embeddings.parquet          embeddings.npz
  normalized_embed 列           embeddings  (7721749, 128) float32
  每行是 [128 个 float] 的        item_ids   (7721749,)    uint32
  嵌套 List 结构（读取极慢）     ~4GB，约 1 秒加载

STEP 2: 数据加载
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
npz → EmbDataset → DataLoader(batch_size=1024)
torch.FloatTensor (1024, 128)
torch.LongTensor  (1024,)       ← index，对齐下游序列用

STEP 3: Encoder
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
(B, 128)            Linear(128→256) + ReLU + Linear(256→32)
        ──────────────────────────────────────────────→
                                                     (B, 32)
                                                     z_e

STEP 4: Residual Quantization（4 层）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
z_e = (B, 32)  ← 第 0 层残差初始值

Level 0:
  residual_l        (B, 32)
       │─→ 计算和 codebook[0] (256, 32) 的 L2 距离  → (B, 256)
       │─→ argmin → indices[0] ∈ {0..255}           (B,)
       │─→ 查表 → quantized[0]                      (B, 32)
       │
       ├─→ cb_loss     += ‖quantized - residual.detach()‖²     → 梯度流向码本
       ├─→ commit_loss += ‖quantized.detach() - residual‖² * β → 梯度流向 encoder
       │
       ├─→ z_q += quantized      (B, 32)  ← 量化结果累加
       └─→ residual = residual - quantized.detach()  (B, 32)  ← 新残差

Level 1 / 2 / 3: 同上，每层独立码本，独立残差

最终:
  codes     shape (B, 4)    ← 每条样本的离散 Token（如 [233,125,10,181]）
  z_q       shape (B, 32)   ← 4 层量化结果累加后的向量
  cb_loss   scalar          ← 4 层码本损失之和
  commit_loss scalar         ← 4 层承诺损失之和

STEP 5: Decoder（use_recon=True 时）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
z_q = (B, 32)            Linear(32→256) + ReLU + Linear(256→128)
        ──────────────────────────────────────────────→
                                                     (B, 128)
                                                     recon_x

STEP 6: 损失计算与反向传播
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
cb_loss     = Σ_l  ‖quantized_l − residual_l.detach()‖²      → 流向 codebook 权重
commit_loss = β × Σ_l  ‖quantized_l.detach() − residual_l‖² → 流向 encoder 权重
recon_loss  = MSE(recon_x, x)                                 → 流向 encoder + decoder

loss_total = cb_loss + commit_loss + recon_loss
（use_recon=False 时 recon_loss = 0）
```

---

## 三个 Loss 分别优化什么

| 损失 | 公式 | 梯度流向 | 作用 |
|------|------|----------|------|
| **cb_loss**（码本损失） | ‖quantized − residual.detach()‖² | codebook 权重 | 让码字学会靠近残差 |
| **commit_loss**（承诺损失） | ‖quantized.detach() − residual‖² × β | encoder 权重 | 让 encoder 输出靠近码字 |
| **recon_loss**（重构损失） | MSE(recon_x, x) | encoder + decoder + codebook | 让量化结果能重建原始 embedding |

---

## use_recon 开关

```
--use_recon True   → 加重构损失   loss = cb + commit + recon
                     训练慢（多一次 decoder forward + loss 计算）
                     但 codes 语义完整，下游大模型效果有保证
                     【需要用 RQVAE 做音频生成时必须用这个】

--use_recon False  → 不加重构损失  loss = cb + commit
                     训练快
                     但 encoder 只能靠 commit_loss 驱动，语义质量无保证
                     【纯做离散 token 化、不需要生成时可用这个】
```

---

## 文件说明

| 文件 | 作用 |
|------|------|
| `preprocess.py` | parquet → npz 预处理（只需跑一次） |
| `dataset.py` | 加载 npz，返回 `(embedding, index)` |
| `model.py` | RQVAE（Encoder + RQ 4层 + Decoder） |
| `trainer.py` | 训练循环 + 三项 loss 监控 + 评估 + 保存 ckpt |
| `main.py` | 入口，参数解析，构建模型 |
| `utils.py` | 目录创建、格式化时间 |
| `generate_indices.py` | 训练完后导出所有 item 的离散 Token 序列 |
| `requirements.txt` | 依赖 |

---

## 使用流程

### Step 1: 预处理（只一次）

```bash
cd pure_rqvae
python preprocess.py \
    --input ../data/embeddings.parquet \
    --output ../data/embeddings.npz \
    --col normalized_embed
```

### Step 2: 训练

```bash
# 含重构（推荐）
python main.py --use_recon True --epochs 200 --device cuda:0

# 纯量化（不用生成）
python main.py --use_recon False --epochs 200 --device cuda:0
```

### Step 3: 导出 Token 序列

```bash
python generate_indices.py \
    --ckpt checkpoints/.../best_collision_e*.pth \
    --data ../data/embeddings.npz \
    --output ../data/item_tokens.json
```

---

## 主要参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input_dim` | 128 | 自动从数据检测 |
| `--code_dim` | 32 | encoder 输出维度 |
| `--num_levels` | 4 | 残差量化层数 |
| `--codebook_size` | 256 | 每层码本大小 |
| `--commitment_beta` | 0.25 | 承诺损失权重 |
| `--use_recon` | True | 是否加重构损失 |
| `--batch_size` | 1024 | |
| `--epochs` | 200 | |
| `--lr` | 1e-3 | |
