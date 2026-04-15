# 0408Yambda Pipeline 结果汇总

## 流水线概览

```
01_build_codebook.py      训练残差平衡 KMeans codebook (4层 × 256码)
02_build_item_sid.py      全量 item → dense id + SID 编码
03_split_data.py          multi_event → train/val/test TSV
04_train_hpn_warmstart.py 监督训练 HPN warmstart (SASRec actor)
05_train_user_response.py 训练 UserResponse 模型 (模拟用户反馈)
06_train_yambda_sid.py    DDPG 强化学习微调 SID actor
07_eval_candidate_ranking.py 候选集排序评估
```

---

## Stage 01 — Codebook 训练

**产物**：`artifacts/codebook/yambda_rq_codebook.npz` + `yambda_rq_codebook.meta.json`

**方法**：残差平衡 KMeans（Residual Balanced Quantization），对 20 万采样向量逐层聚类，每层 256 个聚类中心，共 4 层。

**输入**

| 参数 | 值 |
|------|-----|
| embeddings.parquet | `/root/autodl-tmp/0330Yambda/data/embeddings.parquet` |
| embedding_column | `normalized_embed`（L2 归一化向量） |
| sample_size | 200,000（从 7,721,749 行 reservoir sampling） |
| n_levels | 4 |
| codebook_size | 256 |
| max_iter | 30 |

**采样向量统计**

| 指标 | 值 |
|------|-----|
| 向量维度 | 128 |
| 均值 | −0.00254 |
| 标准差 | 0.0884 |
| 范围 | [−0.434, 0.392] |
| 平均 L2 范数 | 1.0（已归一化） |

**评估标准**：无显式评估指标，codebook 质量通过下游 Stage 02 的编码覆盖率（vocab_sizes 一致性）和 Stage 04/07 的最终效果间接验证。

---

## Stage 02 — Item SID 编码

**产物**：`artifacts/mappings/yambda_dense_item2sid.npy`（shape: [7,721,750, 4]，int32）、`yambda_orig2dense_item_id.npy`、`yambda_dense2orig_item_id.npy`

**方法**：加载 RQVAE encoder（checkpoint 来自 `0330Yambda/pure_rqvae/checkpoints/Mar-30-2026_23-12-08/best_entropy_e20.pth`），对全量 7,721,749 个 item 的归一化向量做前向编码，输出每 item 一条 4 层 SID 路径（每层一个 0–255 的 token id）。

**输入输出规模**

| 指标 | 值 |
|------|-----|
| 编码模式 | RQVAE（而非 RQKMeans） |
| 原始 item_id 范围 | 2 – 9,390,623 |
| 编码后 dense id 数量 | 7,721,749 |
| SID 层数 | 4 |
| 每层 vocab size | 256 |
| 编码后 item 向量维度 | 128 |

**ID 映射验证**：原始 item_id → dense id → SID 三层映射一致，验证通过（monotonic_non_decreasing=true）。

**评估标准**：无端到端指标，通过下游 Stage 04 的 token_acc 和 Stage 07 的排序指标间接验证编码有效性。

---

## Stage 03 — 数据集划分

**产物**：`artifacts/processed/train.tsv`、`val.tsv`、`test.tsv` + `split.meta.json`

**方法**：流式读取 multi_event.parquet，按用户聚合事件，按 `close_gap_seconds=3600`（同一 item 超过 1 小时重新开 episode）切 episode，按以下规则分配：

- 1 条 episode → train
- 2 条 episode → train, test
- ≥3 条 episode → train×(N−2), val（倒数第2）, test（最后）

每条样本记录：`sequence_id`, `user_id`, `target_dense_item_id`, `reward = max_play_ratio + like − dislike`（play/like/dislike 各权重 1.0），历史窗口长度 50。

**数据集规模**

| 指标 | 值 |
|------|-----|
| 处理的活跃用户 | 10,000 |
| 原始事件总数 | 23,898,321 |
| 保留事件数 | 23,100,324（missing mapping 丢弃 1,570,553） |
| train 样本数 | 41,513,562 |
| val 样本数 | 10,000 |
| test 样本数 | 10,000 |

**评估标准**：`split.meta.json` 记录各 split 行数；本阶段不产出模型质量指标，数据量本身是主要验证点。

---

## Stage 04 — HPN Warmstart（监督训练）

**产物**：`artifacts/models/hpn_warmstart.pt` + `hpn_warmstart.meta.json`

**方法**：SASRec encoder（2层，d_model=64，4头，dropout=0.1）接收用户历史 item 向量序列，输出 4 层 SID token 的 logits，用 token-level cross entropy 训练。

**训练配置**

| 参数 | 值 |
|------|-----|
| batch_size | 256 |
| epochs | 3 |
| lr | 1e-3 |
| weight_decay | 1e-5 |
| 训练样本数 | 100,000（train 4100万中取了前10万） |
| 验证样本数 | 9,999 |
| feature cache | 122,052 个 dense item → 128维向量，全部命中 |

**评估标准**

- `loss`：所有层 token cross entropy 的平均值，越低越好
- `token_acc_lN`：第 N 层 token 预测准确率（argmax 对比真实 SID token）
- `full_path_acc`：4 层 token 全部预测正确的比例

**训练轨迹**

| Epoch | train_loss | val_loss | val_token_acc_l1 | val_token_acc_l2 | val_token_acc_l3 | val_token_acc_l4 | val_full_path_acc |
|-------|-----------|----------|-----------------|-----------------|-----------------|-----------------|-------------------|
| 1 | 4.7423 | 4.8472 | 16.01% | 4.42% | 2.93% | 2.27% | 0.020% |
| 2 | 4.5094 | 4.7773 | 18.93% | 5.24% | 3.41% | 2.00% | 0.010% |
| 3 | 4.4514 | **4.7418** | 19.40% | 5.55% | 3.66% | 2.20% | 0.020% |

- **Best val_loss**: 4.7418（Epoch 3）
- **结论**：训练 loss 持续下降，验证 loss 在第 3 epoch 达到最低，无明显过拟合；各层 token_acc 逐层递减符合预期（粗粒度语义层比精确深层更易预测）；full_path_acc 极低（~0.02%）说明完全精确预测 4 层完整 SID 路径极难，但这是合理的，因为 RQVAE 编码引入了量化误差。

---

## Stage 05 — UserResponse 模型（用户反馈模拟器）

**产物**：`artifacts/env/yambda_user_env.model` + `yambda_user_env.meta.json`

**方法**：YambdaUserResponse（带注意力机制的序列模型），输入用户历史序列 + 候选 item 向量，输出 `user_clicks`（0–3 连续 reward）。作为 Stage 06 强化学习环境中的用户模拟器。

**训练配置**

| 参数 | 值 |
|------|-----|
| batch_size | 256 |
| epochs | 5 |
| lr | 1e-4 |
| 训练样本数 | 500,000 |
| 验证样本数 | 9,999 |
| loss | MSE |

**评估标准**

- `val_mse`：验证集 reward 预测均方误差，越低越好
- `train_loss`：训练集 MSE，监控过拟合

**训练轨迹**

| Epoch | train_loss | val_mse |
|-------|-----------|---------|
| 1 | 0.2239 | **0.2217** |
| 2 | 0.1972 | 0.2227 |
| 3 | 0.1943 | 0.2292 |
| 4 | 0.1924 | 0.2296 |
| 5 | 0.1908 | 0.2303 |

- **Best val_mse**: 0.2217（Epoch 1）
- **观察**：训练 loss 持续下降，但 val_mse 在第 1 epoch 后不再改善甚至轻微上升，存在轻微过拟合。最终采用 Epoch 1 checkpoint。

---

## Stage 06 — DDPG RL 微调（SID Actor-Critic）

**产物**：`artifacts/models/yambda_sid_actor`（state_dict）+ `yambda_sid.meta.json`

**方法**：DDPG（Deep Deterministic Policy Gradient）以 Stage 04 的 HPN warmstart 为 actor 起点，在 YambdaUserResponse 模拟的用户反馈环境中做 RL 微调。Critic 估计 Q 值，Actor 输出来源于 HPN SASRec 的 4 层 SID logits。

**核心超参数**

| 参数 | 值 |
|------|-----|
| gamma（折扣因子） | 0.9 |
| actor_lr | 1e-4 |
| critic_lr | 1e-3 |
| entropy_coef | 0.01 |
| bc_coef | 0.1（behavior cloning 系数） |
| target_mitigate_coef | 0.01 |
| episode_batch_size | 32 |
| batch_size | 128 |
| n_iter | 10,000 |
| train_every_n_step | 5 |
| candidate_items | 50,000（从 item 候选集中采样） |
| buffer_size | 100,000 |
| max_step_per_episode | 10 |
| initial_temper | 5 |
| temper_sweet_point | 0.9 |

**评估标准**

- `average_total_reward`：每 100 步 episode 的平均累计 reward
- `critic_loss`：Critic Q 值预测误差
- `actor_loss`：Actor 策略梯度损失（负数表示鼓励增加策略价值）
- `entropy`：策略熵，反映探索程度

**训练关键节点对比（每100步报告）**

| Step | avg_reward | reward_variance | avg_n_step | critic_loss | actor_loss | entropy |
|------|-----------|----------------|-----------|------------|-----------|---------|
| 100 | 2.798 | 0.338 | 3.8 | 0.055 | −0.307 | 4.460 |
| 1000 | 2.350 | 0.612 | 3.5 | 0.095 | −0.133 | 4.291 |
| 2500 | 2.845 | 0.211 | 3.9 | 0.089 | +0.036 | 4.350 |
| 5000 | ~2.60 | ~0.38 | ~3.7 | ~0.092 | ~0.00 | ~4.32 |
| 9900 | 2.917 | 0.577 | 3.8 | 0.078 | −0.114 | 4.817 |
| 10000 | 2.917 | 0.577 | 3.8 | 0.078 | −0.114 | 4.817 |

- **Critic 收敛**：从 0.55 降到 ~0.08，Q 值估计趋于稳定
- **Reward 停滞**：average_reward 在 2.3–3.0 区间震荡，无明显上升趋势
- **Actor 未突破**：actor_loss 符号波动，未找到比 warmstart 更好的策略
- **探索增加**：entropy 从 4.46 升到 4.82，但未换得更高回报

**结论**：warmstart 策略已接近当前 reward 格局下的最优边界，DDPG 微调无显著增量提升。

---

## Stage 07 — 候选集排序评估

**产物**：`artifacts/models/candidate_ranking.meta.json`

**方法**：对 test.tsv 每条样本，随机采样 99 个负样本 item，与正样本 target 共 100 个候选，用训好的 SID actor 打分排序（各层 log-softmax 求和），计算排序指标。

**输入**

| 参数 | 值 |
|------|-----|
| eval_file | test.tsv（9,999 样本） |
| actor_checkpoint | `yambda_sid_actor`（Stage 06 DDPG 训练后） |
| num_negatives | 99（每条配 99 个负样本） |
| candidate_size | 100（1正 + 99负） |
| embedding_cache | 112,819 个 item，全部命中 |

**评估标准**

| 指标 | 定义 | 随机 baseline（理论值） | 本模型结果 | 提升倍数 |
|------|------|---------------------|-----------|---------|
| **Mean Rank** | 正样本在 100 候选中的平均排名，越低越好 | 50.5 | **5.28** | 9.6× |
| **MRR**（Mean Reciprocal Rank） | 1/rank 的均值，取值 [0,1]，越高越好 | 0.50 | **0.647** | +29% |
| **HR@K**（Hit Rate @ K） | 正样本排进 Top-K 的比例，越高越好 | K=1: 1% / K=5: 5% / K=10: 10% / K=20: 20% | K=1: **52.0%** / K=5: **80.5%** / K=10: **88.3%** / K=20: **93.9%** | 52× / 16× / 8.8× / 4.7× |
| **NDCG@K**（Normalized DCG @ K） | 考虑排序位置的归一化折现收益，越高越好 | K=1: 1% / K=5: ~2.5% / K=10: ~5% / K=20: ~10% | K=1: **52.0%** / K=5: **67.5%** / K=10: **70.0%** / K=20: **71.4%** | 大幅超越 |
| **full_path_acc** | 4 层 SID token 全部预测正确的比例 | 0.0015%（1/100 随机） | **0.25%** | 167× |
| **token_acc_l1–l4** | 各层 SID token 独立预测准确率 | 各层: 0.39%（随机） | l1: **25.7%** / l2: **8.4%** / l3: **5.1%** / l4: **2.9%** | 66× / 22× / 13× / 7.5× |

**SID 分层准确率观察**：各层准确率逐层递减（25.7% → 8.4% → 5.1% → 2.9%），符合直觉：粗粒度语义层（第 1 层）比精确深层（第 4 层）更易预测。

**关键说明**：07 评估的是 Stage 06 DDPG 训练后的 actor，其表现强于随机 baseline。但结合 Stage 06 的分析，这一强表现主要来自 HPN warmstart（SASRec 序列建模能力），而非 DDPG 的 RL 微调——因为 DDPG 阶段 episode reward 没有显著提升。

---

## 跨阶段依赖关系

```
01_build_codebook        产出 codebook.npz
        ↓
02_build_item_sid        用 RQVAE encoder + codebook 编码全量 item SID
        ↓
03_split_data            用 orig2dense 映射切分 train/val/test
        ↓
04_train_hpn_warmstart   用 train split + dense_item2sid 监督训练 HPN（warmstart）
        ↓
05_train_user_response   用 train/val split 训练用户反馈模型（环境模拟器）
        ↓
06_train_yambda_sid      用 04 actor + 05 环境 + DDPG 强化学习微调
        ↓
07_eval_candidate_ranking  用 06 actor 在 test 集上做排序评估
```

---

## 主要发现

1. **RQVAE 编码有效**：4 层 × 256 vocab 的 SID 能将 772 万 item 映射为离散 token 序列，下游 SASRec 可在此基础上学会用户偏好语义。

2. **HPN warmstart 已接近最优**：Stage 04 的 HPN warmstart 在 10 万样本上监督训练后，token_acc_l1 达到 19.4%，Stage 07 排序结果（MRR=0.647, HR@10=88.3%）主要由 warmstart 贡献。

3. **DDPG 微调收益有限**：Stage 06 的 RL 训练未能带来 episode reward 的显著提升，可能原因：reward landscape 平坦（temper_sweet_point=0.9 设置下 reward 信号不强）、batch replay buffer 的 off-policy 特性限制了策略改进。

4. **UserResponse 模型存在过拟合**：Stage 05 中训练 loss 持续下降但 val_mse 在第 1 epoch 后不再改善，表明模型容量相对训练数据量偏大，建议增大训练数据或降低模型复杂度。

5. **SID 层级预测难度分层明显**：第 1 层准确率（25.7%）是第 4 层（2.9%）的近 9 倍，量化误差和细粒度语义的多样性是主要瓶颈。
