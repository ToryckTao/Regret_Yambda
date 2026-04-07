# RQVAE 完整脚本流程

---

## Step 1：预处理 parquet → npz（只跑一次）

```bash
cd /root/autodl-tmp/0330Yambda/pure_rqvae

python preprocess.py \
    --input ../data/embeddings.parquet \
    --output ../data/embeddings.npz \
    --col normalized_embed
```

---

## Step 2：训练

```bash
cd /root/autodl-tmp/0330Yambda/pure_rqvae

# 验证脚本能跑通：2 个 epoch
python main.py \
    --data ../data/embeddings.npz \
    --epochs 2 \
    --batch_size 1024 \
    --device cuda:0 \
    --use_recon True \
    --eval_every 1 \
    --save_every 2

# 完整训练：20 个 epoch（实际使用）
python main.py \
    --data ../data/embeddings.npz \
    --epochs 20 \
    --batch_size 1024 \
    --device cuda:0 \
    --use_recon True \
    --eval_every 5 \
    --save_every 10
```

---

## Step 4：断点续训（可选）

```bash
# 从某个 checkpoint 继续训练
python main.py \
    --data ../data/embeddings.npz \
    --epochs 400 \
    --batch_size 1024 \
    --device cuda:0 \
    --use_recon True \
    --resume ./checkpoints/Mar-30-2026_23-12-08/best_entropy_e20.pth
```

---

## Step 5：评估（训练完成后）

```bash
cd /root/autodl-tmp/0330Yambda/pure_rqvae

# 快速采样评估（10 万条，有进度条，约 3 分钟）
python evaluate.py \
    --ckpt ./checkpoints/Mar-30-2026_23-12-08/best_entropy_e20.pth \
    --data ../data/embeddings.npz \
    --output ./eval_results_sample.json \
    --sample_size 100000 \
    --nn_sample 5000 \
    --nn_k 20

# 全量评估（约 770 万条，无采样，较慢）
python evaluate.py \
    --ckpt ./checkpoints/Mar-30-2026_23-12-08/best_entropy_e20.pth \
    --data ../data/embeddings.npz \
    --output ./eval_results_full.json

# 生成评估报告（读取已有结果）
python -c "
import json
from pathlib import Path
md = Path('EVAL_REPORT.md')
print('报告已保存在: ' + str(md.resolve()))
"
```

---

## Step 6：导出 Token 序列（训练完成后）

```bash
# 找到最佳 entropy 或 collision 的 checkpoint
python generate_indices.py \
    --ckpt ./checkpoints/.../best_entropy_e*.pth \
    --data ../data/embeddings.npz \
    --output ../data/item_tokens.json

python generate_indices.py \
    --ckpt ./checkpoints/.../best_collision_e*.pth \
    --data ../data/embeddings.npz \
    --output ../data/item_tokens_collision.json
```

---

## 完整参数说明

```bash
--data             npz 文件路径
--batch_size       批大小，默认 1024
--num_workers      数据加载线程数，默认 8
--device           设备，默认 cuda:0
--epochs           训练轮数，默认 200
--eval_every       每多少轮评估一次，默认 10
--save_every       每多少轮保存一次，默认 50

--input_dim        embedding 维度（自动检测，可不填）
--code_dim         量化隐向量维度，默认 32
--num_levels       残差量化层数，默认 4
--codebook_size    每层码本大小，默认 256
--commitment_beta  承诺损失权重，默认 0.25

--use_recon        是否加重构损失
                   True  → loss = cb + commit + recon（含重建，推荐）
                   False → loss = cb + commit（纯量化，训练快）

--resume           断点续训的 .pth 路径
--ckpt_dir         checkpoint 保存目录，默认 ./checkpoints
```
