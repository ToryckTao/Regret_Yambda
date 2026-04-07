#!/bin/bash
# LETTER-RQVAE 完整脚本（分步执行版）
# 每个步骤独立运行，复制粘贴到终端即可
# 路径全部硬编码为绝对路径，不再依赖环境变量

# =============================================================================
# Step 0：环境检查
# =============================================================================
echo "=============================================================="
echo "Step 0: 环境检查"
echo "=============================================================="
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python3 -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"
python3 -c "import scipy; print(f'scipy: {scipy.__version__}')"

# =============================================================================
# Step 1：预处理 embeddings.npz（如已存在则跳过）
# =============================================================================
echo ""
echo "=============================================================="
echo "Step 1: 预处理 embeddings.parquet → embeddings.npz"
echo "=============================================================="
if [ ! -f "/root/autodl-tmp/0330Yambda/data/embeddings.npz" ]; then
    python /root/autodl-tmp/0330Yambda/pure_rqvae/preprocess.py \
        --input /root/autodl-tmp/0330Yambda/data/embeddings.parquet \
        --output /root/autodl-tmp/0330Yambda/data/embeddings.npz \
        --col normalized_embed
else
    echo "  [SKIP] embeddings.npz 已存在，跳过 Step 1"
fi

# =============================================================================
# Step 2：训练 SASRec 并生成 CF Embeddings（如已存在则跳过）
# =============================================================================
echo ""
echo "=============================================================="
echo "Step 2: 训练 SASRec → 生成 CF Embeddings"
echo "=============================================================="
if [ ! -f "/root/autodl-tmp/0330Yambda/data/cf_embeddings.npz" ]; then
    python /root/autodl-tmp/0330Yambda/LETTER_rqvae/cf_trainer.py \
        --input_combo likes,listens,multi_event \
        --data_dir /root/autodl-tmp/0330Yambda/data/sequential-50m \
        --embeddings /root/autodl-tmp/0330Yambda/data/embeddings.npz \
        --output /root/autodl-tmp/0330Yambda/data/cf_embeddings.npz \
        --dataset_name combined \
        --hidden_units 64 \
        --num_blocks 2 \
        --num_heads 2 \
        --dropout_rate 0.2 \
        --maxlen 50 \
        --num_epochs 200 \
        --batch_size 256 \
        --lr 0.001 \
        --eval_every 20 \
        --patience 5 \
        --device cuda:0
else
    echo "  [SKIP] cf_embeddings.npz 已存在，跳过 Step 2"
fi

# =============================================================================
# Step 3：验证 CF embeddings 质量
# =============================================================================
echo ""
echo "=============================================================="
echo "Step 3: 验证 CF Embeddings"
echo "=============================================================="
python3 -c "
import numpy as np
cf = np.load('/root/autodl-tmp/0330Yambda/data/cf_embeddings.npz')
print(f'  CF embeddings shape: {cf[\"cf_embeddings\"].shape}')
print(f'  item_ids shape:      {cf[\"item_ids\"].shape}')
print(f'  CF dim:              {cf[\"cf_embeddings\"].shape[1]}')
"

# =============================================================================
# Step 4a：LETTER-RQVAE 快速验证（2 epochs）
# =============================================================================
echo ""
echo "=============================================================="
echo "Step 4a: LETTER-RQVAE 快速验证（2 epochs）"
echo "=============================================================="
python /root/autodl-tmp/0330Yambda/LETTER_rqvae/main.py \
    --data /root/autodl-tmp/0330Yambda/data/embeddings.npz \
    --cf_embeddings /root/autodl-tmp/0330Yambda/data/cf_embeddings.npz \
    --epochs 2 \
    --batch_size 1024 \
    --device cuda:0 \
    --alpha 0.1 \
    --beta 0.01 \
    --eval_every 1 \
    --save_every 2 \
    --ckpt_dir /root/autodl-tmp/0330Yambda/LETTER_rqvae/checkpoints

# =============================================================================
# Step 4b：LETTER-RQVAE 完整训练（200 epochs）
# =============================================================================
echo ""
echo "=============================================================="
echo "Step 4b: LETTER-RQVAE 完整训练（200 epochs）"
echo "=============================================================="
python /root/autodl-tmp/0330Yambda/LETTER_rqvae/main.py \
    --data /root/autodl-tmp/0330Yambda/data/embeddings.npz \
    --cf_embeddings /root/autodl-tmp/0330Yambda/data/cf_embeddings.npz \
    --epochs 200 \
    --batch_size 1024 \
    --device cuda:0 \
    --alpha 0.1 \
    --beta 0.01 \
    --eval_every 5 \
    --save_every 10 \
    --ckpt_dir /root/autodl-tmp/0330Yambda/LETTER_rqvae/checkpoints

# =============================================================================
# Step 5：评估
# =============================================================================
echo ""
echo "=============================================================="
echo "Step 5: 评估 LETTER Tokenization 质量"
echo "=============================================================="
BEST_CKPT=$(ls /root/autodl-tmp/0330Yambda/LETTER_rqvae/checkpoints/Mar-*/best_entropy_e*.pth 2>/dev/null | tail -1)
if [ -z "$BEST_CKPT" ]; then
    echo "  [WARN] 未找到 entropy checkpoint，跳过评估"
else
    echo "  Best checkpoint: ${BEST_CKPT}"

    # 快速采样评估（10 万条）
    python /root/autodl-tmp/0330Yambda/LETTER_rqvae/evaluate.py \
        --ckpt "${BEST_CKPT}" \
        --data /root/autodl-tmp/0330Yambda/data/embeddings.npz \
        --cf_embeddings /root/autodl-tmp/0330Yambda/data/cf_embeddings.npz \
        --output /root/autodl-tmp/0330Yambda/LETTER_rqvae/eval_results.json \
        --device cuda:0 \
        --sample_size 100000 \
        --nn_sample 5000 \
        --nn_k 20

    # 全量评估（较慢，按需取消注释）
    # python /root/autodl-tmp/0330Yambda/LETTER_rqvae/evaluate.py \
    #     --ckpt "${BEST_CKPT}" \
    #     --data /root/autodl-tmp/0330Yambda/data/embeddings.npz \
    #     --cf_embeddings /root/autodl-tmp/0330Yambda/data/cf_embeddings.npz \
    #     --output /root/autodl-tmp/0330Yambda/LETTER_rqvae/eval_results_full.json \
    #     --device cuda:0
fi

# =============================================================================
# Step 6：导出 Token 序列
# =============================================================================
echo ""
echo "=============================================================="
echo "Step 6: 导出 Token 序列"
echo "=============================================================="
BEST_CKPT=$(ls /root/autodl-tmp/0330Yambda/LETTER_rqvae/checkpoints/Mar-*/best_entropy_e*.pth 2>/dev/null | tail -1)
if [ -z "$BEST_CKPT" ]; then
    echo "  [WARN] 未找到 checkpoint，跳过导出"
else
    python /root/autodl-tmp/0330Yambda/LETTER_rqvae/generate_indices.py \
        --ckpt "${BEST_CKPT}" \
        --data /root/autodl-tmp/0330Yambda/data/embeddings.npz \
        --output /root/autodl-tmp/0330Yambda/data/item_tokens_letter.json \
        --device cuda:0
fi

echo ""
echo "=============================================================="
echo "全部流程完成！"
echo "=============================================================="
