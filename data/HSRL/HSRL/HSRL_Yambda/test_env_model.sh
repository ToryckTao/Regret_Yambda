#!/bin/bash

# 快速测试脚本 - 使用 debug 数据验证整个流程

export PYTHONPATH="/root/autodl-tmp/data/HSRL/HSRL:${PYTHONPATH}"

# 使用 debug 数据
data_path="/root/autodl-tmp/data/HSRL/dataset/processed/"
output_path="/root/autodl-tmp/data/HSRL/HSRL/output/yambda_hsrl/"

mkdir -p ${output_path}env
mkdir -p ${output_path}env/log
mkdir -p ${output_path}agents

# 测试1: 训练环境模型
echo "===== Step 1: Training Environment Model (debug data) ====="
python train_yambda_env.py \
    --model YambdaUserResponse \
    --reader YambdaDataReader \
    --train_file ${data_path}debug_train.tsv \
    --val_file ${data_path}debug_val.tsv \
    --test_file ${data_path}debug_test.tsv \
    --item_meta_file ${data_path}item_meta.tsv \
    --data_separator $'\t' \
    --meta_data_separator $'\t' \
    --loss mse \
    --l2_coef 0.0001 \
    --lr 0.001 \
    --epoch 2 \
    --seed 19 \
    --model_path ${output_path}env/yambda_user_env_lr0.001_reg0.0001.model \
    --max_seq_len 50 \
    --n_worker 0 \
    --feature_dim 16 \
    --hidden_dims 64 \
    --attn_n_head 2 \
    > ${output_path}env/log/yambda_user_env_lr0.001_reg0.0001.model.log

echo "Environment model training done!"
