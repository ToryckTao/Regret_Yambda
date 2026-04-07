#!/bin/bash

# 使用完整数据训练环境模型

export PYTHONPATH="/root/autodl-tmp/data/HSRL/HSRL:${PYTHONPATH}"

data_path="/root/autodl-tmp/data/HSRL/dataset/processed/"
output_path="/root/autodl-tmp/data/HSRL/HSRL/output/yambda_hsrl/"

mkdir -p ${output_path}env
mkdir -p ${output_path}env/log

echo "===== Training Environment Model with FULL DATA ====="

python train_yambda_env.py \
    --model YambdaUserResponse \
    --reader YambdaDataReader \
    --train_file ${data_path}train_data.tsv \
    --val_file ${data_path}val_data.tsv \
    --test_file ${data_path}test_data.tsv \
    --item_meta_file ${data_path}item_meta.tsv \
    --data_separator $'\t' \
    --meta_data_separator $'\t' \
    --loss mse \
    --l2_coef 0.0001 \
    --lr 0.001 \
    --epoch 10 \
    --seed 19 \
    --model_path ${output_path}env/yambda_user_env_full.model \
    --max_seq_len 50 \
    --n_worker 4 \
    --feature_dim 64 \
    --hidden_dims 256 \
    --attn_n_head 4 \
    > ${output_path}env/log/yambda_user_env_full.model.log 2>&1

echo "Environment model training done!"
