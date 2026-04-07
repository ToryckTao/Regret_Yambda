#!/bin/bash

# 设置 PYTHONPATH，使 Python 能找到 model、reader、env 等模块
export PYTHONPATH="/root/autodl-tmp/data/HSRL/HSRL:${PYTHONPATH}"

# 绝对路径
data_path="/root/autodl-tmp/data/HSRL/dataset/processed/"
output_path="/root/autodl-tmp/data/HSRL/HSRL/output/yambda_hsrl/"

mkdir -p ${output_path}env

python eval_yambda_env.py \
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
    --feature_dim 16 \
    --hidden_dims 256 \
    --attn_n_head 2 \
    --model_path ${output_path}env/yambda_user_env_lr0.001_reg0.0001.model
