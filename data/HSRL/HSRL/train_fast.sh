#!/bin/bash
# 快速训练脚本 - 使用50万条数据

mkdir -p output

# Yambda environment
mkdir -p output/yambda_hsrl/
mkdir -p output/yambda_hsrl/env
mkdir -p output/yambda_hsrl/env/log

data_path="/root/autodl-tmp/data/HSRL/dataset/processed/"
output_path="/root/autodl-tmp/data/HSRL/HSRL/output/yambda_hsrl/"

# 创建50万条的小数据集
if [ ! -f "${data_path}train_data_500k.tsv" ]; then
    echo "Creating 500k dataset..."
    head -n 1 "${data_path}train_data.tsv" > "${data_path}train_data_500k.tsv"
    sed -n '2,500001p' "${data_path}train_data.tsv" >> "${data_path}train_data_500k.tsv"
    echo "Created train_data_500k.tsv"
fi

# 直接运行python，不通过shell变量
cd /root/autodl-tmp/data/HSRL/HSRL

python train_yambda_env.py \
    --model YambdaUserResponse \
    --reader RL4RSDataReader \
    --train_file "$data_path"train_data_500k.tsv \
    --val_file "$data_path"train_data_500k.tsv \
    --item_meta_file "$data_path"item_meta.tsv \
    --data_separator $'\t' \
    --meta_data_separator $'\t' \
    --loss mse \
    --l2_coef 0.0001 \
    --lr 0.001 \
    --epoch 10 \
    --batch_size 512 \
    --seed 19 \
    --model_path "$output_path"env/yambda_user_env.model \
    --max_seq_len 50 \
    --n_worker 4 \
    --feature_dim 16 \
    --hidden_dims 256 \
    --attn_n_head 2 \
    2>&1 | tee "$output_path"env/log/train.log
