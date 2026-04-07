#!/bin/bash

# 设置 PYTHONPATH，使 Python 能找到 model、reader、env 等模块
export PYTHONPATH="/root/autodl-tmp/data/HSRL/HSRL:${PYTHONPATH}"

mkdir -p output

# Yambda environment

mkdir -p output/yambda_hsrl/
mkdir -p output/yambda_hsrl/env
mkdir -p output/yambda_hsrl/env/log

data_path="/root/autodl-tmp/data/HSRL/dataset/processed/"
output_path="/root/autodl-tmp/data/HSRL/HSRL/output/yambda_hsrl/"

# 只跑一组参数，快速验证代码是否正常工作
REG=0.0001
LR=0.001

python train_yambda_env.py\
    --model YambdaUserResponse\
    --reader YambdaDataReader\
    --train_file ${data_path}train_data.tsv\
    --val_file ${data_path}val_data.tsv\
    --test_file ${data_path}test_data.tsv\
    --item_meta_file ${data_path}item_meta.tsv\
    --data_separator $'\t'\
    --meta_data_separator $'\t'\
    --loss mse\
    --l2_coef ${REG}\
    --lr ${LR}\
    --epoch 2\
    --seed 19\
    --model_path ${output_path}env/yambda_user_env_lr${LR}_reg${REG}.model\
    --max_seq_len 50\
    --n_worker 4\
    --feature_dim 16\
    --hidden_dims 256\
    --attn_n_head 2\
    > ${output_path}env/log/yambda_user_env_lr${LR}_reg${REG}.model.log
