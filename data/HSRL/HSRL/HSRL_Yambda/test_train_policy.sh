#!/bin/bash

# 快速测试脚本 - 训练策略

export PYTHONPATH="/root/autodl-tmp/data/HSRL/HSRL:${PYTHONPATH}"

# 使用 debug 数据训练的环境模型
data_path="/root/autodl-tmp/data/HSRL/dataset/processed/"
output_path="/root/autodl-tmp/data/HSRL/HSRL/output/yambda_hsrl/"
log_name="yambda_user_env_lr0.001_reg0.0001"

# 创建空的 .env 文件（占位符）
touch ${output_path}env/${log_name}.env

mkdir -p ${output_path}agents/test_ddpg_sasrec

# 测试2: 训练策略模型
echo "===== Step 2: Training Policy (DDPG + SASRec) ====="
python train_yambda.py \
    --env_class YambdaEnvironment_GPU \
    --policy_class SASRec \
    --critic_class GeneralCritic \
    --agent_class DDPG \
    --facade_class OneStageFacade \
    --seed 7 \
    --cuda 0 \
    --env_path ${output_path}env/${log_name}.env \
    --max_step_per_episode 5 \
    --initial_temper 5 \
    --reward_func direct_score \
    --urm_log_path ${output_path}env/log/${log_name}.model.log \
    --sasrec_n_layer 1 \
    --sasrec_d_model 16 \
    --sasrec_n_head 2 \
    --sasrec_dropout 0.1 \
    --critic_hidden_dims 32 16 \
    --slate_size 3 \
    --buffer_size 1000 \
    --start_timestamp 50 \
    --noise_var 0.1 \
    --train_every_n_step 5 \
    --n_iter 10 \
    --batch_size 8 \
    --episode_batch_size 8 \
    --actor_lr 0.001 \
    --critic_lr 0.001 \
    --actor_decay 0.00001 \
    --critic_decay 0.00001 \
    --gamma 0.9 \
    --target_mitigate_coef 0.01 \
    --save_path ${output_path}agents/test_ddpg_sasrec/model \
    > ${output_path}agents/test_ddpg_sasrec/log 2>&1

echo "Policy training done!"
