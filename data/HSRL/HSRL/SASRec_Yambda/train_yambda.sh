#!/bin/bash

output_path="/root/autodl-tmp/data/HSRL/HSRL/SASRec_Yambda/output"
mkdir -p ${output_path}
mkdir -p ${output_path}/env
mkdir -p ${output_path}/env/log
mkdir -p ${output_path}/agents

# Use existing Yambda environment model from HSRL_Yambda
log_name="yambda_user_env_lr0.001_reg0.0001"
ENV_PATH="/root/autodl-tmp/data/HSRL/HSRL/output/yambda_hsrl/env/${log_name}.env"
URM_LOG_PATH="/root/autodl-tmp/data/HSRL/HSRL/output/yambda_hsrl/env/log/${log_name}.model.log"
ENV_CHECKPOINT_PATH="/root/autodl-tmp/data/HSRL/HSRL/output/yambda_hsrl/env/${log_name}.model.checkpoint"

if [ ! -f "${ENV_PATH}" ]; then
    echo "Error: Environment not found at ${ENV_PATH}"
    echo "Please run run_yambda_env.sh first."
    exit 1
fi

cp -f ${URM_LOG_PATH} ${output_path}/env/log/
cp -f ${ENV_PATH} ${output_path}/env/ 2>/dev/null || true

# Default parameters
SEED=7
ACTOR_LR=0.0001
CRITIC_LR=0.001
NITER=10000
BATCH_SIZE=128
EPISODE_BATCH_SIZE=32
STEP=20

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --seed)
            SEED="$2"
            shift 2
            ;;
        --actor_lr)
            ACTOR_LR="$2"
            shift 2
            ;;
        --critic_lr)
            CRITIC_LR="$2"
            shift 2
            ;;
        --niter)
            NITER="$2"
            shift 2
            ;;
        --bs)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epbs)
            EPISODE_BATCH_SIZE="$2"
            shift 2
            ;;
        --step)
            STEP="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

SAVE_PATH="${output_path}/agents/sasrec_actor${ACTOR_LR}_critic${CRITIC_LR}_niter${NITER}_reg0.00001_ep0_noise0.1_bs${BATCH_SIZE}_epbs${EPISODE_BATCH_SIZE}_step${STEP}_seed${SEED}/model"

echo "======================================"
echo "SASRec Training on Yambda Environment"
echo "======================================"
echo "Seed: ${SEED}"
echo "Actor LR: ${ACTOR_LR}"
echo "Critic LR: ${CRITIC_LR}"
echo "N_iter: ${NITER}"
echo "Batch size: ${BATCH_SIZE}"
echo "Episode batch size: ${EPISODE_BATCH_SIZE}"
echo "Max step per episode: ${STEP}"
echo "Save path: ${SAVE_PATH}"
echo "======================================"

python train_yambda.py \
    --env_class YambdaEnvironment_GPU \
    --policy_class SASRec \
    --critic_class GeneralCritic \
    --agent_class DDPG \
    --facade_class OneStageFacade \
    --seed ${SEED} \
    --cuda 0 \
    --env_path ${ENV_PATH} \
    --reward_func mean_with_cost \
    --max_step_per_episode ${STEP} \
    --initial_temper 20 \
    --urm_log_path ${URM_LOG_PATH} \
    --temper_sweet_point 0.9 \
    --temper_prob_lag 100 \
    --sasrec_n_layer 2 \
    --sasrec_d_model 32 \
    --sasrec_d_forward 64 \
    --sasrec_n_head 4 \
    --sasrec_dropout 0.1 \
    --critic_hidden_dims 256 64 \
    --critic_dropout_rate 0.2 \
    --gamma 0.9 \
    --n_iter ${NITER} \
    --train_every_n_step 5 \
    --initial_greedy_epsilon 0.0 \
    --final_greedy_epsilon 0.0 \
    --elbow_greedy 0.5 \
    --check_episode 10 \
    --with_eval False \
    --save_path ${SAVE_PATH} \
    --episode_batch_size ${EPISODE_BATCH_SIZE} \
    --batch_size ${BATCH_SIZE} \
    --actor_lr ${ACTOR_LR} \
    --critic_lr ${CRITIC_LR} \
    --actor_decay 0.00001 \
    --critic_decay 0.00001 \
    --target_mitigate_coef 0.01
