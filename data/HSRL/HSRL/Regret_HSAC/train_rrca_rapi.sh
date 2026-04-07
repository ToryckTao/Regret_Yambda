#!/bin/bash

# RRCA + RAPI 训练脚本
# 基于 HSRL_Yambda/train_yambda.sh 修改

export PYTHONPATH="/root/autodl-tmp/data/HSRL/HSRL:${PYTHONPATH}"

# 输出路径
output_path="/root/autodl-tmp/data/HSRL/HSRL/output/rrca_rapi/"
log_name="yambda_user_env_lr0.001_reg0.0001"

mkdir -p ${output_path}
mkdir -p ${output_path}env/
mkdir -p ${output_path}agents/

# 使用已有的环境模型（来自 HSRL）
ENV_PATH="/root/autodl-tmp/data/HSRL/HSRL/output/yambda_hsrl/env/${log_name}.env"
URM_LOG_PATH="/root/autodl-tmp/data/HSRL/HSRL/output/yambda_hsrl/env/log/${log_name}.model.log"

# 如果 .env 文件不存在但 .checkpoint 存在，创建一个空的 .env 文件作为占位
if [ ! -f "${ENV_PATH}" ]; then
    touch "${ENV_PATH}"
fi

N_ITER=10000
CONTINUE_ITER=0
GAMMA=0.9
TOPK=1
EMPTY=0

MAX_STEP=20
INITEP=0
REG=0.00001
NOISE=0.1
ELBOW=0.5
EP_BS=32
BS=128
SEED=7
SCORER="SASRec"
CRITIC_LR=0.001
ACTOR_LR=0.0001
BEHAVE_LR=0.00001
TEMPER_SWEET_POINT=0.9

# RRCA + RAPI 参数
RRCA_LAMBDA_UNLIKE=1.0
RRCA_LAMBDA_UNDISLIKE=0.5
RRCA_REGRET_GAMMA=0.9
REGRET_POOL_SIZE=10
REGRET_ETA=1.0
REGRET_W="0.05,0.25,0.7"

# 模型参数
SID_LEVELS=3
CODEBOOK_SIZE=16
EMBED_DIM=64

mkdir -p ${output_path}agents/rrca_rapi_${SCORER}_actor${ACTOR_LR}_critic${CRITIC_LR}_behave${BEHAVE_LR}_niter${N_ITER}_reg${REG}_ep${INITEP}_noise${NOISE}_bs${BS}_epbs${EP_BS}_step${MAX_STEP}_seed${SEED}/

# 运行训练
python Regret_HSAC/train_rrca_rapi.py \
    --env_class SARAEnvironment_GPU \
    --policy_class SIDPolicy_credit \
    --critic_class Token_Critic \
    --agent_class SARA_Session_Agent \
    --facade_class SARAFacade_credit \
    --seed ${SEED} \
    --cuda 0 \
    --env_path ${ENV_PATH} \
    --max_step_per_episode ${MAX_STEP} \
    --initial_temper ${MAX_STEP} \
    --reward_func direct_score \
    --urm_log_path ${URM_LOG_PATH} \
    --sasrec_n_layer 2 \
    --sasrec_d_model 32 \
    --sasrec_n_head 4 \
    --sasrec_dropout 0.1 \
    --critic_hidden_dims 256 64 \
    --slate_size 9 \
    --buffer_size 100000 \
    --start_timestamp 2000 \
    --noise_var ${NOISE} \
    --empty_start_rate ${EMPTY} \
    --train_every_n_step 5 \
    --save_path ${output_path}agents/rrca_rapi_${SCORER}_actor${ACTOR_LR}_critic${CRITIC_LR}_behave${BEHAVE_LR}_niter${N_ITER}_reg${REG}_ep${INITEP}_noise${NOISE}_bs${BS}_epbs${EP_BS}_step${MAX_STEP}_seed${SEED}/model \
    --episode_batch_size ${EP_BS} \
    --batch_size ${BS} \
    --actor_lr ${ACTOR_LR} \
    --critic_lr ${CRITIC_LR} \
    --behavior_lr ${BEHAVE_LR} \
    --actor_decay ${REG} \
    --critic_decay ${REG} \
    --behavior_decay ${REG} \
    --target_mitigate_coef 0.01 \
    --gamma ${GAMMA} \
    --n_iter ${N_ITER} \
    --initial_greedy_epsilon ${INITEP} \
    --final_greedy_epsilon ${INITEP} \
    --elbow_greedy ${ELBOW} \
    --check_episode 10 \
    --topk_rate ${TOPK} \
    --codebook_size ${CODEBOOK_SIZE} \
    --embed_dim ${EMBED_DIM} \
    --sid_levels ${SID_LEVELS} \
    --sid_temp 1.0 \
    --d_model 64 \
    --n_head 4 \
    --d_ff 128 \
    --lambda_unlike ${LAMBDA_UNLIKE} \
    --lambda_undislike ${LAMBDA_UNDISLIKE} \
    --regret_gamma ${REGRET_GAMMA} \
    --regret_pool_size ${REGRET_POOL_SIZE} \
    --regret_eta ${REGRET_ETA} \
    --regret_W ${REGRET_W} \
    --session_max_steps ${MAX_STEP} \
    --token_lr 0.001 \
    > ${output_path}agents/rrca_rapi_${SCORER}_actor${ACTOR_LR}_critic${CRITIC_LR}_behave${BEHAVE_LR}_niter${N_ITER}_reg${REG}_ep${INITEP}_noise${NOISE}_bs${BS}_epbs${EP_BS}_step${MAX_STEP}_seed${SEED}/log
