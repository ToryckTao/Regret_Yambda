#!/bin/bash

mkdir -p output

# SARA Yambda environment

mkdir -p output/yambda_sara/
mkdir -p output/yambda_sara/env/
mkdir -p output/yambda_sara/env/log/
mkdir -p output/yambda_sara/agents/

# 添加 PYTHONPATH - 需要指向 HSRL 目录
export PYTHONPATH="/root/autodl-tmp/data/HSRL/HSRL:${PYTHONPATH}"

output_path="output/yambda_sara/"
log_name="yambda_user_env_lr0.001_reg0.0001"

# Check if environment model exists before running
log_name="yambda_user_env_lr0.001_reg0.0001"
if [ ! -f "output/yambda_hsrl/env/log/${log_name}.model.log" ]; then
    echo "Error: Environment model log not found: output/yambda_hsrl/env/log/${log_name}.model.log"
    exit 1
fi

if [ ! -f "output/yambda_hsrl/env/${log_name}.model.checkpoint" ]; then
    echo "Error: Environment model checkpoint not found: output/yambda_hsrl/env/${log_name}.model.checkpoint"
    exit 1
fi

# 使用已有的环境模型（来自 HSRL）
ENV_PATH="output/yambda_hsrl/env/${log_name}.env"
URM_LOG_PATH="output/yambda_hsrl/env/log/${log_name}.model.log"
ENV_CHECKPOINT_PATH="output/yambda_hsrl/env/${log_name}.model.checkpoint"

# 如果 .env 文件不存在但 .checkpoint 存在，创建一个空的 .env 文件作为占位
if [ ! -f "${ENV_PATH}" ] && [ -f "${ENV_CHECKPOINT_PATH}" ]; then
    touch "${ENV_PATH}"
fi

N_ITER=10000
CONTINUE_ITER=0
# CONTINUE_ITER: 从哪个 checkpoint 继续训练（0=不继续，从头开始）
# 如果设置了 > 0，会跳过数据准备阶段，直接从模型 checkpoint 继续
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
SCORER="SIDPolicy_credit"
CRITIC="Token_Critic"
FACADE="SARAFacade_credit"
AGENT="BehaviorDDPG"
CRITIC_LR=0.001
ACTOR_LR=0.0001
BEHAVE_LR=0.00001
TEMPER_SWEET_POINT=0.9

# SARA 基础版参数（不带惩罚）
SID_PATH="dataset/yambda_item2sid.pkl"

# 奖励函数权重
OMEGA_LISTEN=1.0
OMEGA_LIKE=2.0
OMEGA_DISLIKE=1.0

# 惩罚函数权重
LAMBDA_UNLIKE=1.0
LAMBDA_UNDISLIKE=0.5

# 后悔池参数
REGRET_POOL_SIZE=20
REGRET_PENALTY_WEIGHT=0.5
REGRET_LAYER_WEIGHTS="0.05,0.25,0.7"
REGRET_GAMMA=0.9

for NOISE in 0.1
do
    for REG in 0.00001
    do
        for INITEP in 0
        do
            for CRITIC_LR in 0.001
            do
                for ACTOR_LR in 0.0001
                do
                    for SEED in 7
                    do
                        mkdir -p ${output_path}agents/sara_${SCORER}_actor${ACTOR_LR}_critic${CRITIC_LR}_behave${BEHAVE_LR}_niter${N_ITER}_reg${REG}_ep${INITEP}_noise${NOISE}_bs${BS}_epbs${EP_BS}_step${MAX_STEP}_seed${SEED}/

                        python SARA_Yambda/train_sara.py\
                            --env_class SARAEnvironment_GPU\
                            --policy_class ${SCORER}\
                            --critic_class ${CRITIC}\
                            --agent_class ${AGENT}\
                            --facade_class ${FACADE}\
                            --seed ${SEED}\
                            --cuda 0\
                            --env_path ${ENV_PATH}\
                            --urm_log_path ${URM_LOG_PATH}\
                            --max_step_per_episode ${MAX_STEP}\
                            --initial_temper ${MAX_STEP}\
                            --reward_func direct_score\
                            --sasrec_n_layer 2\
                            --sasrec_d_model 32\
                            --sasrec_n_head 4\
                            --sasrec_dropout 0.1\
                            --sid_levels 3\
                            --sid_vocab_sizes 256\
                            --sid_temp 1.0\
                            --critic_hidden_dims 256 64\
                            --slate_size 9\
                            --buffer_size 100000\
                            --start_timestamp 2000\
                            --noise_var ${NOISE}\
                            --empty_start_rate ${EMPTY}\
                            --train_every_n_step 5\
                            --item2sid ${SID_PATH}\
                            --save_path ${output_path}agents/sara_${SCORER}_actor${ACTOR_LR}_critic${CRITIC_LR}_behave${BEHAVE_LR}_niter${N_ITER}_reg${REG}_ep${INITEP}_noise${NOISE}_bs${BS}_epbs${EP_BS}_step${MAX_STEP}_seed${SEED}/model\
                            --episode_batch_size ${EP_BS}\
                            --batch_size ${BS}\
                            --actor_lr ${ACTOR_LR}\
                            --critic_lr ${CRITIC_LR}\
                            --behavior_lr ${BEHAVE_LR}\
                            --behavior_decay ${REG}\
                            --advantage_bias 0\
                            --entropy_coef 0.1\
                            --target_mitigate_coef 0.01\
                            --gamma ${GAMMA}\
                            --n_iter ${N_ITER}\
                            --continue_iter ${CONTINUE_ITER}\
                            --initial_greedy_epsilon ${INITEP}\
                            --final_greedy_epsilon ${INITEP}\
                            --elbow_greedy ${ELBOW}\
                            --check_episode 10\
                            --topk_rate ${TOPK}\
                            --omega_listen ${OMEGA_LISTEN}\
                            --omega_like ${OMEGA_LIKE}\
                            --omega_dislike ${OMEGA_DISLIKE}\
                            --lambda_unlike ${LAMBDA_UNLIKE}\
                            --lambda_undislike ${LAMBDA_UNDISLIKE}\
                            --regret_pool_size ${REGRET_POOL_SIZE}\
                            --regret_penalty_weight ${REGRET_PENALTY_WEIGHT}\
                            --regret_layer_weights ${REGRET_LAYER_WEIGHTS}\
                            --regret_pool_gamma ${REGRET_GAMMA}\
                            --regret_pool_init_path dataset/regret_pool_init.pkl\
                            --sara_eta 0.5\
                            --sara_layer_weights 0.05,0.25,0.70\
                            > ${output_path}agents/sara_${SCORER}_actor${ACTOR_LR}_critic${CRITIC_LR}_behave${BEHAVE_LR}_niter${N_ITER}_reg${REG}_ep${INITEP}_noise${NOISE}_bs${BS}_epbs${EP_BS}_step${MAX_STEP}_seed${SEED}/log
                        done
                    done
                done
            done
        done
    done
