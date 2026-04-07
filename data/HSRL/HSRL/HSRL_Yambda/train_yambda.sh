#!/bin/bash

# 设置 PYTHONPATH，使 Python 能找到 model、reader、env 等模块
export PYTHONPATH="/root/autodl-tmp/data/HSRL/HSRL:${PYTHONPATH}"

# 修复 OpenMP 线程数设置
unset OMP_NUM_THREADS

# CUDA 内存优化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 实际输出路径（根据已有的 env 文件位置）
output_path="/root/autodl-tmp/data/HSRL/HSRL/output/yambda_hsrl/"
log_name="yambda_user_env_full"

mkdir -p ${output_path}
mkdir -p ${output_path}env/
mkdir -p ${output_path}env/log/
mkdir -p ${output_path}agents/

# Check if environment model exists before running
if [ ! -f "${output_path}env/${log_name}.env" ]; then
    echo "Error: Environment model not found: ${output_path}env/${log_name}.env"
    echo "Please run train_env.py or train_yambda_env.sh first."
    exit 1
fi

if [ ! -f "${output_path}env/log/${log_name}.model.log" ]; then
    echo "Error: Environment model log not found: ${output_path}env/log/${log_name}.model.log"
    exit 1
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
BS=64
SEED=7
SCORER="SASRec"
CRITIC_LR=0.001
ACTOR_LR=0.0001
BEHAVE_LR=0.00001
TEMPER_SWEET_POINT=0.9

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
                        mkdir -p ${output_path}agents/superddpg_${SCORER}_actor${ACTOR_LR}_critic${CRITIC_LR}_behave${BEHAVE_LR}_niter${N_ITER}_reg${REG}_ep${INITEP}_noise${NOISE}_bs${BS}_epbs${EP_BS}_step${MAX_STEP}_seed${SEED}/

                        python train_yambda.py\
                            --env_class YambdaEnvironment_GPU\
                            --policy_class ${SCORER}\
                            --critic_class GeneralCritic\
                            --agent_class DDPG\
                            --facade_class OneStageFacade\
                            --seed ${SEED}\
                            --cuda 0\
                            --env_path ${output_path}env/${log_name}.env\
                            --max_step_per_episode ${MAX_STEP}\
                            --initial_temper ${MAX_STEP}\
                            --reward_func direct_score\
                            --urm_log_path ${output_path}env/log/${log_name}.model.log\
                            --sasrec_n_layer 2\
                            --sasrec_d_model 32\
                            --sasrec_n_head 4\
                            --sasrec_dropout 0.1\
                            --critic_hidden_dims 256 64\
                            --slate_size 9\
                            --buffer_size 100000\
                            --start_timestamp 2000\
                            --noise_var ${NOISE}\
                            --empty_start_rate ${EMPTY}\
                            --train_every_n_step 5\
                            --save_path ${output_path}agents/yambda_${SCORER}_actor${ACTOR_LR}_critic${CRITIC_LR}_behave${BEHAVE_LR}_niter${N_ITER}_reg${REG}_ep${INITEP}_noise${NOISE}_bs${BS}_epbs${EP_BS}_step${MAX_STEP}_seed${SEED}/model\
                            --episode_batch_size ${EP_BS}\
                            --batch_size ${BS}\
                            --actor_lr ${ACTOR_LR}\
                            --critic_lr ${CRITIC_LR}\
                            --behavior_lr ${BEHAVE_LR}\
                            --actor_decay ${REG}\
                            --critic_decay ${REG}\
                            --behavior_decay ${REG}\
                            --target_mitigate_coef 0.01\
                            --entropy_coef 0.01\
                            --bc_coef 0.1\
                            --gamma ${GAMMA}\
                            --n_iter ${N_ITER}\
                            --initial_greedy_epsilon ${INITEP}\
                            --final_greedy_epsilon ${INITEP}\
                            --elbow_greedy ${ELBOW}\
                            --check_episode 10\
                            --use_wandb \
                            --wandb_project yambda_hsrl \
                            --wandb_name ${log_name}\
                            --topk_rate ${TOPK}\
                            > ${output_path}agents/superddpg_${SCORER}_actor${ACTOR_LR}_critic${CRITIC_LR}_behave${BEHAVE_LR}_niter${N_ITER}_reg${REG}_ep${INITEP}_noise${NOISE}_bs${BS}_epbs${EP_BS}_step${MAX_STEP}_seed${SEED}/log
                    done
                done
            done
        done
    done
done
