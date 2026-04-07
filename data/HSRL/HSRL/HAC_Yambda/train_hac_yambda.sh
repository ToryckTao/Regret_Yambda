#!/bin/bash

mkdir -p output

# Yambda environment with HAC agent

mkdir -p output/yambda_hac/
mkdir -p output/yambda_hac/env/
mkdir -p output/yambda_hac/env/log/
mkdir -p output/yambda_hac/agents/

output_path="output/yambda_hac/"
# Use existing Yambda environment model
log_name="yambda_user_env_lr0.001_reg0.0001"

# Check if environment model log exists
source_path="../output/yambda_hsrl/env"
if [ ! -f "${source_path}/log/${log_name}.model.log" ]; then
    echo "Error: Environment model log not found: ${source_path}/log/${log_name}.model.log"
    echo "Please run run_yambda_env.sh first."
    exit 1
fi

# Copy files from source instead of using symlinks (more reliable)
cp -f ${source_path}/log/${log_name}.model.log output/yambda_hac/env/log/
cp -f ${source_path}/${log_name}.env output/yambda_hac/env/ 2>/dev/null || true


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
HA_COEF=0.1

for NOISE in 0.1
do
    for TOPK in 1
    do
        for HA_COEF in 0.1
        do
            for BEHAVE_LR in 0.00001
            do
                for ACTOR_LR in 0.0001
                do
                    for SEED in 7
                    do
                        mkdir -p ${output_path}agents/hac_${SCORER}_actor${ACTOR_LR}_critic${CRITIC_LR}_behave${BEHAVE_LR}_hacoef${HA_COEF}_niter${N_ITER}_reg${REG}_ep${INITEP}_noise${NOISE}_bs${BS}_epbs${EP_BS}_step${MAX_STEP}_topk${TOPK}_seed${SEED}/

                        python train_hac_yambda.py\
                            --env_class YambdaEnvironment_GPU_HAC\
                            --policy_class ${SCORER}\
                            --critic_class GeneralCritic\
                            --agent_class HAC\
                            --facade_class OneStageFacade_HyperAction\
                            --seed ${SEED}\
                            --cuda 0\
                            --env_path ${output_path}env/${log_name}.env\
                            --max_step_per_episode ${MAX_STEP}\
                            --initial_temper ${MAX_STEP}\
                            --temper_sweet_point ${TEMPER_SWEET_POINT}\
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
                            --save_path ${output_path}agents/hac_${SCORER}_actor${ACTOR_LR}_critic${CRITIC_LR}_behave${BEHAVE_LR}_hacoef${HA_COEF}_niter${N_ITER}_reg${REG}_ep${INITEP}_noise${NOISE}_bs${BS}_epbs${EP_BS}_step${MAX_STEP}_topk${TOPK}_seed${SEED}/model\
                            --episode_batch_size ${EP_BS}\
                            --batch_size ${BS}\
                            --actor_lr ${ACTOR_LR}\
                            --critic_lr ${CRITIC_LR}\
                            --behavior_lr ${BEHAVE_LR}\
                            --hyper_actor_coef ${HA_COEF}\
                            --actor_decay ${REG}\
                            --critic_decay ${REG}\
                            --behavior_decay ${REG}\
                            --target_mitigate_coef 0.01\
                            --gamma ${GAMMA}\
                            --n_iter ${N_ITER}\
                            --initial_greedy_epsilon ${INITEP}\
                            --final_greedy_epsilon ${INITEP}\
                            --elbow_greedy ${ELBOW}\
                            --check_episode 10\
                            --topk_rate ${TOPK}\
                            > ${output_path}agents/hac_${SCORER}_actor${ACTOR_LR}_critic${CRITIC_LR}_behave${BEHAVE_LR}_hacoef${HA_COEF}_niter${N_ITER}_reg${REG}_ep${INITEP}_noise${NOISE}_bs${BS}_epbs${EP_BS}_step${MAX_STEP}_topk${TOPK}_seed${SEED}/log
                    done
                done
            done
        done
    done
done
