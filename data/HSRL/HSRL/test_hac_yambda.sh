#!/usr/bin/env bash
set -e

output_path="output/yambda_hac/"
log_name="yambda_user_env_lr0.001_reg0.0001"

# Training params (must match training)
N_ITER=10000
TEST_ITER=1000
GAMMA=0.9
TOPK=1
EMPTY=0
MAX_STEP=20
REG=0.00001
NOISE=0.1
EP_BS=32
BS=128
SCORER="SASRec"
CRITIC_LR=0.001
ACTOR_LR=0.0001
BEHAVE_LR=0.00001
HA_COEF=0.1
SEED=7

# Dynamic SAVE_DIR
SAVE_DIR="${output_path}agents/hac_${SCORER}_actor${ACTOR_LR}_critic${CRITIC_LR}_behave${BEHAVE_LR}_hacoef${HA_COEF}_niter${N_ITER}_reg${REG}_ep0_noise${NOISE}_bs${BS}_epbs${EP_BS}_step${MAX_STEP}_topk${TOPK}_seed${SEED}/"

mkdir -p "$SAVE_DIR"

args=(
  --env_class YambdaEnvironment_GPU_HAC
  --policy_class "$SCORER"
  --critic_class GeneralCritic
  --agent_class HAC
  --facade_class OneStageFacade_HyperAction
  --seed "$SEED"
  --cuda 0
  --env_path "${output_path}env/${log_name}.env"
  --urm_log_path "${output_path}env/log/${log_name}.model.log"
  --data_split test
  --max_step_per_episode "$MAX_STEP"
  --initial_temper "$MAX_STEP"
  --temper_sweet_point 0.9
  --temper_prob_lag 100
  --reward_func mean_with_cost
  --sasrec_n_layer 2
  --sasrec_d_model 32
  --sasrec_d_forward 64
  --sasrec_n_head 4
  --sasrec_dropout 0.1
  --critic_hidden_dims 256 64
  --critic_dropout_rate 0.2
  --slate_size 1
  --buffer_size 100000
  --start_timestamp 2000
  --noise_var 0.1
  --empty_start_rate "$EMPTY"
  --train_every_n_step 5
  --save_path "${SAVE_DIR}model"
  --episode_batch_size "$EP_BS"
  --batch_size "$BS"
  --actor_lr "$ACTOR_LR"
  --critic_lr "$CRITIC_LR"
  --behavior_lr "$BEHAVE_LR"
  --hyper_actor_coef "$HA_COEF"
  --actor_decay "$REG"
  --critic_decay "$REG"
  --behavior_decay "$REG"
  --target_mitigate_coef 0.01
  --gamma "$GAMMA"
  --n_iter "$TEST_ITER"
  --initial_greedy_epsilon 0
  --final_greedy_epsilon 0
  --elbow_greedy 0.5
  --check_episode 10
  --topk_rate "$TOPK"
)

python test.py "${args[@]}" > "${SAVE_DIR}test_log"
echo "DONE. See:"
echo "  Log:    ${SAVE_DIR}test_log"
echo "  Report: ${SAVE_DIR}model.report"
