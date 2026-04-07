#!/usr/bin/env bash
set -e

output_path="output/rl4rs/"
log_name="rl4rs_user_env_lr0.0003_reg0.0001"

# Training params (must match training)
N_ITER=50000
TEST_ITER=1000
GAMMA=0.9
TOPK=1
EMPTY=0
MAX_STEP=20
REG=0.00001
NOISE=0.1
EP_BS=32
BS=64
SCORER="SASRec"
CRITIC_LR=0.001
ACTOR_LR=0.0001
BEHAVE_LR=0
INITEP=0
TEMPER_SWEET_POINT=0.9
SEED=7

# Dynamic SAVE_DIR
SAVE_DIR="${output_path}agents/superddpg_${SCORER}_actor${ACTOR_LR}_critic${CRITIC_LR}_behave${BEHAVE_LR}_niter${N_ITER}_reg${REG}_ep${INITEP}_noise${NOISE}_bs${BS}_epbs${EP_BS}_step${MAX_STEP}_seed${SEED}/"

mkdir -p "$SAVE_DIR"

args=(
  --env_class RL4RSEnvironment_GPU
  --policy_class "$SCORER"
  --critic_class GeneralCritic
  --agent_class BehaviorDDPG
  --facade_class OneStageFacade
  --seed "$SEED"
  --cuda 0
  --env_path "${output_path}env/${log_name}.env"
  --urm_log_path "${output_path}env/log/${log_name}.model.log"
  --data_split test
  --max_step_per_episode "$MAX_STEP"
  --initial_temper "$MAX_STEP"
  --temper_sweet_point "$TEMPER_SWEET_POINT"
  --reward_func mean_with_cost
  --sasrec_n_layer 2
  --sasrec_d_model 32
  --sasrec_n_head 4
  --sasrec_dropout 0.1
  --critic_hidden_dims 256 64
  --slate_size 9
  --buffer_size 100000
  --start_timestamp 0
  --noise_var 0
  --empty_start_rate "$EMPTY"
  --save_path "${SAVE_DIR}model"
  --episode_batch_size "$EP_BS"
  --batch_size "$BS"
  --actor_lr "$ACTOR_LR"
  --critic_lr "$CRITIC_LR"
  --behavior_lr "$BEHAVE_LR"
  --actor_decay "$REG"
  --critic_decay "$REG"
  --behavior_decay "$REG"
  --target_mitigate_coef 0.01
  --gamma "$GAMMA"
  --n_iter "$TEST_ITER"
  --initial_greedy_epsilon 0
  --final_greedy_epsilon 0
  --elbow_greedy 0
  --check_episode 10
  --topk_rate "$TOPK"
)

python test.py "${args[@]}" > "${SAVE_DIR}test_log"
echo "DONE. See:"
echo "  Log:    ${SAVE_DIR}test_log"
echo "  Report: ${SAVE_DIR}model.report"
