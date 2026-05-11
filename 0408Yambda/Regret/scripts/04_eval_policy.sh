#!/usr/bin/env bash
# 用途：评估策略。比较 base 生成和 RAPI 介入后的 simulator rollout reward、depth、negative rate。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGRET_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REGRET_ROOT}/configs/main.env"
cd "$REGRET_ROOT"
mkdir -p artifacts/evals artifacts/logs

SPLIT="${SPLIT:-test}"
EPISODES="${ROLLOUT_EPISODES:-1000}"
ACTOR_CHECKPOINT="${ACTOR_CHECKPOINT:-$POLICY_ACTOR}"
SAVE_META="${SAVE_META:-${REGRET_ROOT}/artifacts/evals/${MAIN_ID}_eta010_${SPLIT}_${EPISODES}.meta.json}"
LOG_PATH="${LOG_PATH:-${REGRET_ROOT}/artifacts/logs/${MAIN_ID}_eta010_${SPLIT}_${EPISODES}.log}"

if [[ -z "${OMP_NUM_THREADS:-}" || "${OMP_NUM_THREADS}" -lt 1 ]]; then
  export OMP_NUM_THREADS=1
fi

python3 -u scripts/09_eval_simulator_rollout.py \
  --transition_root "$TRANSITION_ROOT" \
  --split "$SPLIT" \
  --item_features_npy "$DENSE_ITEM_FEATURES_NPY" \
  --dense_item2sid_npy "$DENSE_ITEM2SID_NPY" \
  --actor_checkpoint "$ACTOR_CHECKPOINT" \
  --simulator_checkpoint "$SIMULATOR_CHECKPOINT" \
  --save_meta "$SAVE_META" \
  --device "${DEVICE:-cuda}" \
  --max_seq_len "${SID_MAX_SEQ_LEN:-50}" \
  --num_episodes "$EPISODES" \
  --batch_size "${ROLLOUT_BATCH_SIZE:-64}" \
  --read_batch_size "${ROLLOUT_READ_BATCH_SIZE:-2048}" \
  --max_steps "$ROLLOUT_MAX_STEPS" \
  --sample_response \
  --negative_patience "${ROLLOUT_NEGATIVE_PATIENCE:-5}" \
  --decode_top_k "$ROLLOUT_DECODE_TOP_K" \
  --action_mode "${ROLLOUT_ACTION_MODE:-sample}" \
  --action_temperature "${ROLLOUT_ACTION_TEMPERATURE:-1.0}" \
  --sara_eta "$SARA_ETA" \
  --sara_layer_weights "${SARA_LAYER_WEIGHTS:-0.05,0.25,0.70}" \
  --regret_pool_size "${REGRET_POOL_SIZE:-20}" \
  --regret_gamma "${REGRET_GAMMA:-0.9}" \
  --regret_phi_scale "${REGRET_PHI_SCALE:-1.0}" \
  --regret_phi_clip "${REGRET_PHI_CLIP:-2.0}" \
  --regret_reward_threshold "${REGRET_REWARD_THRESHOLD:-0.0}" \
  "$@" \
  2>&1 | tee "$LOG_PATH"
