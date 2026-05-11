#!/usr/bin/env bash
# 用途：训练主线策略。流程是 HPN 生成语义 item，simulator 给反馈，RRCA 修正有效 reward，再更新 actor/critic。
# 默认输出 artifacts/models/policy_hsac_main_*；如需另存，设置 POLICY_SAVE_PREFIX。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGRET_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REGRET_ROOT}/configs/main.env"
cd "$REGRET_ROOT"
mkdir -p artifacts/logs artifacts/models

python3 -u scripts/10_train_hsac_simulator_rollout.py \
  --transition_root "$TRANSITION_ROOT" \
  --split "${SPLIT:-train}" \
  --item_features_npy "$DENSE_ITEM_FEATURES_NPY" \
  --dense_item2sid_npy "$DENSE_ITEM2SID_NPY" \
  --actor_init_checkpoint "$ACTOR_INIT_CHECKPOINT" \
  --simulator_checkpoint "$SIMULATOR_CHECKPOINT" \
  --save_prefix "$POLICY_SAVE_PREFIX" \
  --save_meta "$POLICY_SAVE_META" \
  --device "${DEVICE:-cuda}" \
  --episodes "$HSAC_EPISODES" \
  --batch_size "$HSAC_BATCH_SIZE" \
  --read_batch_size "$HSAC_READ_BATCH_SIZE" \
  --epochs "$HSAC_EPOCHS" \
  --max_steps "$HSAC_MAX_STEPS" \
  --actor_lr "$HSAC_ACTOR_LR" \
  --critic_lr "$HSAC_CRITIC_LR" \
  --gamma "$HSAC_GAMMA" \
  --target_tau "${HSAC_TARGET_TAU:-0.05}" \
  --entropy_weight "${HSAC_ENTROPY_WEIGHT:-0.001}" \
  --decode_top_k "$HSAC_DECODE_TOP_K" \
  --action_mode "${HSAC_ACTION_MODE:-sample}" \
  --action_temperature "${HSAC_ACTION_TEMPERATURE:-1.0}" \
  --use_rapi \
  --regret_pool_size "${REGRET_POOL_SIZE:-20}" \
  --regret_gamma "${REGRET_GAMMA:-0.9}" \
  --regret_phi_scale "${REGRET_PHI_SCALE:-1.0}" \
  --regret_phi_clip "${REGRET_PHI_CLIP:-2.0}" \
  --memory_signal_scope "$MEMORY_SIGNAL_SCOPE" \
  --sara_eta "$HSAC_SARA_ETA" \
  --rrca_apply_to effective_reward \
  "$@" \
  2>&1 | tee "${REGRET_ROOT}/artifacts/logs/${MAIN_ID}_policy_train.log"
