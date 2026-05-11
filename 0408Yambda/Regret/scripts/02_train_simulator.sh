#!/usr/bin/env bash
# 用途：训练用户反馈 simulator。它接收历史状态和候选 item，预测近似反馈，并用于后续 rollout 评估。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGRET_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REGRET_ROOT}/configs/main.env"
cd "$REGRET_ROOT"
mkdir -p artifacts/logs artifacts/user_response

OUT_DIR="${OUT_DIR:-$SIM_TRAIN_OUT_DIR}"
mkdir -p "$OUT_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

python3 -u scripts/08_train_yambda_simulator.py \
  --transition_root "$TRANSITION_ROOT" \
  --item_features_npy "$DENSE_ITEM_FEATURES_NPY" \
  --out_dir "$OUT_DIR" \
  --max_seq_len "${SIM_MAX_SEQ_LEN:-50}" \
  --batch_size "$SIM_BATCH_SIZE" \
  --read_batch_size "${SIM_READ_BATCH_SIZE:-2048}" \
  --max_train_rows "$SIM_MAX_TRAIN_ROWS" \
  --max_val_rows "$SIM_MAX_VAL_ROWS" \
  --max_test_rows "$SIM_MAX_TEST_ROWS" \
  --epochs "$SIM_EPOCHS" \
  --lr "$SIM_LR" \
  --weight_decay "${SIM_WEIGHT_DECAY:-0.00001}" \
  --hidden_dim "${SIM_HIDDEN_DIM:-128}" \
  --dropout "${SIM_DROPOUT:-0.1}" \
  --train_sample_across_files \
  --shuffle_train_files \
  --shuffle_buffer_size "${SIM_SHUFFLE_BUFFER_SIZE:-50000}" \
  --decouple_reward_model \
  --best_metric "${SIM_BEST_METRIC:-coarse_error}" \
  --reward_loss_weight "${SIM_REWARD_LOSS_WEIGHT:-0.2}" \
  --listen_loss_weight "${SIM_LISTEN_LOSS_WEIGHT:-0.05}" \
  --play_loss_weight "${SIM_PLAY_LOSS_WEIGHT:-0.2}" \
  --feedback_loss_weight "${SIM_FEEDBACK_LOSS_WEIGHT:-0.3}" \
  --regret_loss_weight "${SIM_REGRET_LOSS_WEIGHT:-0.25}" \
  --negative_pos_weight "${SIM_NEGATIVE_POS_WEIGHT:-2.0}" \
  --negative_type_focal_gamma "${SIM_NEGATIVE_TYPE_FOCAL_GAMMA:-2.0}" \
  --play_bucket_weight_zero "${SIM_PLAY_BUCKET_WEIGHT_ZERO:-3}" \
  --play_bucket_weight_low "${SIM_PLAY_BUCKET_WEIGHT_LOW:-6}" \
  --play_bucket_weight_mid "${SIM_PLAY_BUCKET_WEIGHT_MID:-3}" \
  --play_bucket_weight_high "${SIM_PLAY_BUCKET_WEIGHT_HIGH:-1}" \
  --feedback_pos_weight_like "${SIM_FEEDBACK_POS_WEIGHT_LIKE:-4}" \
  --feedback_pos_weight_dislike "${SIM_FEEDBACK_POS_WEIGHT_DISLIKE:-8}" \
  --feedback_pos_weight_unlike "${SIM_FEEDBACK_POS_WEIGHT_UNLIKE:-6}" \
  --feedback_pos_weight_undislike "${SIM_FEEDBACK_POS_WEIGHT_UNDISLIKE:-4}" \
  --regret_class_weight_low_play "${SIM_REGRET_CLASS_WEIGHT_LOW_PLAY:-1.0}" \
  --regret_class_weight_dislike "${SIM_REGRET_CLASS_WEIGHT_DISLIKE:-8.0}" \
  --regret_class_weight_unlike "${SIM_REGRET_CLASS_WEIGHT_UNLIKE:-6.0}" \
  --eval_test_after_train \
  "$@" \
  2>&1 | tee "${OUT_DIR}/train.log"
