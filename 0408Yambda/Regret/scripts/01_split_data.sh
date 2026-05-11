#!/usr/bin/env bash
# 用途：主线数据切分。按 1 小时断 session，再把同 session 内连续同 item 的事件聚合成一个 RL step。
# 默认不会覆盖已有数据；需要重切时设置 FORCE_SPLIT=1。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGRET_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REGRET_ROOT}/configs/main.env"
cd "$REGRET_ROOT"
mkdir -p artifacts/logs artifacts/transitions

OUT_ROOT="${OUT_ROOT:-$DATA_OUT_ROOT}"

if [[ -e "$OUT_ROOT" && "${FORCE_SPLIT:-0}" != "1" ]]; then
  echo "[skip] transition root exists: $OUT_ROOT"
  echo "[hint] set FORCE_SPLIT=1 to regenerate."
  exit 0
fi

OUT_ROOT="$OUT_ROOT" \
LOG_PATH="${REGRET_ROOT}/artifacts/logs/${MAIN_ID}_split.log"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

regret_memory_args=()
if [[ "$PRECOMPUTE_REGRET_MEMORY" == "1" ]]; then
  regret_memory_args+=(--precompute_regret_memory)
fi

python3 -u scripts/02_split_transitions.py \
  --multi_event_parquet "$MULTI_EVENT_PARQUET" \
  --orig2dense_npy "$ORIG2DENSE_NPY" \
  --out_root "$OUT_ROOT" \
  --history_len "$HISTORY_LEN" \
  --trajectory_mode session_run \
  --session_gap_seconds "$SESSION_GAP_SECONDS" \
  --max_session_span_seconds "$MAX_SESSION_SPAN_SECONDS" \
  --max_run_events "$MAX_RUN_EVENTS" \
  --replay_eval_steps "$REPLAY_EVAL_STEPS" \
  --timestamp_unit_seconds "$TIMESTAMP_UNIT_SECONDS" \
  --anchor_policy "$ANCHOR_POLICY" \
  --max_users "$MAX_USERS" \
  --shard_rows "$SHARD_ROWS" \
  --reward_version "$REWARD_VERSION" \
  --reward_v2_w_like "${REWARD_V2_W_LIKE:-0.8}" \
  --reward_v2_w_dislike "${REWARD_V2_W_DISLIKE:-1.2}" \
  --reward_v2_w_unlike "${REWARD_V2_W_UNLIKE:-0.6}" \
  --reward_v2_w_undislike "${REWARD_V2_W_UNDISLIKE:-0.2}" \
  --reward_v2_clip_min "${REWARD_V2_CLIP_MIN:--2}" \
  --reward_v2_clip_max "${REWARD_V2_CLIP_MAX:-2}" \
  "${regret_memory_args[@]}" \
  --regret_memory_size "$REGRET_MEMORY_SIZE" \
  --regret_memory_gamma "$REGRET_MEMORY_GAMMA" \
  --regret_memory_min_phi "${REGRET_MEMORY_MIN_PHI:-0.000001}" \
  --regret_memory_scope "$REGRET_MEMORY_SCOPE" \
  "$@" \
  2>&1 | tee "$LOG_PATH"
