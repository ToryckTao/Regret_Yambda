#!/usr/bin/env bash
# 用途：扫描 RAPI 介入强度 eta，观察 reward/depth/negative rate 对 eta 的敏感性。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGRET_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REGRET_ROOT}/configs/main.env"
cd "$REGRET_ROOT"
mkdir -p artifacts/evals artifacts/logs

SPLIT="${SPLIT:-test}"
EPISODES="${ROLLOUT_EPISODES:-500}"
ACTOR_CHECKPOINT="${ACTOR_CHECKPOINT:-$POLICY_ACTOR}"
SUMMARY_PATH="${SUMMARY_PATH:-${REGRET_ROOT}/artifacts/evals/${MAIN_ID}_eta_sweep_${SPLIT}_${EPISODES}.tsv}"

printf "eta\tepisodes\tbase_reward\trapi_reward\tdelta_reward\tbase_step\trapi_step\tdelta_step\tbase_neg\trapi_neg\tdelta_neg\n" > "$SUMMARY_PATH"

for eta in $ETA_LIST; do
  eta_tag="${eta//./p}"
  meta_path="${REGRET_ROOT}/artifacts/evals/${MAIN_ID}_eta${eta_tag}_${SPLIT}_${EPISODES}.meta.json"
  log_path="${REGRET_ROOT}/artifacts/logs/${MAIN_ID}_eta${eta_tag}_${SPLIT}_${EPISODES}.log"
  echo "[sweep] eta=${eta} episodes=${EPISODES} max_steps=${ROLLOUT_MAX_STEPS}"
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
    --save_meta "$meta_path" \
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
    --sara_eta "$eta" \
    --sara_layer_weights "${SARA_LAYER_WEIGHTS:-0.05,0.25,0.70}" \
    --regret_pool_size "${REGRET_POOL_SIZE:-20}" \
    --regret_gamma "${REGRET_GAMMA:-0.9}" \
    --regret_phi_scale "${REGRET_PHI_SCALE:-1.0}" \
    --regret_phi_clip "${REGRET_PHI_CLIP:-2.0}" \
    --regret_reward_threshold "${REGRET_REWARD_THRESHOLD:-0.0}" \
    "$@" \
    2>&1 | tee "$log_path"

  python3 - "$meta_path" "$eta" "$SUMMARY_PATH" <<'PY'
import json
import sys

meta_path, eta, summary_path = sys.argv[1], sys.argv[2], sys.argv[3]
with open(meta_path, "r", encoding="utf-8") as f:
    data = json.load(f)
base = data["base"]
rapi = data["rapi"]
delta = data["delta_rapi_minus_base"]
row = [
    eta,
    str(int(data["args"]["num_episodes"])),
    f"{base['avg_cum_reward']:.6f}",
    f"{rapi['avg_cum_reward']:.6f}",
    f"{delta['avg_cum_reward']:.6f}",
    f"{base['avg_step']:.6f}",
    f"{rapi['avg_step']:.6f}",
    f"{delta['avg_step']:.6f}",
    f"{base['negative_rate']:.6f}",
    f"{rapi['negative_rate']:.6f}",
    f"{delta['negative_rate']:.6f}",
]
with open(summary_path, "a", encoding="utf-8") as f:
    f.write("\t".join(row) + "\n")
PY
done

echo "[done] summary saved to ${SUMMARY_PATH}"
cat "$SUMMARY_PATH"
