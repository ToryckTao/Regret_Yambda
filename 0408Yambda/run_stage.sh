#!/usr/bin/env bash
set -euo pipefail

# 正式全量阶段入口。所有默认参数来自 pipeline_config.sh。
#
# 用法：
#   ./run_stage.sh 01
#   ./run_stage.sh all

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE="${1:-help}"

source "${SCRIPT_DIR}/pipeline_config.sh"
cd "${YAMBA_WORK_DIR}"

need_file() {
  local path="$1"
  local hint="$2"
  if [[ ! -f "${path}" ]]; then
    echo "[error] 缺少文件: ${path}" >&2
    echo "        ${hint}" >&2
    exit 1
  fi
}

run_01_codebook() {
  echo "[stage] 01_codebook"
  echo "[output] ${CODEBOOK_NPZ}"
  MPLCONFIGDIR="${MPLCONFIGDIR}" "${PYTHON_BIN}" -u 01_build_codebook.py \
    --embeddings_parquet "${EMBEDDINGS_PARQUET}" \
    --embedding_column "${EMBEDDING_COLUMN}" \
    --sample_size "${CODEBOOK_SAMPLE_SIZE}" \
    --batch_size "${EMBEDDING_BATCH_SIZE}" \
    --n_levels "${CODEBOOK_N_LEVELS}" \
    --codebook_size "${CODEBOOK_SIZE}" \
    --max_iter "${CODEBOOK_MAX_ITER}" \
    --device "${CODEBOOK_DEVICE}" \
    --output_npz "${CODEBOOK_NPZ}" \
    --output_meta "${CODEBOOK_META}"
}

run_02_item_sid() {
  echo "[stage] 02_item_sid"
  need_file "${CODEBOOK_NPZ}" "请先运行 ./run_stage.sh 01"
  echo "[output] ${DENSE_ITEM2SID_NPY}"
  MPLCONFIGDIR="${MPLCONFIGDIR}" "${PYTHON_BIN}" -u 02_build_item_sid.py \
    --embeddings_parquet "${EMBEDDINGS_PARQUET}" \
    --codebook_npz "${CODEBOOK_NPZ}" \
    --embedding_column "${EMBEDDING_COLUMN}" \
    --batch_size "${EMBEDDING_BATCH_SIZE}" \
    --output_dir artifacts/mappings \
    --output_prefix "${OUTPUT_PREFIX}"
}

run_03_split() {
  echo "[stage] 03_split"
  echo "[warning] 正式 split 会写出较大的 train/val/test TSV；请确认磁盘空间。"
  need_file "${ORIG2DENSE_NPY}" "请先运行 ./run_stage.sh 02"
  echo "[output] ${TRAIN_TSV}"
  echo "[output] ${VAL_TSV}"
  echo "[output] ${TEST_TSV}"
  MPLCONFIGDIR="${MPLCONFIGDIR}" "${PYTHON_BIN}" -u 03_split_data.py \
    --multi_event_parquet "${MULTI_EVENT_PARQUET}" \
    --orig2dense_npy "${ORIG2DENSE_NPY}" \
    --history_len "${HISTORY_LEN}" \
    --close_gap_seconds "${CLOSE_GAP_SECONDS}" \
    --anchor_policy "${ANCHOR_POLICY}" \
    --positive_play_threshold "${POSITIVE_PLAY_THRESHOLD}" \
    --max_users "${MAX_USERS}" \
    --output_train_tsv "${TRAIN_TSV}" \
    --output_val_tsv "${VAL_TSV}" \
    --output_test_tsv "${TEST_TSV}" \
    --output_meta "${SPLIT_META}"
}

run_04_hpn() {
  echo "[stage] 04_hpn"
  if [[ "${HPN_MAX_TRAIN_ROWS}" == "0" ]]; then
    echo "[warning] HPN_MAX_TRAIN_ROWS=0 会读取全量 train TSV；大文件可能需要较大内存。"
  fi
  need_file "${TRAIN_TSV}" "请先运行 ./run_stage.sh 03"
  need_file "${VAL_TSV}" "请先运行 ./run_stage.sh 03"
  need_file "${DENSE_ITEM2SID_NPY}" "请先运行 ./run_stage.sh 02"
  echo "[output] ${HPN_CHECKPOINT}"
  MPLCONFIGDIR="${MPLCONFIGDIR}" "${PYTHON_BIN}" -u 04_train_hpn_warmstart.py \
    --train_file "${TRAIN_TSV}" \
    --val_file "${VAL_TSV}" \
    --embeddings_parquet "${EMBEDDINGS_PARQUET}" \
    --orig2dense_npy "${ORIG2DENSE_NPY}" \
    --dense_item2sid_npy "${DENSE_ITEM2SID_NPY}" \
    --embedding_column "${EMBEDDING_COLUMN}" \
    --max_seq_len "${HISTORY_LEN}" \
    --max_train_rows "${HPN_MAX_TRAIN_ROWS}" \
    --max_val_rows "${HPN_MAX_VAL_ROWS}" \
    --batch_size "${HPN_BATCH_SIZE}" \
    --epochs "${HPN_EPOCHS}" \
    --device "${DEVICE}" \
    --save_path "${HPN_CHECKPOINT}" \
    --save_meta "${HPN_META}"
}

run_05_user_response() {
  echo "[stage] 05_user_response"
  if [[ "${URM_MAX_TRAIN_ROWS}" == "0" ]]; then
    echo "[warning] URM_MAX_TRAIN_ROWS=0 会读取全量 train TSV；大文件可能需要较大内存。"
  fi
  need_file "${TRAIN_TSV}" "请先运行 ./run_stage.sh 03"
  need_file "${VAL_TSV}" "请先运行 ./run_stage.sh 03"
  need_file "${TEST_TSV}" "请先运行 ./run_stage.sh 03"
  echo "[output] ${URM_MODEL_PATH}.checkpoint"
  echo "[output] ${URM_LOG_PATH}"
  echo "[output] ${URM_FEATURE_CACHE_NPZ}"
  MPLCONFIGDIR="${MPLCONFIGDIR}" "${PYTHON_BIN}" -u 05_train_user_response.py \
    --train_file "${TRAIN_TSV}" \
    --val_file "${VAL_TSV}" \
    --test_file "${TEST_TSV}" \
    --embeddings_parquet "${EMBEDDINGS_PARQUET}" \
    --orig2dense_npy "${ORIG2DENSE_NPY}" \
    --embedding_column "${EMBEDDING_COLUMN}" \
    --max_seq_len "${HISTORY_LEN}" \
    --max_train_rows "${URM_MAX_TRAIN_ROWS}" \
    --max_val_rows "${URM_MAX_VAL_ROWS}" \
    --train_sample_size "${URM_TRAIN_SAMPLE_SIZE}" \
    --val_sample_size "${URM_VAL_SAMPLE_SIZE}" \
    --batch_size "${URM_BATCH_SIZE}" \
    --epoch "${URM_EPOCHS}" \
    --device "${DEVICE}" \
    --model_path "${URM_MODEL_PATH}" \
    --feature_cache_npz "${URM_FEATURE_CACHE_NPZ}" \
    --effective_train_file "${URM_EFFECTIVE_TRAIN_FILE}" \
    --effective_val_file "${URM_EFFECTIVE_VAL_FILE}" \
    --log_path "${URM_LOG_PATH}" \
    --save_meta "${URM_META}"
}

run_06_sid_rl() {
  echo "[stage] 06_sid_rl"
  need_file "${URM_LOG_PATH}" "请先运行 ./run_stage.sh 05"
  need_file "${DENSE_ITEM2SID_NPY}" "请先运行 ./run_stage.sh 02"
  need_file "${HPN_CHECKPOINT}" "请先运行 ./run_stage.sh 04"
  echo "[output] ${SID_SAVE_PATH}_actor"
  echo "[output] ${SID_SAVE_PATH}_critic"
  MPLCONFIGDIR="${MPLCONFIGDIR}" "${PYTHON_BIN}" -u 06_train_yambda_sid.py \
    --device "${DEVICE}" \
    --urm_log_path "${URM_LOG_PATH}" \
    --dense_item2sid_npy "${DENSE_ITEM2SID_NPY}" \
    --hpn_checkpoint "${HPN_CHECKPOINT}" \
    --save_path "${SID_SAVE_PATH}" \
    --save_meta "${SID_SAVE_META}" \
    --n_iter "${SID_N_ITER}" \
    --episode_batch_size "${SID_EPISODE_BATCH_SIZE}" \
    --batch_size "${SID_BATCH_SIZE}" \
    --start_timestamp "${SID_START_TIMESTAMP}" \
    --buffer_size "${SID_BUFFER_SIZE}" \
    --max_candidate_items "${SID_MAX_CANDIDATE_ITEMS}" \
    --check_episode "${SID_CHECK_EPISODE}" \
    --train_every_n_step "${SID_TRAIN_EVERY_N_STEP}" \
    --gamma "${SID_GAMMA}" \
    --actor_lr "${SID_ACTOR_LR}" \
    --critic_lr "${SID_CRITIC_LR}" \
    --actor_decay "${SID_ACTOR_DECAY}" \
    --critic_decay "${SID_CRITIC_DECAY}" \
    --initial_greedy_epsilon "${SID_INITIAL_GREEDY_EPSILON}" \
    --final_greedy_epsilon "${SID_FINAL_GREEDY_EPSILON}" \
    --elbow_greedy "${SID_ELBOW_GREEDY}" \
    --entropy_coef "${SID_ENTROPY_COEF}" \
    --bc_coef "${SID_BC_COEF}" \
    --critic_hidden_dims ${SID_CRITIC_HIDDEN_DIMS}
}

run_07_eval() {
  echo "[stage] 07_eval"
  need_file "${TEST_TSV}" "请先运行 ./run_stage.sh 03"
  need_file "${EVAL_ACTOR_CHECKPOINT}" "请先运行 ./run_stage.sh 06"
  echo "[output] ${EVAL_META}"
  MPLCONFIGDIR="${MPLCONFIGDIR}" "${PYTHON_BIN}" -u 07_eval_candidate_ranking.py \
    --eval_file "${TEST_TSV}" \
    --embeddings_parquet "${EMBEDDINGS_PARQUET}" \
    --orig2dense_npy "${ORIG2DENSE_NPY}" \
    --dense_item2sid_npy "${DENSE_ITEM2SID_NPY}" \
    --actor_checkpoint "${EVAL_ACTOR_CHECKPOINT}" \
    --embedding_column "${EMBEDDING_COLUMN}" \
    --max_seq_len "${HISTORY_LEN}" \
    --max_eval_rows "${EVAL_MAX_ROWS}" \
    --num_negatives "${EVAL_NUM_NEGATIVES}" \
    --batch_size "${EVAL_BATCH_SIZE}" \
    --device "${DEVICE}" \
    --save_meta "${EVAL_META}"
}

show_help() {
  cat <<EOF
用法:
  ./run_stage.sh <stage>

正式阶段:
  01 / 01_codebook       训练 RQ-kmeans codebook -> ${CODEBOOK_NPZ}
  02 / 02_item_sid       全量 item -> dense id / SID -> ${DENSE_ITEM2SID_NPY}
  03 / 03_split          构造 train/val/test TSV -> ${TRAIN_TSV}
  04 / 04_hpn            HPN warm-start -> ${HPN_CHECKPOINT}
  05 / 05_user_response  训练 UserResponse -> ${URM_MODEL_PATH}.checkpoint
  06 / 06_sid_rl         SID actor-critic -> ${SID_SAVE_PATH}_actor
  07 / 07_eval           候选集排序评估 -> ${EVAL_META}

组合阶段:
  preprocess             依次运行 01, 02, 03
  train                  依次运行 04, 05, 06
  all                    依次运行 01-07

配置文件:
  ${SCRIPT_DIR}/pipeline_config.sh
EOF
}

case "${STAGE}" in
  01|01_codebook) run_01_codebook ;;
  02|02_item_sid) run_02_item_sid ;;
  03|03_split) run_03_split ;;
  04|04_hpn) run_04_hpn ;;
  05|05_user_response) run_05_user_response ;;
  06|06_sid_rl) run_06_sid_rl ;;
  07|07_eval) run_07_eval ;;
  preprocess)
    run_01_codebook
    run_02_item_sid
    run_03_split
    ;;
  train)
    run_04_hpn
    run_05_user_response
    run_06_sid_rl
    ;;
  all)
    run_01_codebook
    run_02_item_sid
    run_03_split
    run_04_hpn
    run_05_user_response
    run_06_sid_rl
    run_07_eval
    ;;
  help|-h|--help) show_help ;;
  *)
    echo "[error] unknown stage: ${STAGE}" >&2
    show_help
    exit 1
    ;;
esac
