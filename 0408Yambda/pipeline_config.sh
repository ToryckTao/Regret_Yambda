#!/usr/bin/env bash
# Yambda-HSRL baseline 正式全量配置。
#
# 用法：
#   source ./pipeline_config.sh
#   ./run_stage.sh 01
#
# 说明：
# - 这里是 shell 流程的唯一集中配置入口。
# - 单个变量可以在命令前用环境变量覆盖，例如：
#   CODEBOOK_SAMPLE_SIZE=500000 ./run_stage.sh 01

set -euo pipefail

CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export YAMBA_WORK_DIR="${YAMBA_WORK_DIR:-${CONFIG_DIR}}"
export HSRL_ROOT="${HSRL_ROOT:-${YAMBA_WORK_DIR}/hsrl_core}"
export YAMBA_DATA_DIR="${YAMBA_DATA_DIR:-/Users/Toryck/Coding/DATASET/Yambda}"
export EMBEDDINGS_PARQUET="${EMBEDDINGS_PARQUET:-${YAMBA_DATA_DIR}/embeddings.parquet}"
export MULTI_EVENT_PARQUET="${MULTI_EVENT_PARQUET:-${YAMBA_DATA_DIR}/sequential/50m/multi_event.parquet}"

export PYTHON_BIN="${PYTHON_BIN:-python3}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp}"
export DEVICE="${DEVICE:-cuda}"
export CODEBOOK_DEVICE="${CODEBOOK_DEVICE:-cuda}"
export EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-4096}"
export EMBEDDING_COLUMN="${EMBEDDING_COLUMN:-normalized_embed}"

# 预编码：正式 baseline 固定为 4 层，每层 256 个 token。
export CODEBOOK_N_LEVELS="${CODEBOOK_N_LEVELS:-4}"
export CODEBOOK_SIZE="${CODEBOOK_SIZE:-256}"
export CODEBOOK_SAMPLE_SIZE="${CODEBOOK_SAMPLE_SIZE:-200000}"
export CODEBOOK_MAX_ITER="${CODEBOOK_MAX_ITER:-30}"
export CODEBOOK_NPZ="${CODEBOOK_NPZ:-artifacts/codebook/yambda_rq_codebook.npz}"
export CODEBOOK_META="${CODEBOOK_META:-artifacts/codebook/yambda_rq_codebook.meta.json}"

export OUTPUT_PREFIX="${OUTPUT_PREFIX:-yambda}"
export ORIG2DENSE_NPY="${ORIG2DENSE_NPY:-artifacts/mappings/yambda_orig2dense_item_id.npy}"
export DENSE2ORIG_NPY="${DENSE2ORIG_NPY:-artifacts/mappings/yambda_dense2orig_item_id.npy}"
export DENSE_ITEM2SID_NPY="${DENSE_ITEM2SID_NPY:-artifacts/mappings/yambda_dense_item2sid.npy}"

# 样本切分：MAX_USERS=0 表示全量用户。
export HISTORY_LEN="${HISTORY_LEN:-50}"
export CLOSE_GAP_SECONDS="${CLOSE_GAP_SECONDS:-3600}"
export ANCHOR_POLICY="${ANCHOR_POLICY:-first_visible}"
export POSITIVE_PLAY_THRESHOLD="${POSITIVE_PLAY_THRESHOLD:-0.8}"
export MAX_USERS="${MAX_USERS:-0}"
export TRAIN_TSV="${TRAIN_TSV:-artifacts/processed/train.tsv}"
export VAL_TSV="${VAL_TSV:-artifacts/processed/val.tsv}"
export TEST_TSV="${TEST_TSV:-artifacts/processed/test.tsv}"
export SPLIT_META="${SPLIT_META:-artifacts/processed/split.meta.json}"

# HPN warm-start：MAX_*_ROWS=0 表示读取全量 split。
export HPN_CHECKPOINT="${HPN_CHECKPOINT:-artifacts/models/hpn_warmstart.pt}"
export HPN_META="${HPN_META:-artifacts/models/hpn_warmstart.meta.json}"
export HPN_MAX_TRAIN_ROWS="${HPN_MAX_TRAIN_ROWS:-0}"
export HPN_MAX_VAL_ROWS="${HPN_MAX_VAL_ROWS:-0}"
export HPN_BATCH_SIZE="${HPN_BATCH_SIZE:-256}"
export HPN_EPOCHS="${HPN_EPOCHS:-3}"

# UserResponse：作为离线 RL 环境里的近似反馈模型。
export URM_MODEL_PATH="${URM_MODEL_PATH:-artifacts/env/yambda_user_env.model}"
export URM_FEATURE_CACHE_NPZ="${URM_FEATURE_CACHE_NPZ:-artifacts/env/yambda_user_env.feature_cache.npz}"
export URM_EFFECTIVE_TRAIN_FILE="${URM_EFFECTIVE_TRAIN_FILE:-artifacts/env/yambda_user_env.train.tsv}"
export URM_EFFECTIVE_VAL_FILE="${URM_EFFECTIVE_VAL_FILE:-artifacts/env/yambda_user_env.val.tsv}"
export URM_LOG_PATH="${URM_LOG_PATH:-artifacts/env/log/yambda_user_env.model.log}"
export URM_META="${URM_META:-artifacts/env/yambda_user_env.meta.json}"
export URM_MAX_TRAIN_ROWS="${URM_MAX_TRAIN_ROWS:-0}"
export URM_MAX_VAL_ROWS="${URM_MAX_VAL_ROWS:-0}"
export URM_TRAIN_SAMPLE_SIZE="${URM_TRAIN_SAMPLE_SIZE:-0}"
export URM_VAL_SAMPLE_SIZE="${URM_VAL_SAMPLE_SIZE:-0}"
export URM_BATCH_SIZE="${URM_BATCH_SIZE:-256}"
export URM_EPOCHS="${URM_EPOCHS:-5}"

# SID actor-critic。
export SID_SAVE_PATH="${SID_SAVE_PATH:-artifacts/models/yambda_sid}"
export SID_SAVE_META="${SID_SAVE_META:-artifacts/models/yambda_sid.meta.json}"
export SID_N_ITER="${SID_N_ITER:-10000}"
export SID_EPISODE_BATCH_SIZE="${SID_EPISODE_BATCH_SIZE:-32}"
export SID_BATCH_SIZE="${SID_BATCH_SIZE:-128}"
export SID_START_TIMESTAMP="${SID_START_TIMESTAMP:-2000}"
export SID_BUFFER_SIZE="${SID_BUFFER_SIZE:-100000}"
export SID_MAX_CANDIDATE_ITEMS="${SID_MAX_CANDIDATE_ITEMS:-50000}"
export SID_CHECK_EPISODE="${SID_CHECK_EPISODE:-100}"
export SID_TRAIN_EVERY_N_STEP="${SID_TRAIN_EVERY_N_STEP:-5}"
export SID_GAMMA="${SID_GAMMA:-0.9}"
export SID_ACTOR_LR="${SID_ACTOR_LR:-0.0001}"
export SID_CRITIC_LR="${SID_CRITIC_LR:-0.001}"
export SID_ACTOR_DECAY="${SID_ACTOR_DECAY:-0.00001}"
export SID_CRITIC_DECAY="${SID_CRITIC_DECAY:-0.00001}"
export SID_INITIAL_GREEDY_EPSILON="${SID_INITIAL_GREEDY_EPSILON:-0}"
export SID_FINAL_GREEDY_EPSILON="${SID_FINAL_GREEDY_EPSILON:-0}"
export SID_ELBOW_GREEDY="${SID_ELBOW_GREEDY:-0.5}"
export SID_ENTROPY_COEF="${SID_ENTROPY_COEF:-0.01}"
export SID_BC_COEF="${SID_BC_COEF:-0.1}"
export SID_CRITIC_HIDDEN_DIMS="${SID_CRITIC_HIDDEN_DIMS:-256 64}"

# 候选集离线评估。
export EVAL_ACTOR_CHECKPOINT="${EVAL_ACTOR_CHECKPOINT:-artifacts/models/yambda_sid_actor}"
export EVAL_META="${EVAL_META:-artifacts/models/candidate_ranking.meta.json}"
export EVAL_MAX_ROWS="${EVAL_MAX_ROWS:-0}"
export EVAL_NUM_NEGATIVES="${EVAL_NUM_NEGATIVES:-99}"
export EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
