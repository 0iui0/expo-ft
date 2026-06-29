#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# EXPO-FT online RL training for CR5AF + TopHand with GR00T N1.7 VLA actor.
#
# Prerequisites:
#   1. A finetuned GR00T checkpoint (e.g. from launch_finetune.py)
#   2. An offline dataset of demos (e.g. /datasets/cr5af_grasp_housing_l50)
#   3. The env server running on client_host:client_port
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ─── Proxy (override if your env is different) ───────────────────────────────
export https_proxy=http://192.168.16.152:7897
export http_proxy=http://192.168.16.152:7897

# ─── PyTorch tuning ──────────────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MALLOC_TRIM_THRESHOLD_=100000
export CUDA_VISIBLE_DEVICES=1

# ─── Paths ───────────────────────────────────────────────────────────────────
GR00T_CKPT="${GR00T_CKPT:-/tmp/cr5af_finetune_v5/checkpoint-20000}"
DATASET_PATH="${DATASET_PATH:-/datasets/cr5af_grasp_housing_l50}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/expo_gr00t_cr5af}"
MAX_STEPS="${MAX_STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
UTD_RATIO="${UTD_RATIO:-20}"
OFFLINE_RATIO="${OFFLINE_RATIO:-0.5}"
REPLAN_STEPS="${REPLAN_STEPS:-8}"
ACTOR_LR="${ACTOR_LR:-3e-5}"
CRITIC_LR="${CRITIC_LR:-3e-4}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-gr00t-cr5af-online}"
WANDB_PROJECT="${WANDB_PROJECT:-expo-ft-gr00t}"

# ─── Launch ──────────────────────────────────────────────────────────────────
cd "$(dirname "$0")"/..

python train_gr00t_robo.py \
  --config configs/model/expo_ft_gr00t_config.py \
  --config_task configs/task/cr5af.py \
  --project_name "$WANDB_PROJECT" \
  --run_name "$RUN_NAME" \
  --gr00t_model_path "$GR00T_CKPT" \
  --dataset_path "$DATASET_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --max_steps "$MAX_STEPS" \
  --batch_size "$BATCH_SIZE" \
  --utd_ratio "$UTD_RATIO" \
  --offline_ratio "$OFFLINE_RATIO" \
  --replan_steps "$REPLAN_STEPS" \
  --seed "$SEED" \
  --num_updates 1 \
  --update_type episode \
  --tqdm \
  --checkpoint_model \
  --checkpoint_interval 5000 \
  --keep_period 5000
