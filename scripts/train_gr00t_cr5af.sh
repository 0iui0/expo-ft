#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# EXPO-FT online RL training for CR5AF + DH PGE gripper with GR00T N1.7 VLA actor.
# Task: grasp motor shaft and insert into bushing (shaft_insert).
#
# Prerequisites:
#   1. Finetuned GR00T checkpoint (SFT on shaft_insert_l50, lookahead=50)
#   2. Offline demo dataset (/datasets/shaft_insert_l50, LeRobot v2)
#   3. Env server running on thor (client_host:client_port):
#        ssh thor 'cd ~/workspaces/expo-ft && \
#          ~/workspaces/hil-serl/.venv/bin/python client/run_client.py \
#          --config_task_path configs/task/cr5af_gripper.py'
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ─── Proxy (override if your env is different) ───────────────────────────────
export https_proxy=http://192.168.16.150:7897
export http_proxy=http://192.168.16.150:7897

# ─── JAX + PyTorch tuning (single-GPU coexistence: jax critic + torch actor) ─
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MALLOC_TRIM_THRESHOLD_=100000
export CUDA_VISIBLE_DEVICES=1

# ─── Paths ───────────────────────────────────────────────────────────────────
GR00T_CKPT="${GR00T_CKPT:-/datasets/checkpoints/shaft_insert_l50/checkpoint-20000}"
DATASET_PATH="${DATASET_PATH:-/datasets/shaft_insert_l50}"
OUTPUT_DIR="${OUTPUT_DIR:-/datasets/expo_gr00t_cr5af/runs}"
MAX_STEPS="${MAX_STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
UTD_RATIO="${UTD_RATIO:-20}"
OFFLINE_RATIO="${OFFLINE_RATIO:-0.5}"
REPLAN_STEPS="${REPLAN_STEPS:-8}"
ACTOR_LR="${ACTOR_LR:-3e-5}"
CRITIC_LR="${CRITIC_LR:-3e-4}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-gr00t-cr5af-shaft-insert}"
WANDB_PROJECT="${WANDB_PROJECT:-expo-ft-gr00t}"

# ─── Env server (thor) ───────────────────────────────────────────────────────
CLIENT_HOST="${CLIENT_HOST:-192.168.16.158}"
CLIENT_PORT="${CLIENT_PORT:-8102}"

# ─── Launch ──────────────────────────────────────────────────────────────────
cd "$(dirname "$0")"/..

python train_gr00t_robo.py \
  --config configs/model/expo_ft_gr00t_config.py \
  --config_task configs/task/cr5af_gripper.py \
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
  --client_host "$CLIENT_HOST" \
  --client_port "$CLIENT_PORT" \
  --tqdm \
  --checkpoint_model \
  --checkpoint_interval 5000 \
  --keep_period 5000
