#!/usr/bin/env bash

source .venv/bin/activate

REPO_ID="cr5af/shaft_insert"

uv run expo_ft/agents/vla/openpi/scripts/compute_norm_stats.py \
    --config-name expo_pi05_droid_lora_finetune_sft_cartesian_state \
    --repo-id "$REPO_ID"
