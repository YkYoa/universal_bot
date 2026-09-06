#!/usr/bin/env bash
# Fine-tune pi0 (weights_local/pi0_base) TREN CPU LOCAL (khong dung GPU/server).
# Muc dich: SMOKE-TEST pipeline fine-tune -> export -> quantize thong suot,
# KHONG ky vong ra policy dung duoc that (dataset openarm_fixture_lerobot_v2.1
# hien chi co 1 episode - qua it de train that).
#
# Chay: ./train_pi05_cpu_local.sh [steps]
set -euo pipefail
cd "$(dirname "$0")"

STEPS="${1:-20}"
DATASET_REPO="local/openarm_fixture_lerobot_v2.1"
DATASET_ROOT="$(pwd)/datasets/openarm_fixture_lerobot_v2.1"
PRETRAINED="$(pwd)/weights_local/pi0_base"
OUT="$(pwd)/outputs/pi0_cpu_smoketest_$(date +%Y%m%d_%H%M%S)"
LOG="/tmp/pi0_cpu_train_$(date +%Y%m%d_%H%M%S).log"

echo "Steps       : $STEPS"
echo "Output      : $OUT"
echo "Log         : $LOG"

./.venv_local/bin/python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="$DATASET_REPO" \
  --dataset.root="$DATASET_ROOT" \
  --policy.type=pi0 \
  --policy.pretrained_path="$PRETRAINED" \
  --policy.repo_id=local/pi0_openarm_cpu_smoketest \
  --policy.push_to_hub=false \
  --policy.device=cpu \
  --policy.use_amp=false \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --batch_size=1 \
  --steps="$STEPS" \
  --log_freq=1 \
  --save_freq="$STEPS" \
  --num_workers=0 \
  --output_dir="$OUT" \
  --job_name=pi0_openarm_cpu_smoketest \
  --wandb.enable=false \
  2>&1 | tee "$LOG"

echo "Checkpoint saved to: $OUT"
echo "Full log: $LOG"
