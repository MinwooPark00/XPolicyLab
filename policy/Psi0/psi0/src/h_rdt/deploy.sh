#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_PATH=${1:-./outputs/quick_start_sanity2/checkpoint-1/model.safetensors}
HOST=${2:-0.0.0.0}
PORT=${3:-8010}

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export LEROBOT_VIDEO_BACKEND="${LEROBOT_VIDEO_BACKEND:-pyav}"

python tools/hrdt_serve.py \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --config_path "${CONFIG_PATH:-configs/hrdt_finetune_lerobot.yaml}" \
  --vision_encoder "${VISION_ENCODER:-dino-siglip}" \
  --device "${DEVICE:-cuda:0}" \
  --host "$HOST" \
  --port "$PORT"
