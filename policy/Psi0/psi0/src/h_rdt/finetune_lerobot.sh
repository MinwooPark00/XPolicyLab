#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash finetune_lerobot.sh /path/to/lerobot_dataset [output_dir]"
  exit 1
fi

export PYTHONPATH="$(pwd)"
export LEROBOT_VIDEO_BACKEND="${LEROBOT_VIDEO_BACKEND:-pyav}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
USE_PRECOMP_LANG_EMBED="${USE_PRECOMP_LANG_EMBED:-1}"
LANG_EMBED_DEVICE="${LANG_EMBED_DEVICE:-cuda:0}"

DATA_ROOT="$1"
OUTPUT_DIR="${2:-./outputs/$(basename "$DATA_ROOT")}"
PRETRAINED_BACKBONE_PATH="${PRETRAINED_BACKBONE_PATH:-}"
DINO_SIGLIP_DIR="${DINO_SIGLIP_DIR:-}"
export DINO_SIGLIP_DIR

if [[ ! -d "$DATA_ROOT" ]]; then
  echo "Dataset root not found: $DATA_ROOT"
  exit 1
fi

if [[ -z "$DINO_SIGLIP_DIR" ]]; then
  echo "Set DINO_SIGLIP_DIR to the local dino-siglip model directory."
  exit 1
fi

if [[ -z "$PRETRAINED_BACKBONE_PATH" ]]; then
  echo "Set PRETRAINED_BACKBONE_PATH to the local pretrained H-RDT backbone checkpoint."
  exit 1
fi

if [[ ! -f "$PRETRAINED_BACKBONE_PATH" ]]; then
  echo "Missing pretrained backbone: $PRETRAINED_BACKBONE_PATH"
  exit 1
fi

needs_lang_embeddings() {
  local meta_dir="$1/meta"
  local lang_map_path="$meta_dir/lang_map.json"
  local embed_dir="$meta_dir/lang_embeddings"

  if [[ ! -f "$lang_map_path" ]]; then
    return 1
  fi

  if [[ ! -d "$embed_dir" ]]; then
    return 1
  fi

  if ! find "$embed_dir" -maxdepth 1 -type f -name '*.pt' | grep -q .; then
    return 1
  fi

  python3 - "$lang_map_path" "$embed_dir" <<'PY'
import json
import sys
from pathlib import Path

lang_map_path = Path(sys.argv[1])
embed_dir = Path(sys.argv[2])

try:
    lang_map = json.loads(lang_map_path.read_text())
except Exception:
    raise SystemExit(1)

if not isinstance(lang_map, dict) or not lang_map:
    raise SystemExit(1)

for embed_name in lang_map.values():
    if not (embed_dir / embed_name).is_file():
        raise SystemExit(1)
PY
  return $?
}

if [[ "$USE_PRECOMP_LANG_EMBED" != "0" ]]; then
  if ! needs_lang_embeddings "$DATA_ROOT"; then
    if [[ -z "${T5_MODEL_PATH:-}" ]]; then
      echo "Missing LeRobot language embeddings for $DATA_ROOT."
      echo "Set T5_MODEL_PATH so finetune_lerobot.sh can generate them automatically,"
      echo "or set USE_PRECOMP_LANG_EMBED=0 to train without precomputed language embeddings."
      exit 1
    fi

    echo "Generating LeRobot language embeddings under $DATA_ROOT/meta/lang_embeddings"
    python datasets/lerobot/encode_lang_batch.py \
      --data_root "$DATA_ROOT" \
      --config_path "configs/hrdt_finetune_lerobot.yaml" \
      --model_path "$T5_MODEL_PATH" \
      --device "$LANG_EMBED_DEVICE"
  fi
fi

PRECOMP_LANG_EMBED_ARGS=()
if [[ "$USE_PRECOMP_LANG_EMBED" != "0" ]]; then
  PRECOMP_LANG_EMBED_ARGS+=(--precomp_lang_embed)
fi

mkdir -p "$OUTPUT_DIR"

TRAIN_LENGTH_ARGS=()
if [[ -n "${MAX_TRAIN_STEPS:-}" ]]; then
  TRAIN_LENGTH_ARGS+=(--max_train_steps "${MAX_TRAIN_STEPS}")
else
  TRAIN_LENGTH_ARGS+=(--num_train_epochs "${NUM_TRAIN_EPOCHS:-1}")
fi

python main.py \
  --pretrained_vision_encoder_name_or_path="dino-siglip" \
  --config_path="configs/hrdt_finetune_lerobot.yaml" \
  --output_dir="$OUTPUT_DIR" \
  --train_batch_size="${TRAIN_BATCH_SIZE:-2}" \
  --sample_batch_size="${SAMPLE_BATCH_SIZE:-2}" \
  --num_sample_batches="${NUM_SAMPLE_BATCHES:-1}" \
  --checkpointing_period="${CHECKPOINTING_PERIOD:-5}" \
  --sample_period="${SAMPLE_PERIOD:--1}" \
  --checkpoints_total_limit="${CHECKPOINTS_TOTAL_LIMIT:-2}" \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=1 \
  --learning_rate="${LEARNING_RATE:-1e-4}" \
  --mixed_precision="${MIXED_PRECISION:-bf16}" \
  --dataloader_num_workers="${DATALOADER_NUM_WORKERS:-0}" \
  --dataset_type="finetune" \
  --dataset_name="lerobot" \
  --dataset_root="$DATA_ROOT" \
  --report_to="${REPORT_TO:-wandb}" \
  --upsample_rate=1 \
  "${PRECOMP_LANG_EMBED_ARGS[@]}" \
  "${TRAIN_LENGTH_ARGS[@]}" \
  --training_mode="lang" \
  --mode="finetune" \
  --pretrained_backbone_path="$PRETRAINED_BACKBONE_PATH"
