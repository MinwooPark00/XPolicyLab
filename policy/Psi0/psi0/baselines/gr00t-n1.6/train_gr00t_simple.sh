#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec python3 "$SCRIPT_DIR/finetune_gr00t.py" \
  --preset finetune_simple \
  --dataset-path "${DATASET_PATH:-/hfm/data/simple/simple/G1WholebodyBendPick-v0-psi0}" \
  --base-model-path "${PRETRAINED_MODEL_PATH:-./checkpoints/pretrain_he_g1_h1_mixed_scratch_gr00t/checkpoint-50000}" \
  --output-dir "${OUTPUT_DIR:-./checkpoints/pretrained_mixed_scratch_downstream}" \
  --cuda-visible-devices "${CUDA_VISIBLE_DEVICES:-5,6}" \
  --master-port "${MASTER_PORT:-29501}" \
  "$@"
