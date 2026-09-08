#!/usr/bin/env bash
set -euo pipefail

EXTRA_ARGS=()
if [[ "${1:-}" == --* ]]; then
  DATA_ROOT=${LEROBOT_DATA_ROOT:-${DATA_ROOT:-/hfm/data/pick_box}}
  SERVER_URL=${SERVER_URL:-http://127.0.0.1:8010/predict}
  MAX_SAMPLES=${MAX_SAMPLES:-32}
  EXTRA_ARGS=("$@")
else
  DATA_ROOT=${1:-${LEROBOT_DATA_ROOT:-${DATA_ROOT:-/hfm/data/pick_box}}}
  SERVER_URL=${2:-${SERVER_URL:-http://127.0.0.1:8010/predict}}
  MAX_SAMPLES=${3:-${MAX_SAMPLES:-32}}
  EXTRA_ARGS=("${@:4}")
fi

USE_PRECOMP_LANG_EMBED=${USE_PRECOMP_LANG_EMBED:-1}
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export LEROBOT_VIDEO_BACKEND="${LEROBOT_VIDEO_BACKEND:-pyav}"

ARGS=(
  --data_root "$DATA_ROOT"
  --server_url "$SERVER_URL"
  --config_path "${CONFIG_PATH:-configs/hrdt_finetune_lerobot.yaml}"
  --max_samples "$MAX_SAMPLES"
  --rollout_stride "${ROLLOUT_STRIDE:-0}"
)

if [[ "$USE_PRECOMP_LANG_EMBED" != "0" ]]; then
  ARGS+=(--use_precomp_lang_embed)
fi

python tools/hrdt_test_serve.py \
  "${ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
