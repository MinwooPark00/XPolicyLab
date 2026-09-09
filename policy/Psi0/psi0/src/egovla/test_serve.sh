#!/bin/bash
set -euo pipefail

EXTRA_ARGS=()
if [[ "${1:-}" == --* ]]; then
  DATA_ROOT=${DATA_ROOT:-/hfm/data/real_teleop_g1/lerobot}
  SERVER_URL=${SERVER_URL:-http://127.0.0.1:8000/predict}
  MAX_SAMPLES=${MAX_SAMPLES:-200}
  EXTRA_ARGS=("$@")
else
  DATA_ROOT=${1:-${DATA_ROOT:-/hfm/data/real_teleop_g1/lerobot}}
  SERVER_URL=${2:-${SERVER_URL:-http://127.0.0.1:8000/predict}}
  MAX_SAMPLES=${3:-${MAX_SAMPLES:-200}}
  EXTRA_ARGS=("${@:4}")
fi
FUTURE_INDEX=${FUTURE_INDEX:-0}
PREDICT_FUTURE_STEP=${PREDICT_FUTURE_STEP:-30}
# Ensure /predict endpoint even if user passes host:port
if [[ "$SERVER_URL" != */predict ]]; then
  SERVER_URL="${SERVER_URL%/}/predict"
fi
export PYTHONPATH="$PWD/VILA:${PYTHONPATH:-}"
export LEROBOT_VIDEO_BACKEND="${LEROBOT_VIDEO_BACKEND:-pyav}"
if [[ -f "$DATA_ROOT/$LEROBOT_TASK_DIR/norm_stats.json" ]]; then
  MIN_ACTION=$(python3 - <<PY
import json
data=json.load(open("$DATA_ROOT/$LEROBOT_TASK_DIR/norm_stats.json"))
print(min(data["norm_stats"]["actions"]["min"]))
PY
)
  MAX_ACTION=$(python3 - <<PY
import json
data=json.load(open("$DATA_ROOT/$LEROBOT_TASK_DIR/norm_stats.json"))
print(max(data["norm_stats"]["actions"]["max"]))
PY
)
  echo "[test] action clip range from stats: min_action=$MIN_ACTION max_action=$MAX_ACTION"
fi

python tools/lerobot_test_serve.py \
  --data_root "$DATA_ROOT" \
  --server_url "$SERVER_URL" \
  --future_index "$FUTURE_INDEX" \
  --predict_future_step "$PREDICT_FUTURE_STEP" \
  --max_samples "$MAX_SAMPLES" \
  --task_dir "${LEROBOT_TASK_DIR:-}" \
  --rollout_stride "${ROLLOUT_STRIDE:-0}" \
  "${EXTRA_ARGS[@]}"
