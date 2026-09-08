#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

cmd=(
  "$REPO_ROOT/.venv/bin/python"
  "$SCRIPT_DIR/eval_simple.py"
  --preset simple_local
  --model-path "${MODEL_PATH:-/hfm/zhenyu/psi/checkpoints/pretrained_mixed_scratch_downstream/checkpoint-50000/}"
  --data-dir "${RUN_DATA_DIR:-/hfm/data/simple/simple/G1WholebodyBendPick-v0-psi0}"
  --num-episodes "${NUM_EPISODES:-10}"
  --num-workers "${NUM_WORKERS:-1}"
)
if [[ -n "${PORT:-}" ]]; then
  cmd+=(--port "$PORT")
fi

exec "${cmd[@]}" "$@"
