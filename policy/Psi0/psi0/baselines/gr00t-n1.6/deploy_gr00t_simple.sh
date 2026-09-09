#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="/hfm/zhenyu/psi/src:/hfm/zhenyu/psi/src/gr00t${PYTHONPATH:+:$PYTHONPATH}"
source ./src/gr00t/.venv/bin/activate
source ~/.env
export TORCHINDUCTOR_DISABLE=1
export TORCH_COMPILE=0
export HF_HOME="${HF_HOME:-/tmp/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HF_HOME}"
MODEL_PATH="${MODEL_PATH:-/hfm/zhenyu/psi/checkpoints/pretrained_g1_ee_downstream/checkpoint-50000/}"
PORT="${PORT:-5556}"

# PROCESSOR_DIR="/hfm/zhenyu/psi/checkpoints/G1WholebodyBendPick-v0-psi0-real/processor"
# if [[ -d "$PROCESSOR_DIR" ]]; then
#   for f in processor_config.json statistics.json embodiment_id.json; do
#     if [[ -f "$PROCESSOR_DIR/$f" ]] && [[ ! -f "$MODEL_PATH/$f" ]]; then
#       cp "$PROCESSOR_DIR/$f" "$MODEL_PATH/$f"
#     fi
#   done
# fi

python -m gr00t.deploy.gr00t_serve_simple \
  --embodiment-tag G1_LOCO_DOWNSTREAM \
  --model-path "$MODEL_PATH" \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port "$PORT" \
  --use-sim-policy-wrapper \
  --strict
