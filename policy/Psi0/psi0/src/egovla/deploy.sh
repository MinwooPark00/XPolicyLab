#!/bin/bash
set -euo pipefail

MODEL_PATH=${1:-checkpoints/lerobot-g1-actionvec-100e-lr2e-5-b4-1}
HOST=${2:-0.0.0.0}
PORT=${3:-8009}
export PYTHONPATH="$PWD:$PWD/VILA:${PYTHONPATH:-}"
DATA_ROOT=${DATA_ROOT:-/hfm/data/real_teleop_g1/lerobot}
LEROBOT_TASK_DIR=${LEROBOT_TASK_DIR:-Pick_bottle_and_turn_and_pour_into_cup}
NORM_STATS_PATH="$DATA_ROOT/$LEROBOT_TASK_DIR/norm_stats.json"
if [[ -f "$NORM_STATS_PATH" ]]; then
  MIN_ACTION=$(python3 - <<PY
import json
data=json.load(open("$NORM_STATS_PATH"))
print(min(data["norm_stats"]["actions"]["min"]))
PY
)
  MAX_ACTION=$(python3 - <<PY
import json
data=json.load(open("$NORM_STATS_PATH"))
print(max(data["norm_stats"]["actions"]["max"]))
PY
)
else
  MIN_ACTION=-0.06
  MAX_ACTION=0.06
fi
echo "Using action clip range: min_action=$MIN_ACTION max_action=$MAX_ACTION"

python tools/lerobot_serve.py \
  --model_name_or_path "$MODEL_PATH" \
  --output_dir /tmp/lerobot_serve \
  --version qwen2 \
  --vision_tower google/siglip-so400m-patch14-384 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --traj_decoder_type transformer_action_vector \
  --traj_action_output_dim 36 \
  --proprio_size 32 \
  --action_tokenizer uniform \
  --min_action "$MIN_ACTION" \
  --max_action "$MAX_ACTION" \
  --num_action_bins 256 \
  --raw_action_label True \
  --mask_input True \
  --predict_future_step 30 \
  --image_aspect_ratio resize \
  --merge_hand False \
  --use_mano False \
  --sep_proprio False \
  --sep_query_token False \
  --next_token_loss_coeff 0.0 \
  --host "$HOST" \
  --port "$PORT"
