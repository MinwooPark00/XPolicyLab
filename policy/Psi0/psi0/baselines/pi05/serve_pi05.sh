#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <task> [ckpt_step] [port]"
    echo "  <task>       : Task name (required)"
    echo "  [ckpt_step]  : Checkpoint step (default: 40000)"
    echo "  [port]       : Port to serve on (default: 9000)"
    exit 1
fi

source .venv-openpi/bin/activate

export task=$1
export ckpt_step=${2:-40000}
export port=${3:-9000}


export UV_ENV_FILE=.venv-openpi 


export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

echo "Serving Openpi05 on GPU $CUDA_VISIBLE_DEVICES"

python src/openpi/deploy/serve_policy.py \
    --port=$port \
    policy:checkpoint \
    --policy.config=$task \
    --policy.dir=.runs/openpi-05/$task/$task/$ckpt_step
