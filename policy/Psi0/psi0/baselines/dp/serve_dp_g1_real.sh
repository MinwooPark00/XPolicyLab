#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

source .venv-dp/bin/activate

# Accept RUN_DIR and CKPT_STEP as command line arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 RUN_DIR CKPT_STEP"
    exit 1
fi

RUN_DIR=$1
CKPT_STEP=$2

python src/dp/deploy/dp_g1_serve_real.py \
    --host=0.0.0.0 \
    --port=22085 \
    --run-dir=$RUN_DIR \
    --ckpt-step=$CKPT_STEP \