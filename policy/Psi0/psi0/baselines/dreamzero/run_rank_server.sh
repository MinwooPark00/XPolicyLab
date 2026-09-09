#!/bin/bash
# Per-rank launcher for the DreamZero SIMPLE policy server (psi0 baseline).
#
# torchrun invokes this once per rank (via --no-python); each rank activates the
# dreamzero env — same setup as baselines/dreamzero/simple_training_lora.sh — and
# exec's the FastAPI server. The server reads RANK / LOCAL_RANK / WORLD_SIZE to
# build the multi-GPU device mesh; rank 0 serves HTTP. Example:
#
# cd /hfm/songlin/psi0
# CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
#     --no-python bash baselines/dreamzero/run_rank_server.sh \
#     --model-path .runs/dreamzero/G1WholebodyLocomotionPickBetweenTablesTeleop-v0-lora \
#     --pretrained-base $PSI_HOME/cache/checkpoints/DreamZero-AgiBot \
#     --prompt-cache $PSI_HOME/cache/dreamzero/prompt_cache_g1_simple.pt \
#     --port 22085
set -e

base_triton="${TRITON_CACHE_DIR:-/mnt/beegfs/scratch/$USER/.cache/triton}"
base_ind="${TORCHINDUCTOR_CACHE_DIR:-/mnt/beegfs/scratch/$USER/.cache/torchinductor}"
export TRITON_CACHE_DIR="${base_triton}/rank${LOCAL_RANK:-0}"
export TORCHINDUCTOR_CACHE_DIR="${base_ind}/rank${LOCAL_RANK:-0}"
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

# Resolve to psi0 project root so relative paths (.env, .venv-dreamzero, src/,
# baselines/) work regardless of where torchrun was launched from.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Activate the dreamzero env (matches baselines/dreamzero/simple_training_lora.sh).
source .venv-dreamzero/bin/activate
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

# Ensure cuDNN sub-libraries load from the active env rather than a mismatched
# system/container build. See baselines/dreamzero/README.md.
_ENV_PREFIX="${CONDA_PREFIX:-${VIRTUAL_ENV:-}}"
_CUDNN_LIB="${_ENV_PREFIX}/lib/python3.11/site-packages/nvidia/cudnn/lib"
if [ -d "$_CUDNN_LIB" ]; then
    export LD_LIBRARY_PATH="${_CUDNN_LIB}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

exec python3 baselines/dreamzero/serve_dreamzero_g1_simple.py "$@"
