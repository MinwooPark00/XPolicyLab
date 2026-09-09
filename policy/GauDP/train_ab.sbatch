#!/usr/bin/env bash
# GauDP recipe training. Account-portable: every path is an argument or an env
# var, so another account runs this unchanged.
#
#   sbatch -p <gpu partition> --qos=<qos> train_ab.sbatch [ARM] [EPOCHS]
#
# Required env (or edit a site copy):
#   GAUDP_DATA      the joint hdf5
#   GAUDP_GAUSSIAN  gaussian run dir holding best.ckpt and features.hdf5
#   GAUDP_OUT       where checkpoints and metrics.jsonl go (use scratch, ~2 GB/arm)
#   GAUDP_ENV       conda env name (default: gaudp)
#
# 24 GB is not enough at batch 128 (~25 GiB allocated); use a 48 GB card.
#SBATCH --gres=gpu:1
#SBATCH --time=14:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# sbatch runs a spool copy of this file, so BASH_SOURCE points at
# /var/spool/slurmd/... -- the checkout is SLURM_SUBMIT_DIR (or GAUDP_HOME when
# the job is submitted from somewhere else).
HERE=${GAUDP_HOME:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}}
[ -f "$HERE/train_ab.py" ] || { echo "no train_ab.py in $HERE -- set GAUDP_HOME" >&2; exit 2; }
ARM=${1:-S02}
EPOCHS=${2:-150}
: "${GAUDP_DATA:?set GAUDP_DATA to the joint hdf5}"
: "${GAUDP_GAUSSIAN:?set GAUDP_GAUSSIAN to the gaussian run dir}"
: "${GAUDP_OUT:?set GAUDP_OUT to a writable checkpoint dir}"

set +u; source ~/.bashrc; conda activate "${GAUDP_ENV:-gaudp}"; set -u
mkdir -p "$HERE/logs"
echo "[ab] node=${SLURM_NODELIST:-local} arm=$ARM epochs=$EPOCHS python=$(which python)"
python -c "import torch;print('[ab] torch',torch.__version__,'device',torch.cuda.get_device_name(0))"

PYTHONNOUSERSITE=1 python "$HERE/train_ab.py" \
  --arm "$ARM" \
  --data "$GAUDP_DATA" \
  --gaussian "$GAUDP_GAUSSIAN/best.ckpt" \
  --gaussian-features "$GAUDP_GAUSSIAN/features.hdf5" \
  --output "$GAUDP_OUT/$ARM" \
  --epochs "$EPOCHS" --num-workers "${GAUDP_WORKERS:-5}" \
  --save-every 10 --full-probe-every 10 \
  "${@:3}"
