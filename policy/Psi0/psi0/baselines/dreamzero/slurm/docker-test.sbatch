#!/usr/bin/env bash
#SBATCH --job-name=songlinwei-docker-test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/beegfs/scratch/songlinwei/docker-test-%j.out

set -euo pipefail

srun \\
  --container-image=/mnt/beegfs/containers/nvidia-pytorch-25.06-py3-cuda12.9.sqsh \\
  --container-mounts=/mnt/beegfs:/mnt/beegfs \\
  --container-workdir=/mnt/beegfs/scratch/songlinwei \\
  bash -lc '
    hostname
    nvidia-smi
    python - <<PY
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
PY
  '
