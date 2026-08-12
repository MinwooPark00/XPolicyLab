#!/bin/bash
set -euo pipefail

# Version pins follow policy/DP/install.sh (same diffusion_policy lineage,
# proven to install cleanly on a modern CUDA stack.) pytorch3d itself is skipped
# entirely: model/common/rotation_transformer.py (the only thing that needs
# it) is excluded from the vendored copy -- see README.md.
pip install torch==2.4.1 torchvision
pip install \
  zarr==2.12.0 wandb ipdb gpustat \
  omegaconf hydra-core==1.2.0 dill==0.3.5.1 \
  einops==0.4.1 diffusers==0.11.1 numba==0.56.4 \
  moviepy imageio av imagecodecs matplotlib termcolor sympy \
  h5py opencv-python numpy==1.23.5 threadpoolctl \
  huggingface_hub==0.25.2 pandas

# install XPolicyLab itself, so `XPolicyLab.policy.LatentToM.model` and
# `client_server.ws` resolve in this policy's environment (same as
# policy/demo_policy/install.sh -- LatentToM has no separate pyproject.toml).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
python -m pip install -e "${XPL_ROOT}"

# The XPolicyLab install above re-bumps numpy past 1.23.5 (its own
# dependencies pull in a newer one), which breaks numba==0.56.4 (numpy<1.24
# required) -- diffusion_policy's ReplayBuffer/sampler code needs numba, so
# re-pin numpy back down as the last step rather than leaving it as a manual
# gotcha (policy/DP/install.sh has the same issue, undocumented-fixed there).
pip install "numpy==1.23.5"
