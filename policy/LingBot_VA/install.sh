# XPolicyLab deploy: policy server env=lingbot_va; run setup_eval_policy_server.sh with this env.
#!/usr/bin/env bash
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LINGBOT_ROOT="${POLICY_DIR}/lingbot_va"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
CONDA_ENV="${LINGBOT_VA_CONDA_ENV:-lingbot_va}"

source "$(conda info --base)/etc/profile.d/conda.sh"

if [[ "${LINGBOT_SKIP_CONDA_CREATE:-0}" != "1" ]]; then
  if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    conda create -n "${CONDA_ENV}" python=3.10.6 -y
  fi
fi

conda activate "${CONDA_ENV}"

# cu128, not upstream's cu126: a cu126 build has no sm_120 kernels, so it runs
# on Ampere and Ada but dies on Blackwell (RTX PRO 6000) with "no kernel image
# is available for execution on the device" at the first CUDA op -- which is
# where MHBench trains. cu128 covers sm_80/86/89/90/120, so one env serves
# every pool here. LINGBOT_TORCH_CUDA overrides it.
TORCH_CUDA="${LINGBOT_TORCH_CUDA:-cu128}"
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 \
  --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
pip install websockets einops diffusers==0.36.0 transformers==4.55.2 accelerate msgpack opencv-python matplotlib ftfy easydict
pip install packaging ninja safetensors pyarrow pandas datasets jsonlines imageio imageio-ffmpeg av tqdm
# flash-attn is optional: it is only reached through attn_mode="flashattn", and
# neither training (flex attention) nor MHBench serving (torch SDPA) uses it.
# The source build takes about an hour against this torch, so it is opt-in.
if [[ "${LINGBOT_WITH_FLASH_ATTN:-0}" == "1" ]]; then
  pip install flash-attn --no-build-isolation
fi
# lerobot without its deps (it pins an older torchvision); wandb WITH them --
# wan_va.train imports wandb at module scope and --no-deps leaves it missing
# `click`, which is a crash before the first step rather than a missing log.
pip install lerobot==0.3.3 scipy --no-deps
pip install wandb

cd "${LINGBOT_ROOT}"
# --no-deps: the pyproject requires flash_attn (an hour-long source build that
# nothing here reaches -- training uses flex attention, MHBench serving torch
# SDPA) and pins numpy<2, which torch 2.9 does not agree with.
pip install -e . --no-deps

cd "${XPOLICYLAB_ROOT}"
pip install -e .

echo "[LingBot_VA] Done. conda activate ${CONDA_ENV}"
