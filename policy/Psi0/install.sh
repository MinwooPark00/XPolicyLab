#!/usr/bin/env bash
# XPolicyLab deploy: policy server env=uv; run setup_eval_policy_server.sh with this env.
#
#   bash policy/Psi0/install.sh
#
# Builds the Psi0 environment at psi0/.venv -- inside the vendored checkout, and
# named .venv, which is what policy/Pi_05 and policy/GR00T_N17 do and what this
# repo's .gitignore covers (upstream calls it .venv-psi; PSI_VENV is how its own
# scripts are pointed at one, and nothing here runs them).
#
# Two deliberate departures from upstream's README:
#
#   torch      cu128, not the PyPI default (cu126). cu128 is the only 2.7.0 build
#              that carries sm_120 kernels, so it is the one wheel that runs on
#              every card this benchmark might use -- A100 (sm_80), A6000 (86),
#              4090 (89) and RTX PRO 6000 (120). PSI0_TORCH_BACKEND overrides it.
#   flash-attn optional, and off by default. PyPI ships it as an sdist, so
#              installing it means a source build; and Psi0 needs it only as an
#              optimisation -- models/psi0.py picks flash_attention_2 when
#              transformers reports it available and sdpa otherwise, so the model
#              is correct either way. PSI0_FLASH_ATTN=1 attempts the build and
#              the install still succeeds if it fails.
set -euo pipefail

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PSI_ROOT="${POLICY_DIR}/psi0"
XPOLICYLAB_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
TORCH_BACKEND="${PSI0_TORCH_BACKEND:-cu128}"

echo "[Psi0] PSI_ROOT=${PSI_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install via: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi
[ -f "${PSI_ROOT}/pyproject.toml" ] || {
  echo "no Psi0 checkout at ${PSI_ROOT} -- clone https://github.com/physical-superintelligence-lab/Psi0" >&2
  exit 1; }

cd "${PSI_ROOT}"
export UV_LINK_MODE=copy GIT_LFS_SKIP_SMUDGE=1
# --allow-existing so a re-run (a failed sync, a new dependency) tops the
# environment up instead of refusing; PSI0_CLEAN_VENV=1 starts over.
venv_flag=--allow-existing
[ "${PSI0_CLEAN_VENV:-0}" = 1 ] && venv_flag=--clear
uv venv .venv --python 3.11 "$venv_flag"
uv sync --group serve --group psi --index-strategy unsafe-best-match

# Place the CUDA build the cards here need, over whatever the lock resolved.
# The version alone matches across CUDA builds, so uv considers cu126 to satisfy
# a cu128 request and does nothing: compare the local version tag and force the
# reinstall ourselves.
have=$(.venv/bin/python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo none)
case "${have}" in
  *"+${TORCH_BACKEND}") echo "[Psi0] torch ${have} already matches ${TORCH_BACKEND}" ;;
  *)
    echo "[Psi0] torch is ${have}; installing the ${TORCH_BACKEND} build"
    uv pip install --python .venv/bin/python --torch-backend="${TORCH_BACKEND}" \
      --reinstall-package torch --reinstall-package torchvision \
      "torch==2.7.0" "torchvision==0.22.0" ;;
esac

# flash-attn from the release wheel that matches this interpreter, torch build
# and C++ ABI -- PyPI ships only an sdist, and building it takes the better part
# of an hour. Not fatal if it fails: both the training and the serving path fall
# back to sdpa, and the training one probes an actual kernel rather than trusting
# the import, so a wheel with no kernels for the card is caught too.
if [ "${PSI0_FLASH_ATTN:-1}" = 1 ]; then
  abi=$(.venv/bin/python -c 'import torch; print("TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE")')
  wheel="https://github.com/Dao-AILab/flash-attention/releases/download/v${PSI0_FLASH_ATTN_VERSION:-2.7.4.post1}/flash_attn-${PSI0_FLASH_ATTN_VERSION:-2.7.4.post1}+cu12torch2.7cxx11abi${abi}-cp311-cp311-linux_x86_64.whl"
  uv pip install --python .venv/bin/python "${wheel}" \
    || echo "[Psi0] flash-attn did not install -- training and serving fall back to sdpa" >&2
fi

# The policy server imports XPolicyLab.policy.Psi0.model, so this env has to
# carry XPolicyLab itself.
uv pip install --python .venv/bin/python -e "${XPOLICYLAB_ROOT}"

# scripts/train.py opens with `assert load_dotenv()`, so the file has to exist
# before anything runs. Written from .env.sample with this site's caches, and
# gitignored (upstream's .gitignore covers it) because that is where the tokens
# would go -- wandb here authenticates through ~/.netrc, so none are set.
#
# Two deliberate omissions from the sample:
#   CUDA_LAUNCH_BLOCKING=true   serialises every kernel launch. It is a debugging
#                               aid and it would cost this run most of its speed.
#   the empty HF_TOKEN / WANDB_API_KEY lines
#                               load_dotenv would export them as empty strings,
#                               which reads as "a key is set" to code that only
#                               checks presence.
# PSI0_KEEP_ENV=1 leaves an existing file alone.
if [ -f .env ] && [ "${PSI0_KEEP_ENV:-0}" = 1 ]; then
  echo "[Psi0] keeping the existing .env"
else
  cat > .env <<ENVEOF
WANDB_ENTITY=${WANDB_ENTITY:-mhbench_baselines}
PSI_HOME=${PSI_ROOT}
DATA_HOME=${POLICY_DIR}/data
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
TORCH_HOME=${TORCH_HOME:-$HOME/.cache/torch}
HF_LEROBOT_HOME=${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}
OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
TOKENIZERS_PARALLELISM=false
DEEPSPEED_LOG_LEVEL=warning
TF_CPP_MIN_LOG_LEVEL=3
AV_LOG_LEVEL=quiet
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENVEOF
  echo "[Psi0] wrote ${PSI_ROOT}/.env"
fi

.venv/bin/python - <<'PY'
import torch, psi
print(f"[Psi0] psi {psi.__version__}, torch {torch.__version__} "
      f"(cuda {torch.version.cuda}, arch {torch.cuda.get_arch_list()})")
import XPolicyLab; print("[Psi0] XPolicyLab ok")
try:
    from transformers.utils import is_flash_attn_2_available
    print(f"[Psi0] attention backend: "
          f"{'flash_attention_2' if is_flash_attn_2_available() else 'sdpa'}")
except Exception:
    print("[Psi0] attention backend: sdpa")
PY

echo "[Psi0] Installation finished."
echo "[Psi0] Activate: source ${PSI_ROOT}/.venv/bin/activate"
