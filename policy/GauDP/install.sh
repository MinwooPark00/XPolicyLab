#!/usr/bin/env bash
set -euo pipefail

# Install GauDP into the *currently active* conda/virtual environment, matching
# XPolicyLab's LatentToM convention. This script never installs or imports a
# separate Policy-Lightning checkout.
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${POLICY_DIR}/../.." && pwd)"
CUROPE_DIR="${POLICY_DIR}/gaudp/third_party/noposplat/model/encoder/backbone/croco/curope"

TORCH_VERSION="${GAUDP_TORCH_VERSION:-2.1.2}"
TORCHVISION_VERSION="${GAUDP_TORCHVISION_VERSION:-0.16.2}"
TORCHAUDIO_VERSION="${GAUDP_TORCHAUDIO_VERSION:-2.1.2}"
TORCH_INDEX_URL="${GAUDP_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu118}"
RASTERIZER_URL="${GAUDP_RASTERIZER_URL:-git+https://github.com/rmurai0610/diff-gaussian-rasterization-w-pose.git}"

echo "[GauDP] python=$(command -v python)"
echo "[GauDP] environment=${CONDA_DEFAULT_ENV:-${VIRTUAL_ENV:-<not activated>}}"
if [[ -z "${CONDA_PREFIX:-}" && -z "${VIRTUAL_ENV:-}" ]]; then
    echo "[GauDP] WARNING: no conda/virtual environment is active; a dedicated environment is recommended." >&2
fi
python - <<'PY'
import sys

if sys.version_info < (3, 10) or sys.version_info >= (3, 12):
    raise SystemExit(
        f"GauDP requires Python 3.10 or 3.11; current interpreter is {sys.version.split()[0]}"
    )
PY

# PyTorch 2.1's cpp_extension imports pkg_resources, which was removed from
# newer setuptools releases. Keep the build frontend compatible with the
# default torch 2.1.2/cu118 extension toolchain.
python -m pip install --upgrade pip "setuptools<81" wheel packaging ninja

if [[ "${GAUDP_SKIP_TORCH_INSTALL:-0}" != "1" ]]; then
    echo "[GauDP] installing PyTorch ${TORCH_VERSION} from ${TORCH_INDEX_URL}"
    python -m pip install \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        "torchaudio==${TORCHAUDIO_VERSION}" \
        --index-url "${TORCH_INDEX_URL}"
else
    echo "[GauDP] keeping the environment's existing PyTorch installation"
fi

# The CUDA wheels include the runtime libraries, but building cuRoPE and the
# Gaussian rasterizer also requires a matching CUDA toolkit (nvcc).  When the
# script is running inside conda, install that toolkit into the active
# environment if it is not already available.  System CUDA installations and
# explicit GAUDP_NVCC/CUDA_HOME selections always take precedence.
if [[ "${GAUDP_SKIP_CUDA_EXTENSIONS:-0}" != "1" ]]; then
    NVCC_CANDIDATE="${GAUDP_NVCC:-}"
    if [[ -z "${NVCC_CANDIDATE}" && -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
        NVCC_CANDIDATE="${CUDA_HOME}/bin/nvcc"
    fi
    if [[ -z "${NVCC_CANDIDATE}" ]]; then
        NVCC_CANDIDATE="$(command -v nvcc || true)"
    fi

    if [[ -z "${NVCC_CANDIDATE}" || ! -x "${NVCC_CANDIDATE}" ]]; then
        if [[ "${GAUDP_AUTO_INSTALL_CUDA_TOOLKIT:-1}" == "1" \
            && -n "${CONDA_PREFIX:-}" \
            && -n "$(command -v conda || true)" ]]; then
            TORCH_CUDA_VERSION="$(python - <<'PY'
import torch

if torch.version.cuda is None:
    raise SystemExit("installed PyTorch is CPU-only; GauDP requires a CUDA build")
print(torch.version.cuda)
PY
)"
            CUDA_TOOLKIT_CHANNEL="${GAUDP_CUDA_TOOLKIT_CHANNEL:-nvidia/label/cuda-${TORCH_CUDA_VERSION}.0}"
            echo "[GauDP] nvcc not found; installing CUDA toolkit ${TORCH_CUDA_VERSION} into ${CONDA_PREFIX}"
            echo "[GauDP] CUDA toolkit channel=${CUDA_TOOLKIT_CHANNEL}"
            # cuda-toolkit is a meta-package.  Using the unversioned nvidia
            # channel can let conda satisfy its lower bounds with newer CUDA
            # components (for example, an 11.8 meta-package with nvcc 13.x).
            # Restrict CUDA packages to the matching versioned label.
            conda install -y --override-channels \
                -c "${CUDA_TOOLKIT_CHANNEL}" \
                -c defaults \
                "cuda-toolkit=${TORCH_CUDA_VERSION}"
            NVCC_CANDIDATE="${CONDA_PREFIX}/bin/nvcc"
            if [[ ! -x "${NVCC_CANDIDATE}" ]]; then
                echo "[GauDP] CUDA toolkit installation completed but nvcc was not found at ${NVCC_CANDIDATE}." >&2
                exit 2
            fi
            export GAUDP_NVCC="${NVCC_CANDIDATE}"
            export CUDA_HOME="${CONDA_PREFIX}"
        else
            echo "[GauDP] nvcc is required to build cuRoPE and the Gaussian rasterizer." >&2
            echo "[GauDP] Activate a conda environment for automatic toolkit installation," >&2
            echo "[GauDP] install a toolkit matching torch.version.cuda manually, or set" >&2
            echo "[GauDP] GAUDP_SKIP_CUDA_EXTENSIONS=1 for data/schema-only usage." >&2
            exit 2
        fi
    fi
fi

# Pinned policy/NoPoSplat Python dependencies. XPolicyLab's own lightweight
# server dependencies are installed by its editable install below.
python -m pip install -r "${POLICY_DIR}/requirements.txt"
python -m pip install -e "${XPL_ROOT}"

if [[ "${GAUDP_SKIP_CUDA_EXTENSIONS:-0}" != "1" ]]; then
    NVCC_BIN="${GAUDP_NVCC:-}"
    if [[ -z "${NVCC_BIN}" && -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
        NVCC_BIN="${CUDA_HOME}/bin/nvcc"
    fi
    if [[ -z "${NVCC_BIN}" ]]; then
        NVCC_BIN="$(command -v nvcc || true)"
    fi
    if [[ -z "${NVCC_BIN}" || ! -x "${NVCC_BIN}" ]]; then
        echo "[GauDP] nvcc is required to build cuRoPE and the Gaussian rasterizer." >&2
        echo "[GauDP] Install a CUDA toolkit matching torch.version.cuda, or set" >&2
        echo "[GauDP] GAUDP_SKIP_CUDA_EXTENSIONS=1 for data/schema-only usage." >&2
        exit 2
    fi
    export GAUDP_NVCC_BIN="${NVCC_BIN}"
    export CUDA_HOME="${CUDA_HOME:-$(cd "$(dirname "${NVCC_BIN}")/.." && pwd)}"

    # PyTorch's extension builder rejects a CUDA major-version mismatch. Fail
    # before a lengthy build with an actionable error.
    python - <<'PY'
import re
import os
import subprocess
import torch

nvcc = os.environ["GAUDP_NVCC_BIN"]
output = subprocess.check_output([nvcc, "--version"], text=True)
match = re.search(r"release\s+(\d+)\.(\d+)", output)
if match is None:
    raise SystemExit("could not determine CUDA version from 'nvcc --version'")
toolkit = f"{match.group(1)}.{match.group(2)}"
runtime = torch.version.cuda
if runtime is None:
    raise SystemExit("installed PyTorch is CPU-only; GauDP requires a CUDA build")
if toolkit.split(".")[0] != runtime.split(".")[0]:
    raise SystemExit(
        "CUDA major-version mismatch: "
        f"nvcc={toolkit}, PyTorch={runtime}. Install a matching toolkit/wheel; "
        "override GAUDP_TORCH_* variables when using CUDA 12."
    )
print(f"[GauDP] CUDA toolchain: {nvcc}={toolkit}, PyTorch={runtime}")
PY

    echo "[GauDP] building vendored cuRoPE (MAX_JOBS=${MAX_JOBS:-4})"
    MAX_JOBS="${MAX_JOBS:-4}" python -m pip install \
        --no-build-isolation --no-cache-dir --force-reinstall --no-deps "${CUROPE_DIR}"

    echo "[GauDP] building NoPoSplat Gaussian rasterizer"
    MAX_JOBS="${MAX_JOBS:-4}" python -m pip install \
        --no-build-isolation --no-cache-dir --force-reinstall --no-deps "${RASTERIZER_URL}"
fi

verify_args=()
if [[ "${GAUDP_SKIP_CUDA_EXTENSIONS:-0}" == "1" ]]; then
    verify_args+=(--skip-cuda-extensions)
fi
if [[ "${GAUDP_SKIP_PIP_CHECK:-0}" != "1" ]]; then
    python -m pip check
fi
python "${POLICY_DIR}/verify_install.py" "${verify_args[@]}"

echo "[GauDP] installation complete"
