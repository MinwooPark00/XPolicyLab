#!/usr/bin/env bash

# Resolve (and, when needed, download) the official NoPoSplat initialization.
# This file is sourced by the Gaussian train/evaluation launchers.
resolve_noposplat_checkpoint() {
    local data=$1
    local policy_dir=$2
    local pretrained="${GAUDP_NOPOSPLAT_CKPT:-}"
    local python_bin="${GAUDP_PYTHON:-python}"

    if ! command -v "${python_bin}" >/dev/null 2>&1; then
        echo "[GauDP] Python executable not found: ${python_bin}; activate the GauDP environment or set GAUDP_PYTHON" >&2
        return 2
    fi

    if [[ -n "${pretrained}" ]]; then
        if [[ ! -s "${pretrained}" ]]; then
            echo "[GauDP] NoPoSplat checkpoint not found or empty: ${pretrained}" >&2
            return 2
        fi
        printf '%s\n' "${pretrained}"
        return 0
    fi

    local num_views
    num_views="$(PYTHONNOUSERSITE=1 "${python_bin}" - "${data}" <<'PY'
import json
import sys

import h5py

with h5py.File(sys.argv[1], "r") as source:
    print(len(json.loads(source.attrs["camera_order"])))
PY
)"

    local default_filename
    if (( num_views > 2 )); then
        default_filename="re10k_3views.ckpt"
    else
        default_filename="re10k.ckpt"
    fi

    local repo_id="${GAUDP_NOPOSPLAT_REPO:-botaoye/NoPoSplat}"
    local filename="${GAUDP_NOPOSPLAT_FILENAME:-${default_filename}}"
    local weights_dir="${GAUDP_NOPOSPLAT_DIR:-${policy_dir}/weights}"
    pretrained="${weights_dir}/${filename}"

    if [[ ! -s "${pretrained}" ]]; then
        echo "[GauDP] downloading NoPoSplat checkpoint ${repo_id}/${filename}" >&2
        echo "[GauDP] destination=${pretrained}" >&2
        GAUDP_DOWNLOAD_REPO="${repo_id}" \
        GAUDP_DOWNLOAD_FILENAME="${filename}" \
        GAUDP_DOWNLOAD_DIR="${weights_dir}" \
        PYTHONNOUSERSITE=1 "${python_bin}" - <<'PY'
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

download_dir = Path(os.environ["GAUDP_DOWNLOAD_DIR"]).expanduser().resolve()
download_dir.mkdir(parents=True, exist_ok=True)
hf_hub_download(
    repo_id=os.environ["GAUDP_DOWNLOAD_REPO"],
    filename=os.environ["GAUDP_DOWNLOAD_FILENAME"],
    local_dir=download_dir,
    local_dir_use_symlinks=False,
)
PY
        echo "[GauDP] downloaded NoPoSplat checkpoint: ${pretrained}" >&2
    else
        echo "[GauDP] reusing NoPoSplat checkpoint: ${pretrained}" >&2
    fi

    if [[ ! -s "${pretrained}" ]]; then
        echo "[GauDP] NoPoSplat checkpoint not found or empty: ${pretrained}" >&2
        return 2
    fi
    printf '%s\n' "${pretrained}"
}
