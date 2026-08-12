#!/bin/bash
set -e

# Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [rotation_rep]
# Output convention: data/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<rotation_rep>/
#   replay_buffer.zarr/ + videos/<episode_idx>/<camera_idx>.mp4 -- upstream
#   LatentToM's own on-disk format (real_data_to_replay_buffer /
#   XarmSplitActionDataset). rotation_rep defaults to "quat" (22D/arm);
#   "rot6d" (26D/arm) is the other option -- see
#   diffusion_policy/common/rotation_conversion.py.
#
# convert_to_replay_buffer.py reads MHBench's raw per-episode trajectory
# HDF5 directly (scripts/data_convertion.py's DemoSource format -- a single
# .hdf5, a directory of <stem>.demo_N.hdf5 shards, or a dataset root above
# one, e.g. datasets/hf/mhbench_cocarry_test) and writes the layout above in
# one pass. No DP-format zarr intermediate and no separate MHBench .venv
# subprocess -- this policy's own env already has h5py (install.sh) on top
# of the zarr/av it already needed to write videos+lowdim.
bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
expert_data_num=${5:-}
rotation_rep=${6:-quat}

if [[ "${rotation_rep}" != "quat" && "${rotation_rep}" != "rot6d" ]]; then
  echo "[LatentToM] rotation_rep must be 'quat' or 'rot6d', got '${rotation_rep}'" >&2
  exit 1
fi

data_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${rotation_rep}"
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MHBENCH_ROOT="$(cd "${POLICY_DIR}/../../../.." && pwd)"
# The raw MHBench trajectory dataset to convert -- a single .hdf5, a shard
# directory, or a dataset root (see convert_to_replay_buffer.py's docstring).
# Override for a specific dataset, e.g.
# MHBENCH_DATASET_PATH=datasets/hf/mhbench_cocarry_test.
MHBENCH_DATASET_PATH="${MHBENCH_DATASET_PATH:-${MHBENCH_ROOT}/datasets/data}"

out_dir="${POLICY_DIR}/data/${data_setting}"

max_demos_args=()
if [[ -n "${expert_data_num}" ]]; then
  max_demos_args=(--max-demos "${expert_data_num}")
fi

python "${POLICY_DIR}/convert_to_replay_buffer.py" \
    "${MHBENCH_DATASET_PATH}" "${out_dir}" \
    --rotation-rep "${rotation_rep}" \
    "${max_demos_args[@]}"

echo "[LatentToM] wrote ${out_dir}/{replay_buffer.zarr,videos/}"
