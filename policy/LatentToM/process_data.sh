#!/bin/bash
set -e

# Usage: bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num]
# Output convention: data/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>/
#   replay_buffer.zarr/ + videos/<episode_idx>/<camera_idx>.mp4 -- upstream
#   LatentToM's own on-disk format (real_data_to_replay_buffer /
#   XarmSplitActionDataset).
#
# convert_to_replay_buffer.py reads MHBench's shared LeRobot v2.1 export
# (scripts/export_lerobot.py's output in the main repo, datasets/<task>/lerobot/)
# and writes the layout above in one pass.
bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
expert_data_num=${5:-}

data_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MHBENCH_ROOT="$(cd "${POLICY_DIR}/../../../.." && pwd)"
# The LeRobot export to convert (has meta/ and data/ inside).
#
# Defaults to datasets/<env_cfg_type>, which is where
# scripts/export_lerobot.py writes it for that task: `env_cfg_type` names
# baselines/env_cfg/<name>.yml (cocarry, handover, door_passage) and the task
# directory carries the same name. So a run whose env_cfg_type is `cocarry`
# reads datasets/cocarry without being told twice. Override for
# anything else -- an export kept outside the repo:
#   MHBENCH_DATASET_PATH=/data/cocarry_v2 bash process_data.sh ...
MHBENCH_DATASET_PATH="${MHBENCH_DATASET_PATH:-${MHBENCH_ROOT}/datasets/${env_cfg_type}}"

out_dir="${POLICY_DIR}/data/${data_setting}"

max_demos_args=()
if [[ -n "${expert_data_num}" ]]; then
  max_demos_args=(--max-demos "${expert_data_num}")
fi

python "${POLICY_DIR}/convert_to_replay_buffer.py" \
    "${MHBENCH_DATASET_PATH}" "${out_dir}" \
    "${max_demos_args[@]}"

echo "[LatentToM] wrote ${out_dir}/{replay_buffer.zarr,videos/}"
