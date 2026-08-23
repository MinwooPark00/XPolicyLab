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
# `env_cfg_type` names baselines/env_cfg/<name>.yml -- cocarry, handover,
# door_passage, frame_hang -- and scripts/export_lerobot.py writes that task
# under datasets/<task>/lerobot/. Two things stand between the two names: the
# task directories are spelled without the underscore (datasets/doorpassage,
# datasets/framehang), which is the same pair of spellings
# baselines/scripts/eval_policy.sbatch already accepts on its command line; and
# the export is the `lerobot/` child, not the task directory itself, which also
# holds the raw HDF5 and the dataset card. Both were wrong here, so the default
# never resolved and every run had to pass MHBENCH_DATASET_PATH.
#
# Override for an export kept anywhere else:
#   MHBENCH_DATASET_PATH=/data/cocarry_v2/lerobot bash process_data.sh ...
task_dir="${env_cfg_type//_/}"
MHBENCH_DATASET_PATH="${MHBENCH_DATASET_PATH:-${MHBENCH_ROOT}/datasets/${task_dir}/lerobot}"
if [[ ! -d "${MHBENCH_DATASET_PATH}/meta" ]]; then
  echo "[LatentToM] no LeRobot export at ${MHBENCH_DATASET_PATH} (needs meta/ beside data/)." >&2
  echo "[LatentToM] Run scripts/export_lerobot.py for '${task_dir}' in the MHBench repo," >&2
  echo "[LatentToM] or point MHBENCH_DATASET_PATH at an export kept elsewhere." >&2
  exit 1
fi

# Say which export this is. There are several copies of `datasets/` on a
# machine that has more than one worktree of this repo, they are NOT
# interchangeable, and nothing downstream can tell them apart: a scene gets
# redesigned and the directory keeps its name. cocarry went from a board on two
# stands to a laundry basket in a kitchen between 2026-08-14 and 2026-08-21,
# same `datasets/cocarry/lerobot` path, and a policy trained on one and
# evaluated in the other produces a video rather than an error.
if [[ -f "${MHBENCH_DATASET_PATH}/meta/mhbench_provenance.json" ]]; then
  python3 - "${MHBENCH_DATASET_PATH}" <<'PROV'
import json, sys, pathlib
root = pathlib.Path(sys.argv[1])
prov = json.loads((root / "meta" / "mhbench_provenance.json").read_text())
info = json.loads((root / "meta" / "info.json").read_text())
written = prov.get("env_args", {}).get("provenance", {})
print(f"[LatentToM] source   {root}")
print(f"[LatentToM] exported {written.get('written_at', '?')[:19]} @ {str(written.get('commit', '?'))[:7]}"
      f"  {info.get('total_episodes')} episodes / {info.get('total_frames')} frames")
sentences = prov.get("task", {}).get("sentences") or ["?"]
print(f"[LatentToM] task     {sentences[0]}")
PROV
fi

out_dir="${POLICY_DIR}/data/${data_setting}"

max_demos_args=()
if [[ -n "${expert_data_num}" ]]; then
  max_demos_args=(--max-demos "${expert_data_num}")
fi

python "${POLICY_DIR}/convert_to_replay_buffer.py" \
    "${MHBENCH_DATASET_PATH}" "${out_dir}" \
    "${max_demos_args[@]}"

echo "[LatentToM] wrote ${out_dir}/{replay_buffer.zarr,videos/}"
