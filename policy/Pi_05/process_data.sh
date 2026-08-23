#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [raw_task_dirs]" >&2
  exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
expert_data_num_or_raw_task_dirs=${5:-}
raw_task_dirs=${6:-}

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -- MHBench ---------------------------------------------------------------
# Nothing is converted into a policy-specific container: pi0.5 reads MHBench's
# own LeRobot export, and the centralized / decentralized split is done by the
# transforms (openpi.policies.mhbench_policy), exactly as GR00T slices it with
# meta/modality.json. All that is needed is a v3.0 *view* of that export -- the
# pinned lerobot refuses v2.1 outright -- plus normalization statistics.
if [[ "${bench_name}" == "mhbench" ]]; then
  BASELINES_DIR="$(cd "${POLICY_DIR}/../../.." && pwd)"
  MHBENCH_ROOT="$(cd "${BASELINES_DIR}/.." && pwd)"
  case "${env_cfg_type}" in
    unitree_g1x2_centralized)   task="${ckpt_name}";          target="centralized" ;;
    unitree_g1x2_decentralized) task="${ckpt_name%_robot_*}"; target="robot_${ckpt_name##*_robot_}" ;;
    *) echo "unknown mhbench env_cfg_type ${env_cfg_type}" >&2; exit 2 ;;
  esac
  train_config_name="pi05_mhbench_${task}_${target}"
  repo_id="mhbench-${task}"

  : "${HF_LEROBOT_HOME:?set HF_LEROBOT_HOME to the LeRobot datasets root}"
  src="${MHBENCH_DATASETS:-${MHBENCH_ROOT}/datasets}/${task}/lerobot"
  dst="${HF_LEROBOT_HOME}/${repo_id}"

  cd "${POLICY_DIR}/openpi"
  if [[ ! -f "${dst}/meta/info.json" ]]; then
    echo "[Pi_05] building the v3.0 view: ${src} -> ${dst}"
    uv run python "${BASELINES_DIR}/scripts/convert_mhbench_lerobot_v30.py" --src "${src}" --dst "${dst}"
  else
    echo "[Pi_05] v3.0 view already present: ${dst}"
  fi

  # The three targets over one task share the export but not the statistics:
  # they normalize different column subsets.
  if [[ ! -f "assets/${train_config_name}/${repo_id}/norm_stats.json" ]]; then
    echo "[Pi_05] computing norm stats for ${train_config_name}"
    uv run scripts/compute_norm_stats.py --config-name "${train_config_name}"
  else
    echo "[Pi_05] norm stats already present for ${train_config_name}"
  fi
  echo "[Pi_05] process_data done."
  exit 0
fi

# -- RoboDojo (upstream) ---------------------------------------------------
mode="${OPENPI_DATA_MODE:-image}"
py_args=("${bench_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" --mode "${mode}")
if [[ -n "${expert_data_num_or_raw_task_dirs}" ]]; then
  py_args+=("${expert_data_num_or_raw_task_dirs}")
fi
if [[ -n "${raw_task_dirs}" ]]; then
  py_args+=("${raw_task_dirs}")
fi

cd "${POLICY_DIR}/openpi"
python scripts/process_data.py "${py_args[@]}"
