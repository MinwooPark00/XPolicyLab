#!/bin/bash
set -euo pipefail
bench_name=$1
task_name=$2
ckpt_name=$3
env_cfg_type=$4
action_type=$5
seed=$6
policy_gpu_id=$7
policy_conda_env=$8
policy_server_port=$9
policy_server_host=${10:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${XPL_ROOT}/policy/${policy_name}/deploy.yml"

source "$("${CONDA_EXE:-conda}" info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

# get_action_dim(env_cfg_type) in XPolicyLab.utils.process_data looks up
# env_cfg/<env_cfg_type>.yml -- that file only exists for the original
# single/bimanual-arm robots (piper, franka, ...), not for MHBench's
# unitree_g1x2_centralized/decentralized, which never got an env_cfg/*.yml of
# their own (their scene lives in MHBench's own Isaac Lab env_cfg tree, not
# XPolicyLab's). utils/get_action_dim.sh and utils/get_state_dim.sh key
# straight off utils/robot/_robot_info.json by env_cfg_type -- the same
# lookup GR00T_N17/setup_eval_policy_server.sh already uses -- so they resolve
# both MHBench env_cfg_types and the original robots without that detour.
action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${BENCH_ROOT}" "${env_cfg_type}")
export ACT_ACTION_DIM="${action_dim}"

state_dim=$(bash "${UTILS_DIR}/get_state_dim.sh" "${BENCH_ROOT}" "${env_cfg_type}")
export ACT_STATE_DIM="${state_dim}"

# ckpt_name is the checkpoint directory under checkpoints/ (full run dir name).
if [[ "${ckpt_name}" == /* ]]; then
    ckpt_dir="${ckpt_name}"
elif [[ -d "${SCRIPT_DIR}/checkpoints/${ckpt_name}" ]]; then
    ckpt_dir="${SCRIPT_DIR}/checkpoints/${ckpt_name}"
elif [[ -d "${SCRIPT_DIR}/${ckpt_name}" ]]; then
    ckpt_dir="${SCRIPT_DIR}/${ckpt_name}"
else
    ckpt_dir="${SCRIPT_DIR}/checkpoints/${ckpt_name}"
fi

echo "[SERVER] policy=${policy_name}, task=${task_name}, policy_server_port=${policy_server_port}, action_dim=${action_dim}"
echo "[SERVER] ckpt_dir=${ckpt_dir}"

exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    python "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            bench_name="${bench_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            ckpt_dir="${ckpt_dir}" \
            env_cfg_type="${env_cfg_type}" \
            seed="${seed}" \
            policy_name="${policy_name}" \
            action_type="${action_type}" \
            action_dim="${action_dim}"