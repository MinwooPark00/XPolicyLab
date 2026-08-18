set -uo pipefail

# bash eval_50.sh mhbench cocarry mhbench-cocarry-mhbench_cocarry-joint-0 mhbench_cocarry joint 0 0 0 mhbench_dp mhbench_dp

# bash eval_50.sh mhbench cocarry cocarry unitree_g1x2_decentralized joint 0 0 0 mhbench_dp mhbench_dp


bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
policy_gpu_id=${7}
env_gpu_id=${8}
policy_conda_env=${9}
eval_env_conda_env=${10}
num_episodes=${11:-50}
video_root=${12:-"$(dirname "${BASH_SOURCE[0]}")/videos/${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "${video_root}"

failures=0
for ((i = 0; i < num_episodes; i++)); do
    printf -v idx '%03d' "$i"
    echo "=== eval_50: episode ${idx} ($((i + 1))/${num_episodes}) ==="
    episode_video_dir="${video_root}/${idx}"
    mkdir -p "${episode_video_dir}"

    EVAL_ENV_TYPE=sim EVAL_VIDEO_DIR="${episode_video_dir}" \
        bash "${SCRIPT_DIR}/eval.sh" \
            "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" \
            "${seed}" "${policy_gpu_id}" "${env_gpu_id}" "${policy_conda_env}" "${eval_env_conda_env}"
    status=$?
    if [[ "${status}" -ne 0 ]]; then
        echo "=== eval_50: episode ${idx} FAILED (exit ${status}) ==="
        failures=$((failures + 1))
    fi
done

echo "=== eval_50: ${num_episodes} episodes done, ${failures} failed, videos under ${video_root} ==="
[[ "${failures}" -eq 0 ]]
