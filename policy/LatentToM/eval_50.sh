#!/bin/bash
# N episodes against ONE persistent policy server -- baselines/scripts/
# eval_groot_decentralized.sbatch's pattern, ported off SLURM/singularity.
# Bypasses eval.sh (which restarts the server every episode too) and calls
# eval_policy_xpolicylab.py directly per episode: XPolicyLab@b7c86e0 found a
# second episode in the same client process reliably crashes Kit after a
# grasp force-limit break, so the client alone is restarted per episode.
#
#   bash eval_50.sh mhbench base base cocarry joint 0 0 0 mhbench_ltm mhbench_ltm
#   bash eval_50.sh mhbench base base cocarry joint 0 0 0 mhbench_ltm mhbench_ltm 50 eval_results/base
#
# STEP_LIMIT and EVAL_DEVICE (default cpu) are env-var overrides.
set -uo pipefail

bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
policy_gpu_id=${7}
env_gpu_id=${8}
policy_conda_env=${9}
num_episodes=${10:-50}
out_dir=${11:-"$(dirname "${BASH_SOURCE[0]}")/eval_results/${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"}
step_limit=${STEP_LIMIT:-1000}
device=${EVAL_DEVICE:-cpu}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
MHBENCH_ROOT="$(cd "${BENCH_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"
POLICY_NAME="$(basename "${SCRIPT_DIR}")"
SIM_PY="${MHBENCH_ROOT}/.venv/bin/python"

ep_dir="${out_dir}/.episodes"
video_dir="${out_dir}/videos"
results="${out_dir}/results.json"
status_tsv="${ep_dir}/client_status.tsv"
mkdir -p "${ep_dir}" "${video_dir}"
rm -rf "${ep_dir:?}"/* "${video_dir:?}"/*   # leftovers from a canceled run would pollute the aggregate
: > "${status_tsv}"

policy_server_port=$(bash "${UTILS_DIR}/get_free_port.sh")
echo "=== eval_50: task=${env_cfg_type} episodes=${num_episodes} seed=${seed} port=${policy_server_port} step_limit=${step_limit} device=${device} ==="

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        echo "=== eval_50: kill server ${SERVER_PID} ==="
        kill "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

bash "${SCRIPT_DIR}/setup_eval_policy_server.sh" \
    "${bench_name}" "${task_name}" "${ckpt_name}" "${env_cfg_type}" "${action_type}" \
    "${seed}" "${policy_gpu_id}" "${policy_conda_env}" "${policy_server_port}" \
    > "${out_dir}/server.log" 2>&1 &
SERVER_PID=$!

bash "${UTILS_DIR}/wait_for_policy_server.sh" localhost "${policy_server_port}" "${SERVER_PID}" "Policy server" 1200

failures=0
for ((i = 0; i < num_episodes; i++)); do
    printf -v idx '%03d' "$i"
    episode_seed=$((seed + i))
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "=== eval_50: policy server died before episode ${idx}; stopping ==="
        break
    fi
    ep_json="${ep_dir}/ep${idx}.json"
    echo "=== eval_50: episode ${idx} ($((i + 1))/${num_episodes}), seed ${episode_seed} ==="
    ep_t0=${SECONDS}
    rc=0
    ( cd "${MHBENCH_ROOT}" \
      && OMNI_KIT_ACCEPT_EULA=YES PYTHONPATH="${MHBENCH_ROOT}/source/mhbench" \
         "${SIM_PY}" scripts/eval_policy_xpolicylab.py \
            --bench_name "${bench_name}" --task_name "${task_name}" \
            --env_cfg_type "${env_cfg_type}" --policy_name "${POLICY_NAME}" \
            --host localhost --port "${policy_server_port}" --root_dir "${BENCH_ROOT}" \
            --device_id "${env_gpu_id}" --device "${device}" --seed "${episode_seed}" \
            --eval_episode_num 1 --episode_step_limit "${step_limit}" \
            --results_path "${ep_json}" --video_dir "${video_dir}" \
    ) || rc=$?
    printf '%s\t%s\t%s\n' "${idx}" "${rc}" "$((SECONDS - ep_t0))" >> "${status_tsv}"
    echo "=== eval_50: episode ${idx} done rc=${rc} (${SECONDS}s elapsed total) ==="
    [[ "${rc}" -ne 0 ]] && failures=$((failures + 1))
done

# Fold the per-episode JSONs (each client's own --results_path) into one
# run summary -- same aggregation eval_groot_decentralized.sbatch does.
python3 - "${results}" "${ep_dir}" "${status_tsv}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

results_path, ep_dir, status_tsv = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
status = {}
for line in status_tsv.read_text().splitlines():
    idx, rc, wall = line.split("\t")
    status[int(idx)] = (int(rc), int(wall))

episodes, meta = [], None
for i in sorted(status):
    rc, wall = status[i]
    ep_file = ep_dir / f"ep{i:03d}.json"
    entry = None
    if ep_file.is_file():
        payload = json.loads(ep_file.read_text())
        meta = meta or payload
        if payload.get("episodes"):
            entry = dict(payload["episodes"][0])
            entry["episode"] = i
            entry["client_rc"] = rc
    if entry is None:
        entry = {
            "success": False, "reason": f"client_crash_rc{rc}", "steps": None,
            "episode": i, "seed": (meta or {}).get("seed", 0) + i if meta else None,
            "wall_s": wall, "client_rc": rc,
        }
    episodes.append(entry)

successes = sum(1 for e in episodes if e.get("success"))
n = len(episodes)
reasons = Counter(e["reason"] for e in episodes)
steps = sorted(e["steps"] for e in episodes if e.get("steps") is not None)
summary = {k: (meta or {}).get(k) for k in
           ("bench_name", "task_name", "env_cfg_type", "policy_name", "episode_step_limit")}
summary.update(
    episodes_run=n, successes=successes,
    success_rate=round(successes / n, 4) if n else None,
    outcomes={r: {"n": c, "rate": round(c / n, 4)} for r, c in reasons.most_common()},
    steps_median=steps[len(steps) // 2] if steps else None,
    steps_min=steps[0] if steps else None, steps_max=steps[-1] if steps else None,
    protocol="one-process-per-episode", final=True, episodes=episodes,
)
results_path.write_text(json.dumps(summary, indent=2))
print(f"[aggregate] {successes}/{n} success -> {results_path}")
for r, c in reasons.most_common():
    print(f"[aggregate]   {r:24s} {c:3d}  {c / n:5.1%}")
PY

rm -rf "${ep_dir}"   # folded into results.json
echo "=== eval_50: ${num_episodes} episodes done, ${failures} client failures ==="
echo "=== eval_50:   results: ${results} ==="
echo "=== eval_50:   videos : ${video_dir} ==="
[[ "${failures}" -eq 0 ]]
