#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=eval_%x_%j.out
#
# LatentToM evaluation on one MHBench task. ONE policy server hosting both arms
# stays up for the whole job; the sim client runs ONE EPISODE PER PROCESS,
# because a second episode in the same process reliably crashes Kit after a
# grasp force-limit break and no reset reachable from Python clears it.
#
#   MHBENCH_ROOT=/path/to/MHBench \
#     sbatch -J latenttom-eval eval_50.sbatch <ckpt_name> [env_cfg] [episodes] [seed]
#
# Add whatever your scheduler needs on the sbatch line -- partition, QOS,
# account, node excludes. Nothing here assumes a particular cluster.
#
# | variable       | default   | |
# |----------------|-----------|---|
# | MHBENCH_ROOT   | required  | the MHBench checkout |
# | POLICY_ENV     | latenttom | conda env holding the policy server |
# | MHBENCH_SIF    | unset     | run the sim client inside this image; see below |
# | MHBENCH_BINDS  | unset     | comma-separated paths to bind into it |
# | EVAL_DEVICE    | cpu       | physics device |
# | STEP_LIMIT     | 1000      | steps before an episode is called unfinished |
set -euo pipefail

CKPT=${1:?usage: sbatch eval_50.sbatch <ckpt_name> [env_cfg_type] [episodes] [seed]}
ENV_CFG=${2:-cocarry}
EPISODES=${3:-50}
SEED=${4:-0}

BENCH=mhbench
ACTION=joint
POLICY_ENV=${POLICY_ENV:-latenttom}
STEP_LIMIT=${STEP_LIMIT:-1000}
# Pink IK is Pinocchio and runs on CPU, so a GPU physics device fails rather
# than goes faster -- MHBench README, "Stack (pinned)".
EVAL_DEVICE=${EVAL_DEVICE:-cpu}

WT=${MHBENCH_ROOT:?set MHBENCH_ROOT to the MHBench checkout}
XPL=$WT/baselines/XPolicyLab
POLICY_DIR=$XPL/policy/LatentToM
SIM_PY=$WT/.venv/bin/python

OUT_DIR=$POLICY_DIR/eval_results/${ENV_CFG}/${CKPT}
EP_DIR=$OUT_DIR/.episodes
VIDEO_ROOT=$OUT_DIR/videos
RESULTS=$OUT_DIR/results.json
STATUS_TSV=$EP_DIR/client_status.tsv

echo "[eval] ckpt=$CKPT env_cfg=$ENV_CFG episodes=$EPISODES seed=$SEED device=$EVAL_DEVICE"

# The client speaks XPolicyLab's websocket protocol, whose wire format is not
# part of the Isaac install. Checked here, or it is a ModuleNotFoundError
# several minutes into a booted simulator, once per episode.
"$SIM_PY" -c "import msgpack, msgpack_numpy" 2>/dev/null || {
    echo "[eval] the sim venv cannot speak the client protocol. Install it with:" >&2
    echo "       uv pip install --python $SIM_PY 'msgpack>=1.0.8' 'msgpack-numpy>=0.4.8'" >&2
    exit 2
}

# Optional container, client only: Isaac Sim's wheels need a newer glibc than
# some hosts have. Point MHBENCH_SIF at an image that satisfies them and
# MHBENCH_BINDS at the filesystems this checkout lives on. Unset on a new
# enough host and no container is involved. The policy server stays outside it
# either way, which is what lets it hold both checkpoints across all episodes.
sim_prefix=()
if [ -n "${MHBENCH_SIF:-}" ]; then
    [ -f "$MHBENCH_SIF" ] || { echo "[eval] no image at $MHBENCH_SIF" >&2; exit 2; }
    sim_prefix=(singularity exec --nv)
    [ -n "${MHBENCH_BINDS:-}" ] && sim_prefix+=(--bind "$MHBENCH_BINDS")
    # --nv injects the driver's libraries but not its data files, and OptiX
    # reads its denoiser weights from /usr/share/nvidia. Without them every
    # frame keeps its path-trace speckle. Bind the directory, not the file.
    [ -d /usr/share/nvidia ] && sim_prefix+=(--bind /usr/share/nvidia:/usr/share/nvidia)
    sim_prefix+=("$MHBENCH_SIF")
fi

# -u off here: /etc/bashrc and conda's shell functions both read variables
# before setting them, which is fatal under -u and not ours to fix.
set +u
source ~/.bashrc
conda activate "$POLICY_ENV"
set -u

mkdir -p "$EP_DIR" "$VIDEO_ROOT"
rm -rf "${EP_DIR:?}"/* "${VIDEO_ROOT:?}"/*   # a canceled run's leftovers would pollute the summary
: > "$STATUS_TSV"

PORT=$(bash "$XPL/utils/get_free_port.sh")
cleanup() { [ -n "${SERVER_PID:-}" ] && kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT

bash "$POLICY_DIR/setup_eval_policy_server.sh" \
    "$BENCH" "$CKPT" "$CKPT" "$ENV_CFG" "$ACTION" "$SEED" 0 "$POLICY_ENV" "$PORT" \
    > "$OUT_DIR/server.log" 2>&1 &
SERVER_PID=$!
bash "$XPL/utils/wait_for_policy_server.sh" localhost "$PORT" "$SERVER_PID" "LatentToM policy server" 1200

for ((i = 0; i < EPISODES; i++)); do
    printf -v idx '%03d' "$i"
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "[eval] server died before episode $idx"; break; }
    ep_json=$EP_DIR/ep${idx}.json
    echo "[eval] === episode $idx ($((i + 1))/$EPISODES), seed $((SEED + i)) ==="
    t0=$SECONDS; rc=0
    "${sim_prefix[@]}" bash -c "
set -euo pipefail
cd '$WT'
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y MHBENCH_LOG=0
export PYTHONPATH='$WT/source/mhbench'
'$SIM_PY' scripts/eval_policy_xpolicylab.py \
  --bench_name '$BENCH' --task_name '$CKPT' --env_cfg_type '$ENV_CFG' \
  --policy_name LatentToM --host localhost --port '$PORT' \
  --root_dir '$WT/baselines' --device '$EVAL_DEVICE' --seed '$((SEED + i))' \
  --eval_episode_num 1 --episode_step_limit '$STEP_LIMIT' \
  --results_path '$ep_json' --video_dir '$VIDEO_ROOT'
" || rc=$?
    printf '%s\t%s\t%s\n' "$idx" "$rc" "$((SECONDS - t0))" >> "$STATUS_TSV"
    echo "[eval] episode $idx done rc=$rc (${SECONDS}s total)"
done

python3 - "$RESULTS" "$EP_DIR" "$STATUS_TSV" <<'PY'
import json, sys
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
            entry["episode"], entry["client_rc"] = i, rc
    episodes.append(entry or {"success": False, "reason": f"client_crash_rc{rc}",
                              "steps": None, "episode": i, "wall_s": wall, "client_rc": rc})

n, successes = len(episodes), sum(1 for e in episodes if e.get("success"))
reasons = Counter(e["reason"] for e in episodes)
steps = sorted(e["steps"] for e in episodes if e.get("steps") is not None)
summary = {k: (meta or {}).get(k) for k in
           ("bench_name", "task_name", "env_cfg_type", "policy_name", "episode_step_limit")}
summary.update(episodes_run=n, successes=successes,
               success_rate=round(successes / n, 4) if n else None,
               outcomes={r: {"n": c, "rate": round(c / n, 4)} for r, c in reasons.most_common()},
               steps_median=steps[len(steps) // 2] if steps else None,
               steps_min=steps[0] if steps else None, steps_max=steps[-1] if steps else None,
               protocol="one-process-per-episode", final=True, episodes=episodes)
results_path.write_text(json.dumps(summary, indent=2))
print(f"[aggregate] {successes}/{n} success -> {results_path}")
for r, c in reasons.most_common():
    print(f"[aggregate]   {r:24s} {c:3d}  {c / n:5.1%}")
PY

rm -rf "$EP_DIR"
echo "[eval] done -- results: $RESULTS  videos: $VIDEO_ROOT"
