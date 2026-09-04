#!/usr/bin/env bash

# Shared argument parsing for GauDP launchers. New task-first calls keep the
# fixed MHBench/GauDP fields implicit; the original XPolicyLab form remains
# available for existing jobs and external callers.
#
# GauDP is a centralized joint-space policy, so its artifacts are named the way
# every other joint-space baseline's are and the way baselines/scripts/
# eval_policy.sbatch's default `run_names` looks for them:
#
#     data/mhbench-<task>-unitree_g1x2_centralized-joint.hdf5
#     checkpoints/mhbench-<task>-unitree_g1x2_centralized-joint-<seed>/
#
# `scene` stays the plain task name (cocarry, handover, ...) because the
# pre-joint Gaussian runs were filed under it -- see gaudp_gaussian_run_dirs.

GAUDP_ACTION_TYPE=joint
GAUDP_DEFAULT_ENV_CFG=unitree_g1x2_centralized

gaudp_task_config() {
    task=$1
    case "${task}" in
        cocarry) scene=cocarry ;;
        handover|handover_easy|handovereasy) task=handover; scene=handover ;;
        door_passage|doorpassage) task=door_passage; scene=door_passage ;;
        frame_hang|framehang) task=frame_hang; scene=frame_hang ;;
        *) scene=${task} ;;
    esac
    env_cfg=${GAUDP_ENV_CFG:-${GAUDP_DEFAULT_ENV_CFG}}
}

gaudp_parse_stage_args() {
    if (( $# >= 6 )) && [[ "$4" == "joint" || "$4" == "ee" ]]; then
        # Legacy: <bench> <ckpt> <env_cfg> <action_type> <seed> <gpu> [extra...]
        bench=$1
        ckpt=$2
        env_cfg=$3
        action_type=$4
        seed=$5
        gpu=$6
        task=${GAUDP_TASK:-${ckpt}}
        scene=${GAUDP_SCENE:-${task}}
        extra=("${@:7}")
        return
    fi

    # Short: <task> [seed] [gpu] [extra...]
    task=${1:?usage: $0 <task> [seed] [gpu] [extra args]}
    shift
    gaudp_task_config "${task}"
    bench=${GAUDP_BENCH:-mhbench}
    ckpt=${GAUDP_CKPT:-${scene}}
    action_type=${GAUDP_ACTION_TYPE}
    seed=${GAUDP_SEED:-0}
    gpu=${GAUDP_GPU:-0}
    if (( $# > 0 )) && [[ "$1" =~ ^[0-9]+$ ]]; then seed=$1; shift; fi
    if (( $# > 0 )) && [[ "$1" =~ ^[0-9]+$ ]]; then gpu=$1; shift; fi
    extra=("$@")
}

gaudp_parse_data_args() {
    if (( $# >= 4 )) && [[ "$4" == "joint" || "$4" == "ee" ]]; then
        # Legacy: <bench> <ckpt> <env_cfg> <action_type> [task] [max_demos]
        bench=$1
        ckpt=$2
        env_cfg=$3
        action_type=$4
        task=${5:-${GAUDP_TASK:-${ckpt}}}
        max_demos=${6:-}
        # Older five-argument calls used argument 5 for max_demos.
        if [[ "${task}" =~ ^[0-9]+$ ]]; then
            max_demos=${task}
            task=${GAUDP_TASK:-${ckpt}}
        fi
        scene=${GAUDP_SCENE:-${task}}
        return
    fi

    # Short: <task> [max_demos]
    task=${1:?usage: $0 <task> [max_demos]}
    max_demos=${2:-}
    gaudp_task_config "${task}"
    bench=${GAUDP_BENCH:-mhbench}
    ckpt=${GAUDP_CKPT:-${scene}}
    action_type=${GAUDP_ACTION_TYPE}
}

gaudp_require_joint_action_type() {
    if [[ "${action_type}" != "joint" ]]; then
        echo "[GauDP] action_type=${action_type} is no longer supported." >&2
        echo "[GauDP] GauDP trains and serves the centralized GR00T joint contract" >&2
        echo "[GauDP] (86D joint state / 70D absolute joint targets). The EE datasets and" >&2
        echo "[GauDP] policy checkpoints under *-ee-* are kept but cannot be resumed into" >&2
        echo "[GauDP] joint space -- convert with process_data.sh and retrain." >&2
        exit 2
    fi
}

# The run directory this stage writes into.
gaudp_run_dir() {
    printf '%s\n' "${POLICY_DIR}/checkpoints/${bench}-${ckpt}-${env_cfg}-${action_type}-${seed}"
}

gaudp_data_path() {
    printf '%s\n' "${POLICY_DIR}/data/${bench}-${ckpt}-${env_cfg}-${action_type}.hdf5"
}

# Where a Gaussian artifact may live, newest naming first.
#
# The NoPoSplat encoder and its 94 GB feature cache are functions of the RGB
# frames and the camera geometry alone -- neither reads a state or an action --
# so the joint-space switch does not invalidate them, and re-running a 26 h
# finetune plus a multi-hour extraction to land byte-identical output would be
# pure waste. The pre-switch runs are named for the scene with `-ee-`, so those
# directories are searched after the current one. `gaudp/dataset.py`'s
# `_validate_feature_source` is the end that actually proves a reused cache
# lines up with the new dataset (same export, same episodes, same cameras);
# this only says where to look.
gaudp_gaussian_run_dirs() {
    printf '%s\n' \
        "$(gaudp_run_dir)" \
        "${POLICY_DIR}/checkpoints/${bench}-${ckpt}-${scene}-ee-${seed}" \
        "${POLICY_DIR}/checkpoints/${scene}-experiment-${scene}-ee-${seed}"
    # The one-arm dataset and its Gaussian cache were produced while the shipped
    # task was still named `handover_easy`.  New joint-policy runs use the renamed
    # `handover` task, but the RGB frames and camera geometry behind this cache did
    # not change, so keep the old artifact discoverable without reverting the new
    # run/data naming.
    if [[ "${task}" == "handover" ]]; then
        printf '%s\n' \
            "${POLICY_DIR}/checkpoints/${bench}-handover_easy-handover_easy-ee-${seed}"
    fi
}

# First existing `gaussian/<name>` across those directories, else empty.
gaudp_find_gaussian_artifact() {
    local name=$1 candidate
    while read -r candidate; do
        [[ -n "${candidate}" ]] || continue
        if [[ -s "${candidate}/gaussian/${name}" ]]; then
            printf '%s\n' "${candidate}/gaussian/${name}"
            return 0
        fi
    done < <(gaudp_gaussian_run_dirs)
    return 1
}
