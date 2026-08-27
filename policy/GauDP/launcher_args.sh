#!/usr/bin/env bash

# Shared argument parsing for GauDP launchers. New task-first calls keep the
# fixed MHBench/GauDP fields implicit; the original XPolicyLab form remains
# available for existing jobs and external callers.

gaudp_task_config() {
    task=$1
    case "${task}" in
        cocarry) env_cfg=cocarry ;;
        handover) env_cfg=handover ;;
        handover_easy|handovereasy) task=handover_easy; env_cfg=handover_easy ;;
        door_passage|doorpassage) task=door_passage; env_cfg=door_passage ;;
        frame_hang|framehang) task=frame_hang; env_cfg=frame_hang ;;
        *) env_cfg=${task} ;;
    esac
}

gaudp_parse_stage_args() {
    if (( $# >= 6 )) && [[ "$4" == "ee" ]]; then
        # Legacy: <bench> <ckpt> <env_cfg> ee <seed> <gpu> [extra...]
        bench=$1
        ckpt=$2
        env_cfg=$3
        action_type=$4
        seed=$5
        gpu=$6
        task=${GAUDP_TASK:-${env_cfg}}
        extra=("${@:7}")
        return
    fi

    # Short: <task> [seed] [gpu] [extra...]
    task=${1:?usage: $0 <task> [seed] [gpu] [extra args]}
    shift
    gaudp_task_config "${task}"
    bench=${GAUDP_BENCH:-mhbench}
    ckpt=${GAUDP_CKPT:-${env_cfg}}
    action_type=ee
    seed=${GAUDP_SEED:-0}
    gpu=${GAUDP_GPU:-0}
    if (( $# > 0 )) && [[ "$1" =~ ^[0-9]+$ ]]; then seed=$1; shift; fi
    if (( $# > 0 )) && [[ "$1" =~ ^[0-9]+$ ]]; then gpu=$1; shift; fi
    extra=("$@")
}

gaudp_parse_data_args() {
    if (( $# >= 4 )) && [[ "$4" == "ee" ]]; then
        # Legacy: <bench> <ckpt> <env_cfg> ee [task] [max_demos]
        bench=$1
        ckpt=$2
        env_cfg=$3
        action_type=$4
        task=${5:-${GAUDP_TASK:-${env_cfg}}}
        max_demos=${6:-}
        # Older five-argument calls used argument 5 for max_demos.
        if [[ "${task}" =~ ^[0-9]+$ ]]; then
            max_demos=${task}
            task=${GAUDP_TASK:-${env_cfg}}
        fi
        return
    fi

    # Short: <task> [max_demos]
    task=${1:?usage: $0 <task> [max_demos]}
    max_demos=${2:-}
    gaudp_task_config "${task}"
    bench=${GAUDP_BENCH:-mhbench}
    ckpt=${GAUDP_CKPT:-${env_cfg}}
    action_type=ee
}
