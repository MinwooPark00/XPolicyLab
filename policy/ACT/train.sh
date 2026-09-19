#!/bin/bash

bench_name=${1}
ckpt_name=${2} # run name
env_cfg_type=${3}
action_type=${4}
seed=${5}
gpu_id=${6}

DEBUG=False

export CUDA_VISIBLE_DEVICES=${gpu_id}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get Action Dimension from env_cfg_type
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
UTILS_DIR="${ROOT_DIR}/XPolicyLab/utils"
action_dim=$(bash "${UTILS_DIR}/get_action_dim.sh" "${ROOT_DIR}" "${env_cfg_type}"); echo -e "\033[33m[INFO] Action dim: ${action_dim}\033[0m"
state_dim=$(bash "${UTILS_DIR}/get_state_dim.sh" "${ROOT_DIR}" "${env_cfg_type}"); echo -e "\033[33m[INFO] State dim: ${state_dim}\033[0m"

# ---- MHBench experiment setting (ACT_VARIANT) -------------------------------
# What a policy observes and what it predicts, as two independent choices. The
# default is the benchmark's own: a per-robot policy on its own observation
# predicting its own action. The other three are options of the same trainer:
#
#   dtde      own obs  -> own action            env_cfg_type unitree_g1x2_decentralized
#   jointobs  both obs -> own action            same profile, state doubled
#   jointact  own obs  -> both robots' action   same profile, action doubled;
#                                               only its own half is executed (model.py)
#   ctce      both obs -> both actions, one policy: env_cfg_type unitree_g1x2_centralized,
#             whose profile already is the doubled pair, so nothing changes here
#
# The profile's dims are one robot's, so the joint settings double the side they
# widen rather than naming 86 and 70. The variant also joins the TASK_CONFIGS
# key (register_task_config.py --variant writes the same one), because a joint
# setting trains a different dataset under the same ckpt_name and profile.
ACT_VARIANT=${ACT_VARIANT:-dtde}
variant_suffix=""
case "${ACT_VARIANT}" in
    dtde|ctce) ;;
    jointobs) state_dim=$((state_dim * 2)); variant_suffix="-jointobs" ;;
    jointact) action_dim=$((action_dim * 2)); variant_suffix="-jointact" ;;
    *) echo "[ACT] unknown ACT_VARIANT '${ACT_VARIANT}' (dtde|jointobs|jointact|ctce)" >&2; exit 2 ;;
esac
if [ -n "${variant_suffix}" ] && [ -z "${CKPT_TAG:-}" ]; then
    # Untagged, the run would write into -- and the eval would then serve -- the
    # default setting's checkpoint directory.
    echo "[ACT] ACT_VARIANT=${ACT_VARIANT} needs a CKPT_TAG (baselines/train/ACT.sh sets it to the variant)" >&2
    exit 2
fi
echo -e "\033[33m[INFO] ACT_VARIANT=${ACT_VARIANT}: state ${state_dim}D, action ${action_dim}D\033[0m"
export ACT_ACTION_DIM=${action_dim}
export ACT_STATE_DIM=${state_dim}
ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"
# CKPT_TAG separates runs over the same data (a probe, a hyperparameter sweep)
# the way GR00T's and FastWAM's do -- without it a tagged run would write into,
# and resume from, the untagged run's directory while the launcher logged a
# tagged name.
ckpt_dir="${SCRIPT_DIR}/checkpoints/${ckpt_setting}-${seed}${CKPT_TAG:+-${CKPT_TAG}}"

# The benchmark's shared budget for the policies that read no language: batch
# 256 for 600 epochs, the same as Diffusion Policy (baselines/scripts/dp_train.sh).
# An epoch is one pass over every frame -- until 2026-09-09 EpisodicDataset
# indexed episodes, so `--batch_size` above the split size (50) did nothing and
# an "epoch" was one optimizer step; 20000 of them were 20000 steps at batch 50.
BATCH_SIZE=${ACT_BATCH_SIZE:-256}
NUM_EPOCHS=${ACT_NUM_EPOCHS:-600}
# Quarters of the run, so the four checkpoints a sweep evaluates are the four
# written -- the shape GR00T's 10k/20k/30k/40k has.
SAVE_FREQ=${ACT_SAVE_FREQ:-150}
LR=${ACT_LR:-1e-4}

python3 imitate_episodes.py \
    --bench_name ${bench_name} \
    --task_name ${ckpt_name} \
    --ckpt_setting "${ckpt_setting}${variant_suffix}" \
    --ckpt_dir "${ckpt_dir}" \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 50 \
    --hidden_dim 512 \
    --batch_size ${BATCH_SIZE} \
    --dim_feedforward 3200 \
    --num_epochs ${NUM_EPOCHS} \
    --lr ${LR} \
    --save_freq ${SAVE_FREQ} \
    --seed ${seed}
