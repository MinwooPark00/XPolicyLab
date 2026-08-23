#!/bin/bash
#SBATCH --job-name=latenttom
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --output=train_%x_%j.out
#SBATCH --error=train_%x_%j.err
#
# Train one LatentToM run.
#
#   MHBENCH_ROOT=/path/to/MHBench \
#     sbatch -J latenttom-cocarry train.sbatch <task> [<ckpt_name> [hydra overrides...]]
#
# <task> is an env_cfg_type: cocarry, handover, door_passage, frame_hang. It
# picks the dataset (datasets/<task>/lerobot) and the run directory, and the
# checkpoints of two tasks never collide.
#
# Add whatever your scheduler needs on the sbatch line -- partition, QOS,
# account, node excludes. Nothing here assumes a particular cluster.
#
# | variable          | default   | |
# |-------------------|-----------|---|
# | MHBENCH_ROOT      | required  | the MHBench checkout |
# | POLICY_ENV        | latenttom | conda env built by install.sh |
# | LATENTTOM_DATA_DIR| unset     | share one converted dataset across a sweep |
set -euo pipefail

WT=${MHBENCH_ROOT:?set MHBENCH_ROOT to the MHBench checkout}
POLICY_ENV=${POLICY_ENV:-latenttom}
POLICY_DIR=$WT/baselines/XPolicyLab/policy/LatentToM

task=${1:?usage: train.sbatch <cocarry|handover|door_passage|frame_hang> [ckpt_name] [overrides...]}
shift
ckpt_name=${1:-$task}
shift 2>/dev/null || true

echo "Running on node(s): ${SLURM_NODELIST:-local}"

# -u off here: /etc/bashrc and conda's shell functions both read variables
# before setting them, which is fatal under -u and not ours to fix.
set +u
source ~/.bashrc
module purge   # removes the cluster's cuda path injections, if any
conda activate "$POLICY_ENV"
set -u

cd "$POLICY_DIR"
echo "[train] task=${task} ckpt_name=${ckpt_name} overrides: $*"

bash train.sh mhbench "${ckpt_name}" "${task}" joint "${SEED:-0}" 0 \
    training.num_epochs=600 \
    training.checkpoint_every=10 \
    "$@"

# Writes checkpoints/mhbench-<ckpt_name>-<task>-joint-<seed>/, which is the name
# baselines/scripts/serve/LatentToM.sh serves from. Resubmitting the
# same line continues that run: epochs count to num_epochs in total.
#
# Useful overrides: logging.entity=<team>, training.debug=true (2 epochs).
