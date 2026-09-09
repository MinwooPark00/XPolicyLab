#!/bin/bash
# Build the dataset Psi0 trains on.
#
#   bash process_data.sh mhbench multitask unitree_g1x2_decentralized joint
#
# The work itself is MHBench's, because the dataset is a *view* of the flattened
# all-task export every language baseline shares -- one build, five views (see
# baselines/scripts/prepare_multitask_data.sh). This script is the XPolicyLab
# entry point onto it, so the adapter answers `process_data.sh` the way every
# other one does; the output lands at
# data/<bench>-<ckpt>-<env_cfg>-<action>{,_val}, symlinked by that script.
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <action_type>" >&2
  exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_ROOT="${MHBENCH_WT:-$(cd "${POLICY_DIR}/../../../.." && pwd)}"
data_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}"

if [[ "${bench_name}" != "mhbench" || "${ckpt_name}" != "multitask" ]]; then
  echo "policy/Psi0 has only the MHBench shared multitask dataset wired up;" >&2
  echo "asked for ${data_setting}" >&2
  exit 2
fi

[ -x "${BENCH_ROOT}/baselines/scripts/prepare_multitask_data.sh" ] || {
  echo "no MHBench workspace at ${BENCH_ROOT} -- set MHBENCH_WT" >&2; exit 1; }

exec bash "${BENCH_ROOT}/baselines/scripts/prepare_multitask_data.sh" psi0
