#!/bin/bash
set -euo pipefail

# LeRobot G1 finetune entrypoint
export PYTHONPATH="$PWD/VILA:${PYTHONPATH:-}"
bash training_scripts/robot_finetuning/lerobot_g1_finetune.sh
