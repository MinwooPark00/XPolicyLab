#!/bin/bash
set -e

ROOT_DIR="$1"
env_cfg_type="$2"

python3 -c '
import sys, os, json

root_dir = sys.argv[1]
env_cfg_type = sys.argv[2]

robot_info = json.load(
    open(os.path.join(root_dir, "XPolicyLab", "utils", "robot", "_robot_info.json"), "r", encoding="utf-8")
)[env_cfg_type]

# Robots without an explicit "state_dim" (the original single/bimanual-arm
# configs) are assumed to have qpos_dim == action_dim, same as before this
# script existed.
if "state_dim" in robot_info:
    print(robot_info["state_dim"])
else:
    print(sum(robot_info["arm_dim"]) + sum(robot_info["ee_dim"]))
' "${ROOT_DIR}" "${env_cfg_type}"
