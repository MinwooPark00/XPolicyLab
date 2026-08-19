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

# Robots without an explicit "num_cameras" default to 1 (head_cam only),
# matching diffusion_policy/config/task/default_task.yaml before this field
# existed -- unrelated robots keep training on exactly what they do today.
print(robot_info.get("num_cameras", 1))
' "${ROOT_DIR}" "${env_cfg_type}"
