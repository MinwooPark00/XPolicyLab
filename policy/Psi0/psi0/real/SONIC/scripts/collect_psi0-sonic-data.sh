#!/bin/bash

# SONIC teleop with Psi0.
#
#   sim mode  : lightweight teleop test in MuJoCo — no camera render, no recording.
#   real mode : full data collection — starts the C++ controller, PICO streamer,
#               and data exporter, recording LeRobot datasets under outputs/.
#
# Usage:
#   ./collect_psi0-sonic-data.sh              # real robot (camera server at $ROBOT_IP)
#   ./collect_psi0-sonic-data.sh sim          # simulation teleop test (no recording)
#   ./collect_psi0-sonic-data.sh <ROBOT_IP>   # real robot at a specific IP

ROBOT_IP=192.168.123.164
TASK="Pick bottle and turn and pour into cup."
FPS=30   # recording frequency (Hz); 30 matches the RealSense camera so no duplicate frames

# Everything lives in the GR00T-WholeBodyControl (sonic) submodule root.
SONIC_DIR="$(cd "$(dirname "$0")/../../../third_party/GR00T-WholeBodyControl" && pwd)"
cd "$SONIC_DIR"

MODE="${1:-$ROBOT_IP}"

if [ "$MODE" = "sim" ]; then
    # Sim: teleop only. Plain run_sim_loop (no --enable-image-publish/--enable-offscreen),
    # so MuJoCo does NOT render/publish camera images — much smoother, but nothing is recorded.
    SESSION=sonic_sim_teleop
    tmux kill-session -t "$SESSION" 2>/dev/null

    tmux new-session -d -s "$SESSION" -c "$SONIC_DIR"
    tmux set-option -t "$SESSION" -g mouse on   # click a pane to focus it (e.g. to press Y in the deploy pane)
    tmux send-keys -t "${SESSION}:0.0" \
        "source .venv_teleop/bin/activate && python gear_sonic/scripts/run_sim_loop.py" C-m

    tmux split-window -h -t "${SESSION}:0.0" -c "$SONIC_DIR/gear_sonic_deploy"
    tmux send-keys -t "${SESSION}:0.1" \
        "source scripts/setup_env.sh && ./deploy.sh --input-type zmq_manager sim" C-m

    tmux split-window -v -t "${SESSION}:0.1" -c "$SONIC_DIR"
    tmux send-keys -t "${SESSION}:0.2" \
        "source .venv_teleop/bin/activate && python gear_sonic/scripts/pico_manager_thread_server.py --manager" C-m

    tmux select-layout -t "$SESSION" tiled
    tmux attach -t "$SESSION"
else
    # Real robot: full data collection.
    # The launcher's built-in preview runs in the data-collection venv (lerobot's headless
    # OpenCV, no window), so disable it and run a working preview from .venv_teleop alongside.
    ( source .venv_teleop/bin/activate && \
      python gear_sonic/scripts/run_camera_viewer.py --camera-host "$MODE" --camera-port 5555 ) &
    PREVIEW_PID=$!

    python gear_sonic/scripts/launch_data_collection.py \
        --camera-host "$MODE" \
        --task-prompt "$TASK" \
        --data-exporter-frequency "$FPS" \
        --no-camera-viewer

    kill "$PREVIEW_PID" 2>/dev/null
fi
