#!/bin/bash

PORT=8014
# INSTRUCTION="Spray the bowl and wipe it and stack it up."
# INSTRUCTION="Pick toys into box and lift and turn and put on the chair new"
INSTRUCTION="pick up the green grapes and place it into the green bowl"

# psi_rtc_sonic_client.py lives at the GR00T-WholeBodyControl (sonic) submodule root
cd "$(dirname "$0")/../../third_party/GR00T-WholeBodyControl"

# Run in SONIC's .venv_teleop (has gear_sonic + cv2 + zmq + msgpack; websocket-client added for the psi0 RTC client)
./.venv_teleop/bin/python psi_rtc_sonic_client.py \
    --port "$PORT" \
    --instruction "$INSTRUCTION"
