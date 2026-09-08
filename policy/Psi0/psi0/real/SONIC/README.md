# Ψ₀ with SONIC — Teleoperation & Data Collection

We follow the official [SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl) setup. Run all commands from the submodule root `third_party/GR00T-WholeBodyControl`.

## Setup (on the workstation)

Fetch the LFS assets, then install the per-use-case environments with SONIC's install scripts (each creates an isolated `uv` venv):

```bash
git lfs pull

bash install_scripts/install_pico.sh              # .venv_teleop          — VR teleoperation
bash install_scripts/install_data_collection.sh   # .venv_data_collection — LeRobot recorder
bash install_scripts/install_mujoco_sim.sh        # .venv_sim             — MuJoCo simulation

python download_from_hf.py                         # SONIC policy + planner ONNX
```

For the C++ whole-body controller and the PICO VR hardware, follow SONIC's official docs:
- [Deployment build (TensorRT + `just build`)](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_deploy.html)
- [VR teleop setup (XRoboToolkit)](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/vr_teleop_setup.html)

## Camera server (on the robot)

Reuse the `vision` conda env created in the robot [Image Server setup](../README.md#image-server-robot-only) (it already has `pyrealsense2`, `opencv`, `pyzmq`); just add the three remaining packages:

```bash
conda activate vision
pip install msgpack msgpack-numpy tyro
```

Copy the SONIC camera module from the workstation (run from the submodule root; G1 default IP `192.168.123.164`):

```bash
ssh unitree@192.168.123.164 mkdir -p ~/SONIC_psi0_release/gear_sonic
scp gear_sonic/__init__.py gear_sonic/version.py unitree@192.168.123.164:~/SONIC_psi0_release/gear_sonic/
scp -r gear_sonic/camera unitree@192.168.123.164:~/SONIC_psi0_release/gear_sonic/
scp real/SONIC/realsense_server.py unitree@192.168.123.164:~/SONIC_psi0_release/
```

Start the server on the robot (keep it running):

```bash
conda activate vision
cd ~/SONIC_psi0_release
python -m gear_sonic.camera.composed_camera --ego-view-camera realsense --port 5555
```

## Run

Edit `ROBOT_IP` / `TASK` at the top of the script (recording runs at 30 fps to match the camera), then:

```bash
bash ./real/SONIC/scripts/collect_psi0-sonic-data.sh sim   # MuJoCo test (no robot/camera, no recording)
bash ./real/SONIC/scripts/collect_psi0-sonic-data.sh       # real robot — records to outputs/ (LeRobot format)
```

Engage teleop and record per SONIC's [data collection tutorial](https://nvlabs.github.io/GR00T-WholeBodyControl/tutorials/data_collection.html): calibration pose → **A+B+X+Y** → **A+X**, then **left grip + A** to start/stop an episode (**left grip + B** to discard).The data will be saved to `third_party/GR00T-WholeBodyControl/outputs/` in LeRobot format.

No tmux? Run each component in its own terminal instead:

```bash
# sim teleop test
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh sim          # MuJoCo
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh deploy sim   # C++ controller (sim)
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh pico         # PICO streamer
```

```bash
# real robot
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh deploy       # C++ controller
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh pico         # PICO streamer
bash ./real/SONIC/scripts/collect_psi0-sonic-data-manual.sh exporter     # data exporter (records)
```
