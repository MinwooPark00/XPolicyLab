# Ψ₀ with SONIC — Real-World Deployment Guide

Please run the following commands.

## 1. Launch the camera server on the robot (keep it running):

```bash
conda activate vision
cd ~/SONIC_psi0_release
python realsense_server.py
```

## 2. Launch the policy server on the workstation (keep it running):

```bash
bash ./scripts/deploy/serve_psi0-rtc-sonic.sh
```

## 3. Launch the whole-body controller on the robot (keep it running):

```bash
bash ./real/scripts/deploy_psi0-sonic-rtc-robot.sh
```

When you see "Init done." You can press the **]** button to set the robot stand up. After that, you can press **ENTER** to start deploying the policy. When you finish the deployment, you can press **ENTER** again to stop the policy and set the robot back to the default pose.

## 4. Launch the policy client on the workstation:

```bash
bash ./real/scripts/deploy_psi0-sonic-rtc-client.sh
```