from __future__ import annotations

import os
import time
from typing import Literal
import dataclasses
from pathlib import Path
import numpy as np
import tyro
import yourdfpy

import viser
from viser.extras import ViserUrdf
from typing import Any
import pinocchio as pin
from fk import G1FK
from g1 import *
from psi.utils import resolve_data_path

@dataclasses.dataclass
class Args:
    urdf: str = "assets/robots/g1/g1_body29_hand14.urdf"
    data_dir: str = "./data/pick_n_squat"
    episode_idx: int = 0
    host: str = "0.0.0.0"
    port: int = 9000

    def __post_init__(self) -> None:
        if not Path(self.data_dir).is_absolute():
            self.data_dir = str(resolve_data_path(self.data_dir))
        if not Path(self.urdf).is_absolute():
            self.urdf = str(resolve_data_path(self.urdf, auto_download=True))

def create_robot_control_sliders(
    server: viser.ViserServer, viser_urdf: ViserUrdf, joint_cfg: dict[str, float] | None = None
) -> tuple[dict[str, viser.GuiInputHandle[float]], dict[str, float]]:
    slider_handles: dict[str, viser.GuiInputHandle[float]] = {}
    initial_config: dict[str, float] = {}
    for joint_name, (
        lower,
        upper,
    ) in viser_urdf.get_actuated_joint_limits().items():
        lower = lower if lower is not None else -np.pi
        upper = upper if upper is not None else np.pi
        initial_pos = joint_cfg[joint_name] if joint_cfg and joint_name in joint_cfg else 0.0

        if initial_pos < lower or initial_pos > upper:
            print(f"Warning: Initial position {initial_pos} for joint {joint_name} is out of bounds [{lower}, {upper}].")

        slider = server.gui.add_slider(
            label=joint_name,
            min=lower,#min(lower,initial_pos),
            max=upper,
            step=1e-3,
            initial_value=np.clip(initial_pos, lower, upper),
        )
        def _on_slider_update(_):
            global suppress_joint_slider_events
            if suppress_joint_slider_events:
                return
            viser_urdf.update_cfg({name: s.value for name, s in slider_handles.items()}) # type: ignore

        slider.on_update(_on_slider_update)
        slider_handles[joint_name] = slider
        initial_config[joint_name] = initial_pos
    return slider_handles, initial_config

def extract_proprio_state(frame: dict[str, Any]) -> dict[str, float]:
    proprio_state = {}
    # proprio_state.update(dict(zip(HAND_JOINT_NAMES, frame["observation.hand_joints"].numpy().tolist())))
    # proprio_state.update(dict(zip(ARM_JOINT_NAMES, frame["observation.arm_joints"].numpy().tolist())))
    # proprio_state.update(dict(zip(LEG_JOINT_NAMES, frame["observation.leg_joints"].numpy().tolist())))
    proprio_state.update(dict(zip(HAND_JOINT_NAMES, frame["states"].numpy()[:14].tolist())))
    proprio_state.update(dict(zip(ARM_JOINT_NAMES, frame["states"].numpy()[14:28].tolist())))
    proprio_state.update(dict(zip(LEG_JOINT_NAMES, [0]*len(LEG_JOINT_NAMES))))
    return proprio_state

def extract_action_joints(frame: dict[str, Any]) -> dict[str, float]:
    action_np = frame["action"].numpy().tolist()
    action = {}
    # action.update(dict(zip(HAND_JOINT_NAMES, action_np[18:32])))
    # action.update(dict(zip(LEG_JOINT_NAMES, action_np[32:])))
    # action.update(dict(zip(ARM_JOINT_NAMES, action_np[4:18])))

    # mobile pick & pack
    # action.update(dict(zip(HAND_JOINT_NAMES, action_np[22:36])))
    # action.update(dict(zip(LEG_JOINT_NAMES, action_np[36:])))
    # action.update(dict(zip(ARM_JOINT_NAMES, action_np[8:22])))

    # g1 real - he2lerobot_ours
    action.update(dict(zip(HAND_JOINT_NAMES, action_np[0:14])))
    action.update(dict(zip(LEG_JOINT_NAMES, [0]*len(LEG_JOINT_NAMES))))
    action.update(dict(zip(ARM_JOINT_NAMES, action_np[14:28])))
    return action

def _matrix_to_quat(rot_matrix):
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)"""
    from scipy.spatial.transform import Rotation
    r = Rotation.from_matrix(rot_matrix)
    quat_xyzw = r.as_quat()  # returns [x, y, z, w]
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])  # convert to [w, x, y, z]


is_playing = False
current_frame = 0
init_base_height = 0.75
play_speed = 10 # jump frames per step

get_base_height = lambda frame: frame["states"][-1]  # frame["observation.prev_height"]
suppress_joint_slider_events = False

def main(args: Args) -> None:
    global current_frame
    server = viser.ViserServer(args.host, args.port)

    # Load the LeRobot dataset
    from psi.data.lerobot.compat import LeRobotDataset
    dataset = LeRobotDataset(args.data_dir, episodes=[args.episode_idx])

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        client.add_notification(
            f"Loaded {len(dataset)} frames.",
            f"from {os.path.basename(args.data_dir)}, episode {args.episode_idx}",
            with_close_button=True,
        )

    # Create GUI controls
    with server.gui.add_folder("Dataset Playback"):
        play_button = server.gui.add_button("Play/Pause")
        frame_slider = server.gui.add_slider(
            "Frame",
            min=0,  # Start from frame 1 since we need prev_row
            max=len(dataset) - 1,
            step=1,
            initial_value=0,
        )
        speed_slider = server.gui.add_slider(
            "Speed",
            min=1,
            max=50,
            step=1,
            initial_value=10,
        )

    @speed_slider.on_update
    def _(_):
        global play_speed
        play_speed = int(speed_slider.value)

    @frame_slider.on_update
    def _(_):
        global current_frame, suppress_joint_slider_events
        current_frame = int(frame_slider.value)
        # update robot qpos
        frame = dataset[current_frame]
        joint_states = extract_proprio_state(frame)
        
        # batch-update sliders without triggering their callbacks
        suppress_joint_slider_events = True
        try:
            for jname, q in joint_states.items():
                slider_handles[jname].value = q
        finally:
            suppress_joint_slider_events = False
        # single robot update after batching slider changes
        viser_urdf.update_cfg(joint_states) # type: ignore
        # update z
        viser_urdf._visual_root_frame.position = (0, 0, get_base_height(frame) - init_base_height) # type: ignore
        # update kinmatics frames
        result = g1.fk(joint_states)
        
        l_ee_pose = result["l_ee"]["position"]
        l_ee_pose[2] += get_base_height(frame) - init_base_height
        l_ee_frame_handle.position = tuple(l_ee_pose)

        r_ee_pose = result["r_ee"]["position"]
        r_ee_pose[2] += get_base_height(frame) - init_base_height
        r_ee_frame_handle.position = tuple(r_ee_pose)

        action = g1.fk(extract_action_joints(frame))
        l_ee_pose_action = action["l_ee"]["position"]
        l_ee_pose_action[2] += get_base_height(frame) - init_base_height
        l_ee_handle.position = tuple(l_ee_pose_action)

        r_ee_pose_action = action["r_ee"]["position"]
        r_ee_pose_action[2] += get_base_height(frame) - init_base_height
        r_ee_handle.position = tuple(r_ee_pose_action)


    @play_button.on_click
    def _(event: viser.GuiEvent):
        global is_playing
        is_playing = not is_playing
        client = event.client
        assert client is not None
        client.add_notification(
            f"Playing" if is_playing else "Paused",
            f"total frames: {len(dataset)}" if is_playing else "Paused",
            with_close_button=True,
        )

    urdf = yourdfpy.URDF.load(args.urdf)
    viser_urdf = ViserUrdf(
        server,
        urdf_or_path=urdf,
        load_meshes=True,
        load_collision_meshes=False,
        collision_mesh_color_override=(1.0, 0.0, 0.0, 0.5),
        # root_node_name="/world/robot",
    )

    frame = dataset[current_frame]
    joint_cfg = extract_proprio_state(frame)
    init_base_height = frame["states"][-1] #frame["observation.prev_height"]
    # init_base_height = frame["observation.odometry.position"][2]

    g1 = G1FK(args.urdf, mode="default")
    
    action = g1.fk(extract_action_joints(frame))
    # actions
    l_ee_handle = server.scene.add_icosphere(
        "/action/l_ee",
        radius=0.05,
        color=(1.0, 0.0, 0.0),
        position=tuple(action["l_ee"]["position"])
    )
    r_ee_handle = server.scene.add_icosphere(
        "/action/r_ee",
        radius=0.05,
        color=(1.0, 0.0, 0.0), 
        position=tuple(action["r_ee"]["position"])
    )

    states = g1.fk(joint_cfg)
    # states
    l_ee_frame_handle = server.scene.add_frame(
        "/states/l_ee_frame",
        wxyz=_matrix_to_quat(np.asarray(states["l_ee"]["matrix"])[:3, :3]),
        position=tuple(states["l_ee"]["position"]),
        axes_length=0.1,
        axes_radius=0.005
    )
    r_ee_frame_handle = server.scene.add_frame(
        "/states/r_ee_frame",
        wxyz=_matrix_to_quat(np.asarray(states["r_ee"]["matrix"])[:3, :3]),
        position=tuple(states["r_ee"]["position"]),
        axes_length=0.1,
        axes_radius=0.005
    )

    # Create sliders in GUI that help us move the robot joints.
    with server.gui.add_folder("Joint position control"):
        (slider_handles, initial_config) = create_robot_control_sliders(
            server, viser_urdf, joint_cfg
        )

     # Add visibility checkboxes.
    with server.gui.add_folder("Visibility"):
        show_meshes_cb = server.gui.add_checkbox(
            "Show meshes",
            viser_urdf.show_visual,
        )
        show_collision_meshes_cb = server.gui.add_checkbox(
            "Show collision meshes", viser_urdf.show_collision
        )

    @show_meshes_cb.on_update
    def _(_):
        viser_urdf.show_visual = show_meshes_cb.value

    @show_collision_meshes_cb.on_update
    def _(_):
        viser_urdf.show_collision = show_collision_meshes_cb.value

    # Hide checkboxes if meshes are not loaded.
    show_meshes_cb.visible = True
    show_collision_meshes_cb.visible = False # doesn't work

    # Set initial robot configuration.
    viser_urdf.update_cfg(initial_config) # type: ignore

    # Create grid.
    trimesh_scene = viser_urdf._urdf.scene or viser_urdf._urdf.collision_scene
    server.scene.add_grid(
        "/grid",
        width=2,
        height=2,
        position=(
            0.0,
            0.0,
            # Get the minimum z value of the trimesh scene.
            trimesh_scene.bounds[0, 2] if trimesh_scene is not None else 0.0,
        ),
    )

    # Create joint reset button.
    reset_button = server.gui.add_button("Reset")

    @reset_button.on_click
    def _(_):
        global suppress_joint_slider_events
        suppress_joint_slider_events = True
        try:
            for jname, init_q in initial_config.items():
                slider_handles[jname].value = init_q
        finally:
            suppress_joint_slider_events = False
        viser_urdf.update_cfg(initial_config) # type: ignore

    # Sleep forever.
    while True:
        if is_playing:
            current_frame = (current_frame + 1*play_speed) % len(dataset)
            frame_slider.value = current_frame

        time.sleep(0.01)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    tyro.cli(main)
