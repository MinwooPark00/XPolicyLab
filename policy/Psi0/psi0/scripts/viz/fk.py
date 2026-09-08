import argparse
import json
import os
import sys

import numpy as np
import pinocchio as pin

LOCKED_MIXED = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
]

LOCKED_FINGERS_ONLY = [
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
]


class G1FK:
    def __init__(self, urdf_path, mesh_dir=None, mode="reduced"):
        if mesh_dir is None:
            mesh_dir = os.path.dirname(urdf_path)
        self.robot_full = pin.RobotWrapper.BuildFromURDF(urdf_path, mesh_dir)

        
        if mode == "reduced":
            to_lock = LOCKED_MIXED
        elif mode == "whole":
            to_lock = LOCKED_FINGERS_ONLY
        elif mode == "default":
            to_lock = []
        else:
            raise ValueError("mode must be 'reduced' or 'whole'")

        if len(to_lock) > 0:
            qref = np.zeros(self.robot_full.model.nq)
            self.model = self.robot_full.buildReducedRobot(
                list_of_joints_to_lock=to_lock, reference_configuration=qref
            ).model
        else:
            self.model = self.robot_full.model

        self.add_frame("l_ee", "left_wrist_yaw_joint", [0.0, 0.0, 0.0])
        self.add_frame("r_ee", "right_wrist_yaw_joint", [0.0, 0.0, 0.0])

        self.add_frame("l_foot", "left_ankle_pitch_joint", [0.0, 0.0, -0.0])
        self.add_frame("r_foot", "right_ankle_pitch_joint", [0.0, 0.0, -0.0])

        self.data = self.model.createData()

    def add_frame(self, frame_name, parent_joint, offset):
        fid = self.model.getFrameId(frame_name)
        exists = fid < self.model.nframes and self.model.frames[fid].name == frame_name
        if not exists:
            jid = self.model.getJointId(parent_joint)
            if jid == 0:
                raise ValueError(f"Joint not found: {parent_joint}")
            self.model.addFrame(
                pin.Frame(
                    frame_name,
                    jid,
                    pin.SE3(
                        np.eye(3),
                        np.array(offset, dtype=float).reshape(
                            3,
                        ),
                    ),
                    pin.FrameType.OP_FRAME,
                )
            )

    def joints_to_q(self, joint_dict):
        q = pin.neutral(self.model)
        for name, val in joint_dict.items():
            jid = self.model.getJointId(name)
            if jid == 0:
                print(f"[warn] joint not in model: {name}", file=sys.stderr)
                continue
            j = self.model.joints[jid]
            if j.nq == 1:
                q[j.idx_q] = float(val)
            else:
                print(f"[warn] joint {name} has nq={j.nq}; skipping", file=sys.stderr)
        return q

    def fk(self, joint_dict, base_frame="world"):
        q = self.joints_to_q(joint_dict)

        data = self.model.createData()

        pin.forwardKinematics(self.model, data, q)
        pin.updateFramePlacements(self.model, data)

        def pose_for(frame_name):
            fid = self.model.getFrameId(frame_name)
            if fid >= len(data.oMf):
                raise RuntimeError(
                    f"Frame {frame_name} id={fid} >= nframes={len(data.oMf)}"
                )
            oMf = data.oMf[fid]  # world->frame
            if base_frame == "base":
                oMb = data.oMi[1]  # root joint
                T = oMb.inverse() * oMf
            else:
                T = oMf
            return {
                "matrix": T.homogeneous.tolist(),
                "position": T.translation.tolist(),
                "quat_xyzw": T.rotation.tolist(),
            }

        return {
            "l_ee": pose_for("l_ee"), 
            "r_ee": pose_for("r_ee"), 
            "l_foot": pose_for("l_foot"),
            "r_foot": pose_for("r_foot")
        }


def _parse_q(args):
    if args.qjson:
        with open(args.qjson, "r") as f:
            return json.load(f)
    if args.q:
        return json.loads(args.q)
    return {}


def main():
    ap = argparse.ArgumentParser(description="Unitree G1 FK for hands")
    ap.add_argument("--urdf", required=True, help="Path to g1_body29_hand14.urdf")
    ap.add_argument(
        "--meshdir", default=None, help="Directory with meshes (defaults to urdf dir)"
    )
    ap.add_argument(
        "--mode",
        choices=["reduced", "whole"],
        default="reduced",
        help="reduced: lock hips+waist+fingers; whole: lock only fingers",
    )
    ap.add_argument("--base_frame", choices=["world", "base"], default="world")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--q", help="JSON string of {joint_name: angle_rad}")
    group.add_argument("--qjson", help="Path to JSON file with {joint_name: angle_rad}")
    args = ap.parse_args()

    qdict = _parse_q(args)

    fk_solver = G1FK(args.urdf, args.meshdir, args.mode)
    out = fk_solver.fk(qdict, base_frame=args.base_frame)
    json.dump(out, sys.stdout, indent=2)


if __name__ == "__main__":
    main()
