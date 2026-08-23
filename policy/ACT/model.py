import cv2
import numpy as np
import torch
from .detr.act_policy import ACT
from argparse import Namespace

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import pack_robot_state, unpack_robot_state, get_robot_action_dim_info
from XPolicyLab.utils.checkpoint_resolver import build_run_dir_name
import os

# MHBenchTaskEnv.get_obs (scripts/mhbench_xpolicylab_env.py) maps each robot's
# own ego camera onto these XPolicyLab-generic slot names: the fixed
# third-person camera is cam_head, robot_a's ego is cam_left_wrist, robot_b's
# is cam_right_wrist. Same mapping GR00T_N17/model.py's MHBENCH_CAMERA_SLOT
# uses, since it comes from the env, not the policy.
MHBENCH_CAMERA_SLOT = {"robot_a": "cam_left_wrist", "robot_b": "cam_right_wrist"}

class Model(ModelTemplate):

    def __init__(self, model_cfg):
        self._mhbench_decentralized = (
            str(model_cfg.get("bench_name") or "") == "mhbench"
            and model_cfg.get("env_cfg_type") == "unitree_g1x2_decentralized"
        )
        if self._mhbench_decentralized:
            self._init_mhbench_decentralized(model_cfg)
            return

        self.camera_names = model_cfg.get('camera_names', [])
        model_cfg['camera_names'] = self.camera_names

        self.model = self.get_model(model_cfg=model_cfg)
        self.action_type = model_cfg['action_type']
        try:
            self.robot_action_dim_info = get_robot_action_dim_info(model_cfg['env_cfg_type'])
        except FileNotFoundError:
            # MHBench env_cfg_types (unitree_g1x2_centralized, ...) have no
            # env_cfg/<type>.yml -- their scene lives in MHBench's own Isaac
            # Lab env_cfg tree, not XPolicyLab's. Fine here: the mhbench
            # dual-robot path (get_action/encode_obs below) never touches
            # this, since it packs/unpacks mhbench_state/mhbench_raw_action
            # directly instead of going through pack_robot_state/unpack_robot_state.
            self.robot_action_dim_info = None

    def get_model(self, model_cfg):
        if not model_cfg.get('ckpt_dir'):
            if not model_cfg.get('ckpt_name'):
                raise ValueError("ACT requires ckpt_name or ckpt_dir during evaluation.")
            # ckpt_name is the full run directory name under checkpoints/.
            model_cfg['ckpt_dir'] = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'checkpoints', str(model_cfg['ckpt_name']))
        return ACT(model_cfg, Namespace(**model_cfg))

    def _init_mhbench_decentralized(self, model_cfg):
        """Two single-robot ACT checkpoints served from one process.

        Trained separately as mhbench-<task>_robot_a/-unitree_g1x2_decentralized
        and ..._robot_b (baselines/README.md's ACT Train section), so unlike
        the centralized case one checkpoint cannot answer for both robots.
        Mirrors GR00T_N17/model.py's `_init_mhbench`/`_resolve_mhbench_model_dir`/
        `_get_action_mhbench`: one server, one policy instance per robot,
        combined into the same `mhbench_raw_action` shape
        `MHBenchTaskEnv.take_action` already expects for the centralized case.
        `ckpt_name` here is the task (e.g. "cocarry"), not a run dir -- same
        convention GR00T's mhbench mode uses.
        """
        self.action_type = model_cfg['action_type']
        task = str(model_cfg.get('ckpt_name') or '').strip()
        if not task:
            raise ValueError("mhbench decentralized eval needs ckpt_name=<task> (e.g. cocarry)")

        checkpoints_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
        self._sub_models = {}
        for robot, camera_name in MHBENCH_CAMERA_SLOT.items():
            # Explicit deploy.yml override first, matching GR00T's model_dir_<robot>.
            explicit = model_cfg.get(f'model_dir_{robot}')
            if explicit:
                ckpt_dir = explicit
            else:
                run_cfg = dict(model_cfg)
                run_cfg['ckpt_name'] = f"{task}_{robot}"
                run_name = build_run_dir_name(run_cfg)
                if run_name is None:
                    raise ValueError("bench_name/ckpt_name/env_cfg_type/action_type/seed required to name the run dir")
                ckpt_dir = os.path.join(checkpoints_dir, run_name)
            if not os.path.isdir(ckpt_dir):
                raise FileNotFoundError(f"{robot} checkpoint not found: {ckpt_dir}")

            sub_cfg = dict(model_cfg)
            sub_cfg['ckpt_dir'] = ckpt_dir
            sub_cfg['camera_names'] = [camera_name]
            self._sub_models[robot] = ACT(sub_cfg, Namespace(**sub_cfg))
            print(f"[ACT][mhbench] {robot}: {ckpt_dir} (camera={camera_name})")

    def update_obs(self, obs):
        if self._mhbench_decentralized:
            for robot, sub_model in self._sub_models.items():
                sub_model.update_obs(self._encode_mhbench_robot_obs(obs, robot))
            return

        self._mhbench_dual_robot = "mhbench_state" in obs
        encoded_obs = self.encode_obs(obs, self.action_type, self.robot_action_dim_info)
        self.model.update_obs(encoded_obs)

    def _encode_mhbench_robot_obs(self, observation, robot):
        camera_name = MHBENCH_CAMERA_SLOT[robot]
        color = cv2.resize(observation["vision"][camera_name]["color"], (640, 480), interpolation=cv2.INTER_LINEAR)
        color = np.moveaxis(color, -1, 0) / 255.0
        joint_pos = np.asarray(observation["mhbench_state"][robot]["joint_pos"], dtype=np.float32)
        return {camera_name: color, "qpos": joint_pos}

    # def update_obs_batch(self, obs_list): # TODO
    #     pass

    @staticmethod
    def _act_chunk(act) -> np.ndarray:
        """``(n, action_dim)`` -- the whole action chunk where that is exact.

        ACT queries its network every ``query_frequency`` steps and reads one
        column of the result per step, so handing the caller the remaining
        columns at once produces the same actions with one round trip instead
        of ``query_frequency`` of them -- and the rollout loop then stops
        rendering and shipping an observation it would only discard
        (`utils/rollout.py`). Under temporal aggregation the ensemble is
        defined per step and there is no chunk to return: one action, and the
        loop keeps observing every step, which that mode requires.
        """
        if act.temporal_agg:
            return np.atleast_2d(np.asarray(act.get_action()))
        return np.asarray(act.get_action_chunk())

    def get_action(self):
        if self._mhbench_decentralized:
            per_robot = {
                robot: self._act_chunk(sub_model)
                for robot, sub_model in self._sub_models.items()
            }
            # Both robots run the same chunk length; take the shorter one
            # anyway, so a future per-robot horizon cannot silently pair
            # step i of one robot with step j of the other.
            steps = min(chunk.shape[0] for chunk in per_robot.values())
            return [
                {
                    "mhbench_raw_action": {
                        robot: self._pack_single_robot_action(chunk[t])
                        for robot, chunk in per_robot.items()
                    }
                }
                for t in range(steps)
            ]

        if getattr(self, "_mhbench_dual_robot", False):
            return [
                {"mhbench_raw_action": self._pack_dual_arm_action(a)}
                for a in self._act_chunk(self.model)
            ]

        # Non-MHBench benches keep the one-action-per-call contract
        # `unpack_robot_state` was written against.
        actions = self.model.get_action()
        return unpack_robot_state(actions, self.action_type, self.robot_action_dim_info, source_type='obs')

    @staticmethod
    def _pack_single_robot_action(flat_action: np.ndarray) -> dict:
        """One robot's 35D ACT action -> MHBenchTaskEnv.take_action's
        {joint_targets, base_vel, height}.

        mhbench_keys.ACTION_KEYS' training-time concatenation order:
        [left_arm(7) right_arm(7) left_hand(7) right_hand(7) waist(3)
        base_height_command(1) navigate_command(3)]. The first 31 are already
        gr00t_joint_names() order (ACTION_JOINT_GROUPS lists the same 5 groups
        in the same order -- no permutation needed); base_height_command ->
        height, navigate_command -> base_vel.
        """
        assert flat_action.shape[-1] == 35, f"expected 35D per-robot action, got {flat_action.shape}"
        return {
            "joint_targets": flat_action[0:31],
            "height": flat_action[31:32],
            "base_vel": flat_action[32:35],
        }

    def _pack_dual_arm_action(self, flat_action: np.ndarray) -> dict:
        """This adapter's 70D flat action -> one dict per robot, via
        :meth:`_pack_single_robot_action` on each robot's 35D slice."""
        assert flat_action.shape[-1] == 70, f"expected 70D dual-robot action, got {flat_action.shape}"
        return {
            robot: self._pack_single_robot_action(flat_action[i * 35 : (i + 1) * 35])
            for i, robot in enumerate(("robot_a", "robot_b"))
        }

    # def get_action_batch(self, env_idx_list): # TODO
    #     pass

    def reset(self):
        if self._mhbench_decentralized:
            for sub_model in self._sub_models.values():
                self._reset_act_instance(sub_model)
            return
        self._reset_act_instance(self.model)

    @staticmethod
    def _reset_act_instance(act):
        # Reset temporal aggregation state if enabled
        if act.temporal_agg:
            act.all_time_actions = torch.zeros([
                act.max_timesteps,
                act.max_timesteps + act.num_queries,
                act.state_dim,
            ]).to(act.device)
        act.t = 0

    def encode_obs(self, observation, action_type, robot_action_dim_info):
        res_dict = dict()

        for camera_name in self.camera_names:
            if camera_name not in observation["vision"]:
                raise ValueError(f"Expected camera '{camera_name}' not found in observation['vision']")
            color = cv2.resize(observation["vision"][camera_name]["color"], (640, 480), interpolation=cv2.INTER_LINEAR)
            color = np.moveaxis(color, -1, 0) / 255.0
            res_dict[camera_name] = color
        
        mhbench_state = observation.get("mhbench_state")
        if mhbench_state is not None:
            # MHBench's two-full-humanoid state doesn't fit XPolicyLab's
            # generic single-bimanual-robot obs['state'] schema (pack_robot_state
            # only knows arm_dim+ee_dim=70, but this robot's qpos is 86: all 7
            # URDF joint groups per robot -- mhbench_state's per-robot joint_pos
            # is already in that exact order, see mhbench_state_joint_names()).
            res_dict["qpos"] = np.concatenate([
                np.asarray(mhbench_state["robot_a"]["joint_pos"], dtype=np.float32),
                np.asarray(mhbench_state["robot_b"]["joint_pos"], dtype=np.float32),
            ])
        else:
            res_dict["qpos"] = pack_robot_state(observation, action_type, robot_action_dim_info, source_type="obs")

        return res_dict