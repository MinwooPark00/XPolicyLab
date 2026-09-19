import pickle

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

# The order a two-camera MHBench policy stacks its views in: ego_a then ego_b,
# which is the order training reads them in (register_task_config.py sorts the
# hdf5's camera keys). It is not cosmetic -- DETRVAE runs one shared backbone
# and concatenates the views along width, so a swapped pair is a different
# input. The centralized policy and the joint-obs per-robot policies both use it.
MHBENCH_JOINT_CAMERAS = [MHBENCH_CAMERA_SLOT["robot_a"], MHBENCH_CAMERA_SLOT["robot_b"]]

# One robot's widths (configs/gr00t/mhbench_keys.py: JOINTS_PER_ROBOT,
# ACTION_DIMS_PER_ROBOT). A per-robot checkpoint whose stats are twice as wide
# on one side is one of train.sh's joint settings (ACT_VARIANT).
MHBENCH_STATE_DIM = 43
MHBENCH_ACTION_DIM = 35
MHBENCH_ROBOTS = ("robot_a", "robot_b")


def _checkpoint_dims(ckpt_dir):
    """(state_dim, action_dim) a checkpoint was trained at, from its own stats.

    imitate_episodes.py saves the normalization stats beside the weights, and
    their widths are the network's: the checkpoint says what it is, so serving
    needs no variant knob that could disagree with what was trained.
    """
    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
    if not os.path.isfile(stats_path):
        raise FileNotFoundError(f"ACT dataset stats not found: {stats_path}")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    return int(np.size(stats["qpos_mean"])), int(np.size(stats["action_mean"]))


def _set_act_dims(cfg, state_dim, action_dim):
    """detr's build() reads the widths from the environment at construction
    (train.sh exports them); `action_dim` in the config is what ACT's temporal
    aggregation buffer is sized by."""
    os.environ["ACT_STATE_DIM"] = str(state_dim)
    os.environ["ACT_ACTION_DIM"] = str(action_dim)
    cfg["action_dim"] = action_dim

# cv2.resize (width, height) of the training frames: MHBench's converter keeps
# 320x240; detr/process_data.py upscales other benches to 640x480.
MHBENCH_IMAGE_SIZE = (320, 240)
UPSTREAM_IMAGE_SIZE = (640, 480)

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
        if str(model_cfg.get("bench_name") or "") == "mhbench":
            # Both ego views, in training's order whatever deploy.yml lists.
            self.camera_names = list(MHBENCH_JOINT_CAMERAS)
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
            # ckpt_name is the full run directory name under checkpoints/ --
            # except from MHBench's runner, which passes the task word and
            # expects the shared naming rule (it only names the directory
            # itself, as `ckpt_dir`, when it stages checkpoints).
            run_name = str(model_cfg['ckpt_name'])
            if str(model_cfg.get("bench_name") or "") == "mhbench" and not run_name.startswith("mhbench-"):
                run_name = self._mhbench_run_name(model_cfg)
            model_cfg['ckpt_dir'] = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 'checkpoints', run_name)
        if str(model_cfg.get("bench_name") or "") == "mhbench":
            state_dim, action_dim = _checkpoint_dims(model_cfg['ckpt_dir'])
            if (state_dim, action_dim) != (2 * MHBENCH_STATE_DIM, 2 * MHBENCH_ACTION_DIM):
                raise ValueError(
                    f"{model_cfg['ckpt_dir']} is {state_dim}D state / {action_dim}D action, not a centralized "
                    f"(ctce) checkpoint -- a per-robot setting is served with MHBENCH_MODE=decentralized")
            _set_act_dims(model_cfg, state_dim, action_dim)
        return ACT(model_cfg, Namespace(**model_cfg))

    @staticmethod
    def _mhbench_run_name(cfg):
        """`mhbench-<ckpt>-<env_cfg_type>-<action>-<seed>[-<tag>]`: train.sh's
        ckpt_dir and common.sh's mh_run_name. build_run_dir_name stops at the
        seed, and the tag is what separates the joint settings' runs from the
        default one's, so without it a tagged eval would serve the untagged run."""
        run_name = build_run_dir_name(cfg)
        if run_name is None:
            raise ValueError("bench_name/ckpt_name/env_cfg_type/action_type/seed required to name the run dir")
        tag = str(cfg.get("ckpt_tag") or "").strip()
        return f"{run_name}-{tag}" if tag else run_name

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

        Each checkpoint is one of three per-robot settings (train.sh's
        ACT_VARIANT), read off its own stats rather than off a knob:

            dtde      43D state, 35D action   own obs -> own action
            jointobs  86D state, 35D action   both robots' obs -> own action
            jointact  43D state, 70D action   own obs -> both actions; this
                                              robot executes its own half only

        `ckpt_tag` (the runner's CKPT_TAG, `jointobs`/`jointact`) is what picks
        the run directory, so the two robots always come from the same setting.
        """
        self.action_type = model_cfg['action_type']
        task = str(model_cfg.get('ckpt_name') or '').strip()
        if not task:
            raise ValueError("mhbench decentralized eval needs ckpt_name=<task> (e.g. cocarry)")

        checkpoints_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
        self._sub_models = {}
        self._joint_obs = {}
        self._joint_action = {}
        for robot, camera_name in MHBENCH_CAMERA_SLOT.items():
            # Explicit deploy.yml override first, matching GR00T's model_dir_<robot>.
            explicit = model_cfg.get(f'model_dir_{robot}')
            if explicit:
                ckpt_dir = explicit
            else:
                run_cfg = dict(model_cfg)
                run_cfg['ckpt_name'] = f"{task}_{robot}"
                run_name = self._mhbench_run_name(run_cfg)
                ckpt_dir = os.path.join(checkpoints_dir, run_name)
            if not os.path.isdir(ckpt_dir):
                raise FileNotFoundError(f"{robot} checkpoint not found: {ckpt_dir}")

            state_dim, action_dim = _checkpoint_dims(ckpt_dir)
            if state_dim not in (MHBENCH_STATE_DIM, 2 * MHBENCH_STATE_DIM) \
                    or action_dim not in (MHBENCH_ACTION_DIM, 2 * MHBENCH_ACTION_DIM) \
                    or (state_dim, action_dim) == (2 * MHBENCH_STATE_DIM, 2 * MHBENCH_ACTION_DIM):
                raise ValueError(
                    f"{robot}: {ckpt_dir} is {state_dim}D state / {action_dim}D action, which is no per-robot "
                    f"setting (dtde 43/35, jointobs 86/35, jointact 43/70); 86/70 is centralized")
            self._joint_obs[robot] = state_dim == 2 * MHBENCH_STATE_DIM
            self._joint_action[robot] = action_dim == 2 * MHBENCH_ACTION_DIM

            sub_cfg = dict(model_cfg)
            sub_cfg['ckpt_dir'] = ckpt_dir
            sub_cfg['camera_names'] = list(MHBENCH_JOINT_CAMERAS) if self._joint_obs[robot] else [camera_name]
            _set_act_dims(sub_cfg, state_dim, action_dim)
            self._sub_models[robot] = ACT(sub_cfg, Namespace(**sub_cfg))
            variant = "jointobs" if self._joint_obs[robot] else "jointact" if self._joint_action[robot] else "dtde"
            print(f"[ACT][mhbench] {robot}: {ckpt_dir} (variant={variant}, state={state_dim}D, "
                  f"action={action_dim}D, cameras={sub_cfg['camera_names']})")
        if len({(self._joint_obs[r], self._joint_action[r]) for r in self._sub_models}) != 1:
            raise ValueError("robot_a and robot_b checkpoints are different settings; one CKPT_TAG names one setting")

    def update_obs(self, obs):
        if self._mhbench_decentralized:
            for robot, sub_model in self._sub_models.items():
                sub_model.update_obs(self._encode_mhbench_robot_obs(obs, robot))
            return

        self._mhbench_dual_robot = "mhbench_state" in obs
        encoded_obs = self.encode_obs(obs, self.action_type, self.robot_action_dim_info)
        self.model.update_obs(encoded_obs)

    def _encode_mhbench_robot_obs(self, observation, robot):
        """One per-robot policy's input: its own view and joints, or under
        joint-obs both robots' -- always robot_a then robot_b, the order the
        training rows hold them in (mhbench_keys.state_keys(None)), whichever
        robot the policy drives."""
        joint = self._joint_obs[robot]
        encoded = {}
        for camera_name in (MHBENCH_JOINT_CAMERAS if joint else [MHBENCH_CAMERA_SLOT[robot]]):
            if camera_name not in observation["vision"]:
                raise ValueError(f"{robot}: camera '{camera_name}' not in observation['vision'] "
                                 f"(a joint-obs policy needs both ego views in EVAL_OBS_CAMERAS)")
            color = cv2.resize(observation["vision"][camera_name]["color"], MHBENCH_IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
            encoded[camera_name] = np.moveaxis(color, -1, 0) / 255.0
        encoded["qpos"] = np.concatenate([
            np.asarray(observation["mhbench_state"][r]["joint_pos"], dtype=np.float32)
            for r in (MHBENCH_ROBOTS if joint else (robot,))
        ])
        return encoded

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
                robot: self._own_action(robot, self._act_chunk(sub_model))
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

    def _own_action(self, robot, chunk: np.ndarray) -> np.ndarray:
        """``(n, 35)`` -- the part of a per-robot policy's chunk it executes.

        A joint-action policy predicts both robots' 70D (robot_a's 35 then
        robot_b's, mhbench_keys.action_keys(None)) and drives only itself: its
        prediction of the partner is a training target, never a command -- the
        partner's own policy issues that.
        """
        if not self._joint_action[robot]:
            return chunk
        i = MHBENCH_ROBOTS.index(robot)
        return chunk[:, i * MHBENCH_ACTION_DIM:(i + 1) * MHBENCH_ACTION_DIM]

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
        mhbench_state = observation.get("mhbench_state")
        image_size = MHBENCH_IMAGE_SIZE if mhbench_state is not None else UPSTREAM_IMAGE_SIZE

        for camera_name in self.camera_names:
            if camera_name not in observation["vision"]:
                raise ValueError(f"Expected camera '{camera_name}' not found in observation['vision']")
            color = cv2.resize(observation["vision"][camera_name]["color"], image_size, interpolation=cv2.INTER_LINEAR)
            color = np.moveaxis(color, -1, 0) / 255.0
            res_dict[camera_name] = color

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