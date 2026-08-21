import torch
import yaml
import cv2
import numpy as np
import hydra
import dill
import sys, os

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(current_file_path)
sys.path.append(parent_dir)

from diffusion_policy.workspace.robotworkspace import RobotWorkspace
from diffusion_policy.env_runner.dp_runner import DPRunner
from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import pack_robot_state, unpack_robot_state, get_robot_action_dim_info
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root, build_run_dir_name

# MHBenchTaskEnv.get_obs (scripts/mhbench_xpolicylab_env.py) maps each robot's
# own ego camera onto these XPolicyLab-generic slot names -- same mapping
# ACT/model.py and GR00T_N17/model.py's MHBENCH_CAMERA_SLOT use, since it
# comes from the env, not the policy.
MHBENCH_CAMERA_SLOT = {"robot_a": "cam_left_wrist", "robot_b": "cam_right_wrist"}


def _prep_camera(color: np.ndarray) -> np.ndarray:
    """XPolicyLab vision obs (HWC uint8) -> DP's CHW float32 input, 320x240."""
    img = np.moveaxis(color, -1, 0) / 255.0
    return np.transpose(
        cv2.resize(np.transpose(img, (1, 2, 0)), (320, 240), interpolation=cv2.INTER_AREA), (2, 0, 1)
    )


def _pack_single_robot_action(flat_action: np.ndarray) -> dict:
    """One robot's 35D DP action -> MHBenchTaskEnv.take_action's
    {joint_targets, base_vel, height}. Same mhbench_keys.ACTION_KEYS layout
    ACT's model.py packs (scripts/data_convertion.py's `keys_for` is the same
    function behind both --format act and --format diffusion_policy, so the
    two adapters see the same 35D order)."""
    assert flat_action.shape[-1] == 35, f"expected 35D per-robot action, got {flat_action.shape}"
    return {
        "joint_targets": flat_action[0:31],
        "height": flat_action[31:32],
        "base_vel": flat_action[32:35],
    }


def _pack_dual_arm_action(flat_action: np.ndarray) -> dict:
    """This adapter's 70D flat action -> one dict per robot."""
    assert flat_action.shape[-1] == 70, f"expected 70D dual-robot action, got {flat_action.shape}"
    return {
        robot: _pack_single_robot_action(flat_action[i * 35 : (i + 1) * 35])
        for i, robot in enumerate(("robot_a", "robot_b"))
    }


class Model(ModelTemplate):

    def __init__(self, model_cfg):
        load_config_path = os.path.join(parent_dir, 'diffusion_policy/config/robot_dp.yaml')
        with open(load_config_path, "r", encoding="utf-8") as f:
            model_training_config = yaml.safe_load(f)
        self.n_obs_steps = model_training_config['n_obs_steps']
        self.n_action_steps = model_training_config['n_action_steps']
        self.action_type = model_cfg['action_type']

        self._mhbench_decentralized = (
            str(model_cfg.get("bench_name") or "") == "mhbench"
            and model_cfg.get("env_cfg_type") == "unitree_g1x2_decentralized"
        )
        if self._mhbench_decentralized:
            self._init_mhbench_decentralized(model_cfg)
            return

        self._mhbench_dual_robot = False  # set per-batch in update_obs_batch
        self.runner = DPRunner(n_obs_steps=self.n_obs_steps, n_action_steps=self.n_action_steps)
        self.model = self.get_model(model_cfg=model_cfg)
        try:
            self.robot_action_dim_info = get_robot_action_dim_info(model_cfg['env_cfg_type'])
        except FileNotFoundError:
            # MHBench env_cfg_types (unitree_g1x2_centralized, mhbench_cocarry,
            # ...) have no env_cfg/<type>.yml -- their scene lives in MHBench's
            # own Isaac Lab env_cfg tree, not XPolicyLab's. Fine here: the
            # mhbench dual-robot path never touches this, since it packs
            # agent_pos from mhbench_state and the action from
            # mhbench_raw_action directly instead of going through
            # pack_robot_state/unpack_robot_state.
            self.robot_action_dim_info = None
        self._latest_env_idx_list = None

    def _init_mhbench_decentralized(self, model_cfg):
        """Two single-robot DP checkpoints served from one process.

        Trained separately as mhbench-<task>_robot_a/-unitree_g1x2_decentralized
        and ..._robot_b (baselines/README.md's DP Train section), each with
        one camera stream (train.sh's num_cameras=1 for this env_cfg_type).
        Mirrors ACT's/GR00T_N17's mhbench decentralized mode: one server, one
        policy instance per robot, combined into the same `mhbench_raw_action`
        shape `MHBenchTaskEnv.take_action` expects. `ckpt_name` here is the
        task (e.g. "cocarry"), not a run dir.
        """
        task = str(model_cfg.get('ckpt_name') or '').strip()
        if not task:
            raise ValueError("mhbench decentralized eval needs ckpt_name=<task> (e.g. cocarry)")

        checkpoints_dir = os.path.join(parent_dir, "checkpoints")
        self._sub_policies = {}
        self._sub_runners = {}
        for robot, camera_name in MHBENCH_CAMERA_SLOT.items():
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

            self._sub_policies[robot] = self._load_policy(ckpt_dir, model_cfg.get('checkpoint_num', 'latest'))
            self._sub_runners[robot] = DPRunner(n_obs_steps=self.n_obs_steps, n_action_steps=self.n_action_steps)
            print(f"[DP][mhbench] {robot}: {ckpt_dir} (camera={camera_name})")

    def _load_policy(self, ckpt_dir, checkpoint_num):
        ckpt_file = self._resolve_checkpoint_file(ckpt_dir, checkpoint_num)

        # load checkpoint and workspace
        payload = torch.load(open(ckpt_file, "rb"), pickle_module=dill)
        cfg = payload["cfg"]
        cls = hydra.utils.get_class(cfg._target_)
        workspace = cls(cfg, output_dir=None)
        workspace: RobotWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # get policy from workspace
        policy = workspace.model
        if cfg.training.use_ema:
            policy = workspace.ema_model

        device = torch.device("cuda:0")
        policy.to(device)
        policy.eval()

        return policy

    def get_model(self, model_cfg):
        ckpt_dir = resolve_checkpoint_root(
            model_cfg,
            os.path.join(parent_dir, "checkpoints"),
            policy_dir=parent_dir,
            must_exist=False,
        )
        return self._load_policy(ckpt_dir, model_cfg.get('checkpoint_num', 'latest'))

    def _resolve_checkpoint_file(self, ckpt_dir, checkpoint_num):
        ckpt_dir = os.fspath(ckpt_dir)
        checkpoint_num = "latest" if checkpoint_num is None else str(checkpoint_num)

        if checkpoint_num.lower() not in {"", "latest", "none"}:
            ckpt_file = os.path.join(ckpt_dir, f"{checkpoint_num}.ckpt")
            if not os.path.isfile(ckpt_file):
                raise FileNotFoundError(f"DP checkpoint not found: {ckpt_file}")
            return ckpt_file

        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"DP checkpoint directory not found: {ckpt_dir}")

        candidates = []
        for name in os.listdir(ckpt_dir):
            if not name.endswith(".ckpt"):
                continue
            stem = name[:-5]
            if stem.isdigit():
                candidates.append((int(stem), os.path.join(ckpt_dir, name)))

        if not candidates:
            raise FileNotFoundError(f"No numeric DP checkpoints found under: {ckpt_dir}")

        return max(candidates, key=lambda item: item[0])[1]

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        env_idx_list = [obs["env_idx"] for obs in obs_list]

        if self._mhbench_decentralized:
            for robot, runner in self._sub_runners.items():
                encoded = [self._encode_mhbench_robot_obs(obs, robot) for obs in obs_list]
                runner.update_obs(encoded, env_idx_list)
            self._latest_env_idx_list = env_idx_list
            return

        self._mhbench_dual_robot = bool(obs_list) and "mhbench_state" in obs_list[0]
        encoded_list = [
            encode_obs(obs, self.action_type, self.robot_action_dim_info, self._mhbench_dual_robot)
            for obs in obs_list
        ]
        self.runner.update_obs(encoded_list, env_idx_list)
        self._latest_env_idx_list = env_idx_list

    def _encode_mhbench_robot_obs(self, observation, robot):
        """One robot's single-camera decentralized obs: head_cam is that
        robot's own ego view -- no left_cam/right_cam, matching what its
        checkpoint was trained with (train.sh's num_cameras=1)."""
        camera_name = MHBENCH_CAMERA_SLOT[robot]
        head_img = _prep_camera(observation["vision"][camera_name]["color"])
        agent_pos = np.asarray(observation["mhbench_state"][robot]["joint_pos"], dtype=np.float32)
        return dict(head_cam=head_img, agent_pos=agent_pos)

    def get_action(self):
        if not self._latest_env_idx_list:
            raise RuntimeError("get_action() called before update_obs().")

        action_list = self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]])

        return action_list[0]

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = self._latest_env_idx_list
        if not env_idx_list:
            raise RuntimeError("get_action_batch() called before update_obs_batch().")

        if self._mhbench_decentralized:
            per_robot = {
                robot: runner.get_action(self._sub_policies[robot], env_idx_list)
                for robot, runner in self._sub_runners.items()
            }  # each: (len(env_idx_list), n_action_steps, 35)
            steps = min(arr.shape[1] for arr in per_robot.values())
            return [
                [
                    {
                        "mhbench_raw_action": {
                            robot: _pack_single_robot_action(per_robot[robot][env_i, t])
                            for robot in per_robot
                        }
                    }
                    for t in range(steps)
                ]
                for env_i in range(len(env_idx_list))
            ]

        actions = self.runner.get_action(self.model, env_idx_list)  # (len(env_idx_list), n_action_steps, action_dim)

        if self._mhbench_dual_robot:
            return [
                [{"mhbench_raw_action": _pack_dual_arm_action(actions[i, t])} for t in range(actions.shape[1])]
                for i in range(len(env_idx_list))
            ]

        return [
            unpack_robot_state(actions[i], self.action_type, self.robot_action_dim_info, source_type='obs')
            for i in range(len(env_idx_list))
        ]

    def reset(self):
        if self._mhbench_decentralized:
            for runner in self._sub_runners.values():
                runner.reset_obs()
            self._latest_env_idx_list = None
            return
        self.runner.reset_obs()
        self._latest_env_idx_list = None

def encode_obs(observation, action_type, robot_action_dim_info, mhbench_dual_robot=False):
    left_cam = _prep_camera(observation["vision"]["cam_left_wrist"]["color"])
    right_cam = _prep_camera(observation["vision"]["cam_right_wrist"]["color"])

    if mhbench_dual_robot:
        # MHBench's two-full-humanoid state doesn't fit XPolicyLab's generic
        # single-bimanual-robot obs['state'] schema (pack_robot_state only
        # knows arm_dim+ee_dim=70, but this robot's qpos is 86) -- same reason
        # ACT/model.py's encode_obs branches here. Only the two ego cameras
        # are used (num_cameras=2, no head_cam) -- cam_head is the scene
        # camera, never in the training zarr or shape_meta for this robot;
        # see RobotImageDataset's camera_map-derived cam_obs_names.
        mhbench_state = observation["mhbench_state"]
        agent_pos = np.concatenate([
            np.asarray(mhbench_state["robot_a"]["joint_pos"], dtype=np.float32),
            np.asarray(mhbench_state["robot_b"]["joint_pos"], dtype=np.float32),
        ])
        return dict(left_cam=left_cam, right_cam=right_cam, agent_pos=agent_pos)

    head_img = _prep_camera(observation["vision"]["cam_head"]["color"])
    agent_pos = pack_robot_state(observation, action_type, robot_action_dim_info, source_type='obs')
    return dict(head_cam=head_img, left_cam=left_cam, right_cam=right_cam, agent_pos=agent_pos)
