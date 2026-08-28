#!/usr/bin/env python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
from pathlib import Path
import os
import sys
from typing import Any

import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.shared import normalize as _normalize
from openpi.training import config as _config

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import build_run_dir_name, candidate_checkpoint_roots
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


_POLICY_DIR = Path(__file__).resolve().parent
_CHECKPOINTS_DIR = _POLICY_DIR / "checkpoints"


# -- MHBench ---------------------------------------------------------------
# MHBench drives two Unitree G1 humanoids. Neither its observation (two robots'
# 43 joint angles, two ego cameras) nor its action (`mhbench_raw_action`, one
# entry per robot) fits XPolicyLab's generic single-bimanual-robot schema, so
# this adapter branches on bench_name exactly as ACT, DP and GR00T_N17 do.
MHBENCH_BENCH_NAME = "mhbench"

# Which env_cfg_type the checkpoints were *trained* under. Eval is launched with
# the scene's name instead (cocarry / handover / door_passage / frame_hang), so
# the run directory has to be rebuilt with the training one -- same substitution
# GR00T_N17/model.py makes.
MHBENCH_CENTRALIZED_ENV_CFG_TYPE = "unitree_g1x2_centralized"
MHBENCH_DECENTRALIZED_ENV_CFG_TYPE = "unitree_g1x2_decentralized"

MHBENCH_ROBOTS = ("robot_a", "robot_b")

# MHBenchTaskEnv.get_obs puts each robot's ego camera on an XPolicyLab-generic
# slot name; the scene camera lands on cam_head and no pi0.5 target reads it.
MHBENCH_VIDEO_SLOT = {"ego_a": "cam_left_wrist", "ego_b": "cam_right_wrist"}
MHBENCH_EGO_VIEW = {"robot_a": "ego_a", "robot_b": "ego_b"}

# The run name of the shared decentralized policy -- one set of weights over
# every task and both roles, so it is named for neither.
MHBENCH_MULTITASK_CKPT = "multitask"

# Per robot the policy commands 35 numbers, and MHBenchTaskEnv.take_action reads
# them back in these three pieces. Same split ACT and DP pack (see
# DP/model.py::_pack_single_robot_action).
MHBENCH_JOINT_TARGET_DIM = 31
MHBENCH_ACTION_DIM = 35


def _mhbench_mode(model_cfg: dict[str, Any]) -> str:
    mode = str(model_cfg.get("mhbench_mode") or "decentralized").strip().lower()
    if mode not in ("centralized", "decentralized"):
        raise ValueError(f"mhbench_mode must be 'centralized' or 'decentralized', got {mode!r}")
    return mode


def _mhbench_train_config_name(task: str, robot: str | None) -> str:
    """The TrainConfig this checkpoint was produced by.

    Derived from the task and the target rather than read from a `deploy.yml`
    key, so a checkpoint can never be served through another target's config --
    which would silently apply the wrong action width, prompt and norm stats.

    The shared multitask policy is one config over the flattened all-task
    dataset (`ckpt_name` is `multitask`, not a task), and it is the same one
    for both agents -- what tells them apart is the instruction each is sent.
    """
    if task == MHBENCH_MULTITASK_CKPT:
        return "pi05_mhbench_multitask_decentralized"
    return f"pi05_mhbench_{task}_{robot or 'centralized'}"


def _mhbench_model_dir(model_cfg: dict[str, Any], task: str, robot: str | None) -> Path:
    """`checkpoints/mhbench-<task>[_<robot>]-<env_cfg_type>-joint-<seed>/<step>`."""
    explicit = model_cfg.get(
        "model_dir" if robot is None or task == MHBENCH_MULTITASK_CKPT else f"model_dir_{robot}"
    )
    if explicit:
        return _resolve_pi05_model_root({**model_cfg, "model_path": explicit})

    run_cfg = dict(model_cfg)
    if robot is None:
        run_cfg["ckpt_name"] = task
        run_cfg["env_cfg_type"] = MHBENCH_CENTRALIZED_ENV_CFG_TYPE
    elif task == MHBENCH_MULTITASK_CKPT:
        # One run directory, named for neither a task nor a robot.
        run_cfg["ckpt_name"] = task
        run_cfg["env_cfg_type"] = MHBENCH_DECENTRALIZED_ENV_CFG_TYPE
    else:
        run_cfg["ckpt_name"] = f"{task}_{robot}"
        run_cfg["env_cfg_type"] = MHBENCH_DECENTRALIZED_ENV_CFG_TYPE
    run_name = build_run_dir_name(run_cfg)
    if run_name is None:
        raise ValueError("bench_name/ckpt_name/env_cfg_type/action_type/seed are required to name the run dir")

    run_dir = _CHECKPOINTS_DIR / run_name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"{robot or 'centralized'} checkpoint not found: {run_dir}")
    return _resolve_pi05_model_root({**model_cfg, "model_path": str(run_dir)})


def _encode_mhbench_obs(observation: dict[str, Any]) -> dict[str, Any]:
    """MHBenchTaskEnv observation -> the keys `MHBenchInputs` reads.

    These are the post-repack names, because `create_trained_policy` applies the
    data transforms but *not* the repack transform -- the adapter stands in for
    it. The full 86-dim joint vector goes over regardless of target: slicing it
    is `mhbench_policy`'s job, so training and serving cut the same columns in
    the same place.
    """
    state = observation["mhbench_state"]
    encoded: dict[str, Any] = {
        "observation/state": np.concatenate(
            [np.asarray(state[robot]["joint_pos"], dtype=np.float32) for robot in MHBENCH_ROBOTS]
        )
    }
    for camera, slot in MHBENCH_VIDEO_SLOT.items():
        encoded[f"observation/{camera}"] = np.asarray(observation["vision"][slot]["color"])
    return encoded


def _encode_mhbench_shared_obs(observation: dict[str, Any], robot: str) -> dict[str, Any]:
    """One agent's view of the observation, as `MHBenchSharedInputs` reads it.

    The shared policy trains on the flattened all-task dataset, where a row is
    one robot: 43 joint angles and one camera. Which robot is decided here;
    what it is being asked to do is the prompt, which the eval client sends per
    agent.
    """
    return {
        "observation/state": np.asarray(
            observation["mhbench_state"][robot]["joint_pos"], dtype=np.float32
        ),
        "observation/ego": np.asarray(
            observation["vision"][MHBENCH_VIDEO_SLOT[MHBENCH_EGO_VIEW[robot]]]["color"]
        ),
    }


def _pack_mhbench_robot_action(flat: np.ndarray) -> dict[str, np.ndarray]:
    if flat.shape[-1] != MHBENCH_ACTION_DIM:
        raise ValueError(f"expected a {MHBENCH_ACTION_DIM}D per-robot action, got {flat.shape}")
    return {
        "joint_targets": flat[:MHBENCH_JOINT_TARGET_DIM],
        "height": flat[MHBENCH_JOINT_TARGET_DIM : MHBENCH_JOINT_TARGET_DIM + 1],
        "base_vel": flat[MHBENCH_JOINT_TARGET_DIM + 1 :],
    }


def _extract_step_number(value: Any) -> int | None:
    matches = [part for part in str(value).split("/") if part]
    if not matches:
        return None
    digits = "".join(ch for ch in matches[-1] if ch.isdigit())
    return int(digits) if digits else None


def _resolve_pi05_model_root(model_cfg: dict[str, Any]) -> Path:
    # Shared precedence: model_path/checkpoint_path keys > ckpt_name-as-path >
    # {bench}-{ckpt}-{env}-{action}-{seed} concat > checkpoints/<ckpt_name>.
    candidates = candidate_checkpoint_roots(
        model_cfg,
        _CHECKPOINTS_DIR,
        policy_dir=_POLICY_DIR,
        explicit_keys=("model_path", "checkpoint_path"),
    )
    if not candidates:
        raise ValueError("ckpt_name or model_path is required for Pi_05.")
    checkpoint_root = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if not checkpoint_root.is_dir():
        return checkpoint_root

    candidate_dirs = []
    if (checkpoint_root / "params").exists() or (checkpoint_root / "assets").exists():
        candidate_dirs.append(checkpoint_root)
    candidate_dirs.extend(
        child
        for child in sorted(checkpoint_root.iterdir())
        if child.is_dir() and ((child / "params").exists() or (child / "assets").exists())
    )
    if not candidate_dirs:
        return checkpoint_root

    checkpoint_num = model_cfg.get("checkpoint_num")
    desired_step = _extract_step_number(checkpoint_num)
    if desired_step is not None:
        normalized = str(desired_step)
        for candidate in candidate_dirs:
            name = candidate.name.lstrip("0") or "0"
            if name == normalized:
                return candidate

        for candidate in candidate_dirs:
            candidate_step = _extract_step_number(candidate.name)
            if candidate_step is None:
                continue
            scaled_step = desired_step
            while len(str(scaled_step)) < len(str(candidate_step)):
                scaled_step *= 10
            if candidate_step in {desired_step, scaled_step}:
                return candidate

    numeric_dirs = [candidate for candidate in candidate_dirs if _extract_step_number(candidate.name) is not None]
    if numeric_dirs:
        return max(numeric_dirs, key=lambda candidate: _extract_step_number(candidate.name) or -1)
    return candidate_dirs[0]


class Model(ModelTemplate):
    def __init__(self, model_cfg: dict[str, Any]):
        self.task_name = model_cfg["task_name"]
        self.action_type = model_cfg.get("action_type", "joint")
        self.observation_window: dict[str, Any] | None = None
        self._latest_env_idx_list: list[int] = [0]

        self._mhbench = str(model_cfg.get("bench_name") or "") == MHBENCH_BENCH_NAME
        if self._mhbench:
            self._init_mhbench(model_cfg)
            return

        self.robot_action_dim_info = (
            get_robot_action_dim_info(model_cfg["env_cfg_type"]) if model_cfg.get("env_cfg_type") is not None else None
        )
        self.policy = self.get_model(model_cfg=model_cfg)
        self.model = self.policy

    # -- MHBench ----------------------------------------------------------

    def _init_mhbench(self, model_cfg: dict[str, Any]) -> None:
        """One process, one or two pi0.5 policies, driving the humanoid pair.

        `centralized` is a single policy answering with all seventy numbers.
        `decentralized` is the two independently trained halves, held side by
        side -- there is no communication between them, they only share a
        process, because XPolicyLab's eval plumbing has exactly one model client.
        Same arrangement as GR00T_N17.

        Decentralized has two forms. The shipped one is a single *shared*
        policy trained on every task and both roles, queried once per agent
        with that agent's own camera, state and instruction -- one set of
        weights, and `ckpt_name` is `multitask`. The other is the older pair,
        two independently trained halves held side by side, where `ckpt_name`
        is the task (cocarry / handover / ...) and the run directories are
        derived per robot.
        """
        self.robot_action_dim_info = None  # MHBench never goes through pack_robot_state
        self._mhbench_mode = _mhbench_mode(model_cfg)
        task = str(model_cfg.get("ckpt_name") or "").strip()
        if not task:
            raise ValueError("mhbench eval needs ckpt_name=<task> (e.g. cocarry)")

        self._mhbench_shared = task == MHBENCH_MULTITASK_CKPT
        targets: tuple[str | None, ...] = (
            (None,) if self._mhbench_mode == "centralized" else MHBENCH_ROBOTS
        )
        # One policy object for both agents when it is one checkpoint: loading
        # it twice would cost a second copy of the weights for nothing.
        load_targets = targets[:1] if self._mhbench_shared else targets
        self._mhbench_policies: dict[str | None, Any] = {}
        for robot in load_targets:
            config_name = _mhbench_train_config_name(task, robot)
            model_dir = _mhbench_model_dir(model_cfg, task, robot)
            config = _config.get_config(config_name)
            # norm_stats deliberately not passed: create_trained_policy loads
            # them from `<model_dir>/assets/<asset_id>` using the config's own
            # asset id, which is what training wrote there. Naming them from a
            # deploy.yml `repo_id` instead is how they end up silently missing.
            self._mhbench_policies[robot] = _policy_config.create_trained_policy(config, str(model_dir))
            print(f"[Pi_05][mhbench] {robot or 'centralized'}: {config_name} <- {model_dir}")
        if self._mhbench_shared:
            for robot in targets:
                self._mhbench_policies[robot] = self._mhbench_policies[load_targets[0]]

        horizon = _config.get_config(_mhbench_train_config_name(task, targets[0])).model.action_horizon
        exec_horizon = model_cfg.get("exec_horizon")
        self._mhbench_exec_horizon = min(int(exec_horizon), horizon) if exec_horizon else horizon

        self.policy = self._mhbench_policies[targets[0]]
        self.model = self.policy

    def _mhbench_update_obs_batch(self, obs_list: list[dict[str, Any]]) -> None:
        self._latest_env_idx_list = [obs.get("env_idx", index) for index, obs in enumerate(obs_list)]
        self.observation_window = [_encode_mhbench_obs(obs) for obs in obs_list]
        # The shared policy re-encodes per agent and needs the instructions the
        # env sent, neither of which survives the pair-shaped encoding above.
        self.observation_window_raw = list(obs_list)

    def _mhbench_get_action_batch(self, env_idx_list: list[int] | None) -> list[list[dict[str, Any]]]:
        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch first!")
        env_idx_list = env_idx_list or self._latest_env_idx_list

        batch: list[list[dict[str, Any]]] = []
        for index in range(len(env_idx_list)):
            observation = self.observation_window[index]
            if self._mhbench_mode == "centralized":
                actions = np.asarray(self._mhbench_policies[None].infer(observation)["actions"])
                per_robot = {
                    robot: actions[:, i * MHBENCH_ACTION_DIM : (i + 1) * MHBENCH_ACTION_DIM]
                    for i, robot in enumerate(MHBENCH_ROBOTS)
                }
            elif self._mhbench_shared:
                # The same policy twice, each time as one agent: its own view,
                # its own 43 joints, its own sentence. The prompt is what makes
                # the two answers differ.
                source = self.observation_window_raw[index]
                per_robot = {}
                for robot in MHBENCH_ROBOTS:
                    encoded = _encode_mhbench_shared_obs(source, robot)
                    prompt = (source.get("mhbench_instruction") or {}).get(robot)
                    if prompt:
                        encoded["prompt"] = str(prompt)
                    per_robot[robot] = np.asarray(
                        self._mhbench_policies[robot].infer(encoded)["actions"]
                    )
            else:
                per_robot = {
                    robot: np.asarray(policy.infer(observation)["actions"])
                    for robot, policy in self._mhbench_policies.items()
                }

            steps = min(self._mhbench_exec_horizon, min(a.shape[0] for a in per_robot.values()))
            batch.append(
                [
                    {
                        "mhbench_raw_action": {
                            robot: _pack_mhbench_robot_action(per_robot[robot][step])
                            for robot in MHBENCH_ROBOTS
                        }
                    }
                    for step in range(steps)
                ]
            )
        return batch

    def get_model(self, model_cfg: dict[str, Any]):
        train_config_name = model_cfg.get("train_config_name", "pi05_aloha")
        repo_id = model_cfg.get("repo_id", "1118")
        model_root = _resolve_pi05_model_root(model_cfg)

        config = _config.get_config(train_config_name)
        norm_stats = None
        if repo_id is not None:
            norm_stats = _normalize.load(model_root / "assets" / str(repo_id))

        return _policy_config.create_trained_policy(config, str(model_root), norm_stats=norm_stats)

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        if self._mhbench:
            self._mhbench_update_obs_batch(obs_list)
            return
        self._latest_env_idx_list = [obs.get("env_idx", index) for index, obs in enumerate(obs_list)]
        encoded_obs_list = [
            encode_obs(obs, self.action_type, self.robot_action_dim_info) for obs in obs_list
        ]
        self.observation_window = stack_obs(encoded_obs_list)

    def get_action(self, **kwargs):
        action_list = self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]], **kwargs)
        return action_list[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if self._mhbench:
            return self._mhbench_get_action_batch(env_idx_list)

        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch first!")

        env_idx_list = env_idx_list or self._latest_env_idx_list
        # actions = self.policy.infer(self.observation_window, **kwargs)["actions"]
        action_list = []

        for batch_index, _ in enumerate(env_idx_list):
            single_observation = slice_stacked_obs(self.observation_window, batch_index)
            actions = self.policy.infer(single_observation, **kwargs)["actions"]
            if self.robot_action_dim_info is None:
                action_list.append(actions)
            else:
                action_list.append(
                    unpack_robot_state(
                        actions,
                        self.action_type,
                        self.robot_action_dim_info,
                        source_type="obs",
                    )
                )

        return action_list

    def reset(self):
        self.observation_window = None
        self._latest_env_idx_list = [0]

    def reset_obsrvationwindows(self):
        self.reset()


def encode_obs(observation, action_type, robot_action_dim_info):
    if "images" in observation and "state" in observation:
        state = np.asarray(observation["state"], dtype=np.float32)
        images = {
            "cam_high": ensure_chw_uint8(observation["images"]["cam_high"]),
            "cam_left_wrist": ensure_chw_uint8(observation["images"]["cam_left_wrist"]),
            "cam_right_wrist": ensure_chw_uint8(observation["images"]["cam_right_wrist"]),
        }
        prompt = observation.get("instruction")
        return {"state": state, "images": images, "prompt": prompt}

    if robot_action_dim_info is None:
        raise ValueError("env_cfg_type is required when encoding raw environment observations.")

    images = {
        "cam_high": ensure_chw_uint8(extract_image(observation, ["cam_high", "cam_head", "head_camera", "top_camera"])),
        "cam_left_wrist": ensure_chw_uint8(
            extract_image(observation, ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"])
        ),
        "cam_right_wrist": ensure_chw_uint8(
            extract_image(observation, ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"])
        ),
    }
    state = pack_robot_state(observation, action_type, robot_action_dim_info, source_type="obs").astype(np.float32)
    prompt = observation.get("instruction")
    return {"state": state, "images": images, "prompt": prompt}


def stack_obs(obs_list: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "state": np.stack([obs["state"] for obs in obs_list], axis=0),
        "images": {
            "cam_high": np.stack([obs["images"]["cam_high"] for obs in obs_list], axis=0),
            "cam_left_wrist": np.stack([obs["images"]["cam_left_wrist"] for obs in obs_list], axis=0),
            "cam_right_wrist": np.stack([obs["images"]["cam_right_wrist"] for obs in obs_list], axis=0),
        },
        "prompt": [obs["prompt"] for obs in obs_list],
    }


def slice_stacked_obs(obs: dict[str, Any], batch_index: int) -> dict[str, Any]:
    return {
        "state": obs["state"][batch_index],
        "images": {
            "cam_high": obs["images"]["cam_high"][batch_index],
            "cam_left_wrist": obs["images"]["cam_left_wrist"][batch_index],
            "cam_right_wrist": obs["images"]["cam_right_wrist"][batch_index],
        },
        "prompt": obs["prompt"][batch_index],
    }


def extract_image(observation, candidate_names):
    vision = observation.get("vision", {})
    for candidate_name in candidate_names:
        if candidate_name not in vision:
            continue
        image = vision[candidate_name]
        if isinstance(image, dict):
            for image_key in ("color", "rgb"):
                if image_key in image:
                    return image[image_key]
        else:
            return image
    raise KeyError(f"Could not find any image for candidates: {candidate_names}")


def ensure_chw_uint8(image):
    image = np.asarray(image)

    if image.ndim != 3:
        raise ValueError(f"Expected image ndim=3, got shape {image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if image.shape[-1] in (1, 3):
        image_hwc = image
    elif image.shape[0] in (1, 3):
        image_hwc = np.transpose(image, (1, 2, 0))
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    return np.transpose(image_hwc, (2, 0, 1))
