import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import build_run_dir_name
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)


POLICY_DIR = Path(__file__).resolve().parent
FASTWAM_ROOT = POLICY_DIR / "FastWAM"
FASTWAM_SRC = FASTWAM_ROOT / "src"

# MHBench camera slots, as every mhbench adapter maps them (DP/ACT/GR00T_N17):
# the env sends ego_a as cam_left_wrist and ego_b as cam_right_wrist
# (mhbench_xpolicylab_env.py's _VISION_SLOT). The names are historical --
# cam_left_wrist is robot A's *head* camera.
MHBENCH_CAMERA_SLOT = {"robot_a": "cam_left_wrist", "robot_b": "cam_right_wrist"}

# The serving task yaml per checkpoint profile -- the same yamls train.sh
# trains under, so serving cannot compose a different processor than training.
# Where _encode_mhbench parks the env's per-agent sentences inside the encoded
# observation. Not a policy target, so the inference loop skips it.
INSTRUCTIONS_KEY = "__mhbench_instructions__"

MHBENCH_SIM_TASK = {
    "unitree_g1x2_centralized": "mhbench_uncond_2cam_384_1e-4",
    "unitree_g1x2_decentralized": "mhbench_uncond_1cam_192_1e-4",
}


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, str) and value.strip().lower() in {"", "none", "null"}


def _standardize_rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected HWC image with 3 channels, got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.shape[:2] != (240, 320):
        image = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
    if image.shape != (240, 320, 3):
        raise ValueError(f"Expected standardized RGB shape (240, 320, 3), got {image.shape}")
    return image


def _get_instruction(obs: dict, fallback: str) -> str:
    value = obs.get("task_instruction")
    if value is None:
        value = obs.get("instruction", obs.get("instructions"))
    if isinstance(value, (list, tuple)):
        return str(value[0]) if value else fallback
    if value is None:
        return fallback
    if hasattr(value, "item"):
        value = value.item()
    text = str(value).strip()
    return text if text else fallback


def _pack_single_robot_action(flat_action: np.ndarray) -> dict:
    """One robot's 35D action -> MHBenchTaskEnv.take_action's
    {joint_targets, height, base_vel}. The mhbench_keys.ACTION_KEYS layout
    every MHBench baseline trains on (DP's model.py is the reference)."""
    flat_action = np.asarray(flat_action, dtype=np.float32)
    assert flat_action.shape[-1] == 35, f"expected 35D per-robot action, got {flat_action.shape}"
    return {
        "joint_targets": flat_action[0:31],
        "height": flat_action[31:32],
        "base_vel": flat_action[32:35],
    }


def _pack_dual_arm_action(flat_action: np.ndarray) -> dict:
    """A centralized 70D flat action -> one dict per robot."""
    flat_action = np.asarray(flat_action, dtype=np.float32)
    assert flat_action.shape[-1] == 70, f"expected 70D dual-robot action, got {flat_action.shape}"
    return {
        robot: _pack_single_robot_action(flat_action[i * 35 : (i + 1) * 35])
        for i, robot in enumerate(("robot_a", "robot_b"))
    }


def _mhbench_image_tensor_method(env_cfg_type: str):
    """The canvas builder for one MHBench mode, bound onto the upstream
    WorldActionRobotWinPolicy instance in place of its robotwin one.

    Training's geometry, reproduced in the training order: per-view frames
    arrive 240x320 (the env's native resolution and the data config's
    per-camera size, so the first Resize is a no-op), then each mode resizes
    once to its video_size canvas -- centralized stacks the pair vertically to
    480x320 and resizes to 384x320, decentralized resizes its single view to
    192x320. Both therefore show a view at the same 192x320, and both sides are
    a multiple of 32, which the VAE (16x) and DiT patch (2x2) require. PIL
    bilinear, as the upstream robotwin builder uses.
    """
    from PIL import Image
    import torch

    def _resize(image: np.ndarray, size_wh) -> np.ndarray:
        return np.asarray(
            Image.fromarray(image.astype(np.uint8), mode="RGB").resize(size_wh, resample=Image.BILINEAR),
            dtype=np.uint8,
        )

    def build(self, observation):
        images = observation["images"]
        if env_cfg_type == "unitree_g1x2_centralized":
            canvas = np.concatenate([images["ego_a"], images["ego_b"]], axis=0)  # (480, 320, 3)
            canvas = _resize(canvas, (320, 384))  # (384, 320, 3)
        else:
            canvas = _resize(images["ego"], (320, 192))  # (192, 320, 3)
        # Decoded observation arrays are read-only views (AGENTS.md); copy before
        # handing them to torch.
        tensor = torch.from_numpy(np.array(canvas, dtype=np.uint8, copy=True)).permute(2, 0, 1).unsqueeze(0).to(
            device=self.model.device, dtype=self.model.torch_dtype
        )
        return tensor * (2.0 / 255.0) - 1.0

    return build


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        self.action_type = self.model_cfg["action_type"]
        self.env_cfg_type = self.model_cfg["env_cfg_type"]
        self.action_horizon = 1
        self.replan_steps = int(self.model_cfg.get("replan_steps") or 24)
        self.default_instruction = str(
            self.model_cfg.get("default_instruction")
            or self.model_cfg.get("prompt")
            or "follow the instruction"
        )
        self.last_obs = None
        self.last_instruction = self.default_instruction
        self.model = None
        self.allow_dummy_policy = _is_true(self.model_cfg.get("allow_dummy_policy", False))

        self._mhbench = str(self.model_cfg.get("bench_name") or "") == "mhbench"
        if self._mhbench:
            self._init_mhbench()
            return

        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        checkpoint_path = self.model_cfg.get("checkpoint_path") or self.model_cfg.get("ckpt_setting")
        dataset_stats_path = self.model_cfg.get("dataset_stats_path")

        if self.allow_dummy_policy:
            print("[FastWAM] allow_dummy_policy=true; real checkpoint loading is skipped for debug flow only.")
            return

        if _is_none_like(checkpoint_path):
            raise FileNotFoundError("FastWAM requires checkpoint_path/ckpt_setting for real deployment.")
        if _is_none_like(dataset_stats_path):
            raise FileNotFoundError("FastWAM requires dataset_stats_path for real deployment.")

        self.model = self._load_upstream_policy(
            checkpoint_path, dataset_stats_path, sim_cfg_name=None, sim_task=None
        )
        self.action_horizon = int(self.model.action_horizon)
        self.replan_steps = int(self.model.replan_steps)

    # ------------------------------------------------------------------
    # MHBench: two Unitree G1 robots, centralized (one 70D policy) or
    # decentralized (one 35D policy per robot in the same server). Mirrors
    # DP's mhbench branches: agent state comes from obs["mhbench_state"]
    # (the standard XPolicyLab state slots have no room for robot_b) and the
    # action goes out as `mhbench_raw_action`, the layout
    # MHBenchTaskEnv.take_action requires.
    # ------------------------------------------------------------------

    def _init_mhbench(self):
        sim_task = self.model_cfg.get("sim_task")
        if _is_none_like(sim_task) or str(sim_task).startswith("robotwin"):
            sim_task = MHBENCH_SIM_TASK.get(self.env_cfg_type)
        if sim_task is None:
            raise ValueError(f"unsupported mhbench env_cfg_type for FastWAM: {self.env_cfg_type}")
        self._sim_task = str(sim_task)
        self._decentralized = self.env_cfg_type == "unitree_g1x2_decentralized"
        # What a decentralized checkpoint is: `shared` is one multitask policy
        # driving both agents, told them apart by the instruction each is
        # given; `per_robot` is the older pair, one checkpoint per (task,
        # robot). eval_policy.sbatch sends whichever the serving hook declares.
        self._shared = str(
            self.model_cfg.get("mhbench_decentralized_style") or "per_robot"
        ).strip().lower() == "shared"
        task = str(self.model_cfg.get("ckpt_name") or "").strip()
        if not task:
            raise ValueError("mhbench eval needs ckpt_name=<task> (e.g. cocarry)")

        self._policies: dict[str, Any] = {}
        self._instructions: dict[str, str] = {}
        self._obs_of: dict[str, dict] = {}
        self._batch: dict[int, dict[str, dict]] = {}

        if self.allow_dummy_policy:
            print("[FastWAM][mhbench] allow_dummy_policy=true; serving zero actions for debug flow only.")
            targets = ("robot_a", "robot_b") if self._decentralized else ("duo",)
            for target in targets:
                self._instructions[target] = self.default_instruction
            return

        if self._decentralized and self._shared:
            # One policy trained on every task and both roles, queried once per
            # agent. Its checkpoint is not named for a task, so `ckpt_name`
            # (multitask) is the run name itself and the instruction is what
            # tells the two agents apart -- the eval client sends each its own.
            ckpt = self.model_cfg.get("model_dir") or self.model_cfg.get("checkpoint_path")
            run_name = build_run_dir_name(dict(self.model_cfg))
            if _is_none_like(ckpt):
                ckpt = self._newest_weights(POLICY_DIR / "checkpoints" / run_name)
            stats = self._mhbench_stats_path(task)
            policy = self._load_upstream_policy(
                ckpt, stats, sim_cfg_name="sim_mhbench.yaml", sim_task=self._sim_task
            )
            fallback = self._mhbench_instruction(task)
            for robot in ("robot_a", "robot_b"):
                self._policies[robot] = policy
                self._instructions[robot] = fallback
            print(f"[FastWAM][mhbench] shared (multitask): {ckpt}")
        elif self._decentralized:
            for robot in ("robot_a", "robot_b"):
                ckpt = self.model_cfg.get(f"model_dir_{robot}")
                run_cfg = dict(self.model_cfg)
                run_cfg["ckpt_name"] = f"{task}_{robot}"
                run_name = build_run_dir_name(run_cfg)
                if _is_none_like(ckpt):
                    ckpt = self._newest_weights(POLICY_DIR / "checkpoints" / run_name)
                stats = self._mhbench_stats_path(f"{task}_{robot}")
                self._policies[robot] = self._load_upstream_policy(
                    ckpt, stats, sim_cfg_name="sim_mhbench.yaml", sim_task=self._sim_task
                )
                self._instructions[robot] = self._mhbench_instruction(f"{task}_{robot}")
                print(f"[FastWAM][mhbench] {robot}: {ckpt}")
        else:
            ckpt = self.model_cfg.get("model_dir") or self.model_cfg.get("checkpoint_path")
            run_cfg = dict(self.model_cfg)
            run_name = build_run_dir_name(run_cfg)
            if _is_none_like(ckpt):
                ckpt = self._newest_weights(POLICY_DIR / "checkpoints" / run_name)
            stats = self._mhbench_stats_path(task)
            self._policies["duo"] = self._load_upstream_policy(
                ckpt, stats, sim_cfg_name="sim_mhbench.yaml", sim_task=self._sim_task
            )
            self._instructions["duo"] = self._mhbench_instruction(task)
            print(f"[FastWAM][mhbench] centralized: {ckpt}")

        first = next(iter(self._policies.values()))
        self.action_horizon = int(first.action_horizon)
        self.replan_steps = int(first.replan_steps)

    def _mhbench_data_root(self, ckpt_name: str) -> Path:
        data_key = f"mhbench-{ckpt_name}-{self.env_cfg_type}-{self.action_type}"
        return POLICY_DIR / "data" / data_key

    def _mhbench_stats_path(self, ckpt_name: str) -> str:
        explicit = self.model_cfg.get("dataset_stats_path")
        if not _is_none_like(explicit):
            return str(explicit)
        return str(self._mhbench_data_root(ckpt_name) / "dataset_stats.json")

    def _mhbench_instruction(self, ckpt_name: str) -> str:
        """The instruction the checkpoint trained with, read from the converted
        dataset's own tasks.jsonl -- the same source training's prompt came
        from. deploy.yml's `prompt` (default_instruction) overrides it."""
        if self.model_cfg.get("prompt") or self.model_cfg.get("default_instruction"):
            return self.default_instruction
        tasks_file = self._mhbench_data_root(ckpt_name) / "lerobot" / "meta" / "tasks.jsonl"
        try:
            first = json.loads(tasks_file.read_text().splitlines()[0])
            return str(first["task"])
        except (OSError, IndexError, KeyError, json.JSONDecodeError):
            print(f"[FastWAM][mhbench] no readable {tasks_file}; using the default instruction")
            return self.default_instruction

    def _mhbench_instruction_for(self, target: str, wire: dict[str, str]) -> str:
        """What this agent is told to do, for one inference.

        The env publishes all three sentences per step (`mhbench_instruction`,
        from scripts/_task_text.py) and they are the authority: a shared
        multitask checkpoint has one dataset behind it and could not name its
        task any other way. deploy.yml's `prompt` overrides, and the
        dataset-read fallback covers a deploy.py outside the eval client.
        """
        if self.model_cfg.get("prompt") or self.model_cfg.get("default_instruction"):
            return self.default_instruction
        sentence = wire.get(target)
        if sentence and str(sentence).strip():
            return str(sentence)
        return self._instructions[target]

    def _newest_weights(self, run_dir: Path) -> str:
        """A servable weights file the trainer wrote under a run
        (<run>/checkpoints/weights/step_XXXXXX.pt): the newest, or the one
        `checkpoint_num` (EVAL_CKPT_NUM) names."""
        ckpt_num = self.model_cfg.get("checkpoint_num")
        if not _is_none_like(ckpt_num) and str(ckpt_num) != "latest":
            path = run_dir / "checkpoints" / "weights" / f"step_{int(ckpt_num):06d}.pt"
            if not path.is_file():
                raise FileNotFoundError(f"no FastWAM weights at {path} (checkpoint_num={ckpt_num})")
            return str(path)
        weights = sorted((run_dir / "checkpoints" / "weights").glob("step_*.pt"))
        if not weights:
            raise FileNotFoundError(
                f"no FastWAM weights under {run_dir}/checkpoints/weights "
                "(train first, or pass model_dir/checkpoint_path)"
            )
        return str(weights[-1])

    def _load_upstream_policy(self, checkpoint_path, dataset_stats_path, sim_cfg_name, sim_task):
        for path in (str(FASTWAM_ROOT), str(FASTWAM_SRC)):
            if path not in sys.path:
                sys.path.insert(0, path)
        from experiments.robotwin.fastwam_policy.deploy_policy import get_model

        upstream_cfg = dict(self.model_cfg)
        upstream_cfg["ckpt_setting"] = str(Path(str(checkpoint_path)).expanduser().resolve())
        upstream_cfg["dataset_stats_path"] = str(Path(str(dataset_stats_path)).expanduser().resolve())
        if sim_cfg_name is None:
            upstream_cfg.setdefault("sim_cfg_name", "sim_robotwin.yaml")
            upstream_cfg.setdefault("sim_task", "robotwin_uncond_3cam_384_1e-4")
        else:
            upstream_cfg["sim_cfg_name"] = sim_cfg_name
            upstream_cfg["sim_task"] = sim_task
        policy = get_model(upstream_cfg)
        if self._mhbench:
            # The upstream image builder is the robotwin three-view canvas;
            # MHBench's views and canvas differ, everything downstream of the
            # tensor does not. Replacing the bound method keeps the rest of
            # the upstream inference path canonical.
            policy._build_robotwin_image_tensor = types.MethodType(
                _mhbench_image_tensor_method(self.env_cfg_type), policy
            )
        return policy

    def _encode_mhbench(self, obs: dict) -> dict[str, dict]:
        """One env observation -> per-policy upstream observations."""
        vision = obs["vision"]
        state = obs.get("mhbench_state")
        if state is None:
            if not self.allow_dummy_policy:
                raise KeyError(
                    "obs has no 'mhbench_state' -- the MHBench env client publishes it; "
                    "this branch cannot run against a generic client"
                )
            # The debug client's generic observation carries no mhbench_state;
            # zeros keep the wiring check (encode -> ws -> action packing)
            # running under allow_dummy_policy without touching the real path.
            state = {
                robot: {"joint_pos": np.zeros(43, dtype=np.float32)}
                for robot in ("robot_a", "robot_b")
            }
        ego = {
            robot: _standardize_rgb(vision[MHBENCH_CAMERA_SLOT[robot]]["color"])
            for robot in ("robot_a", "robot_b")
            if MHBENCH_CAMERA_SLOT[robot] in vision
        }
        encoded: dict[str, dict] = {}
        if self._decentralized:
            for robot in ("robot_a", "robot_b"):
                encoded[robot] = {
                    "images": {"ego": ego[robot]},
                    "joint_action": {
                        "vector": np.asarray(state[robot]["joint_pos"], dtype=np.float32)
                    },
                }
        else:
            encoded["duo"] = {
                "images": {"ego_a": ego["robot_a"], "ego_b": ego["robot_b"]},
                "joint_action": {
                    "vector": np.concatenate(
                        [
                            np.asarray(state["robot_a"]["joint_pos"], dtype=np.float32),
                            np.asarray(state["robot_b"]["joint_pos"], dtype=np.float32),
                        ]
                    )
                },
            }
        # The sentences ride with the observation rather than on the model, so
        # a batch of envs cannot hand one env's instruction to another's
        # inference. `duo` is the pair's, for the centralized policy.
        encoded[INSTRUCTIONS_KEY] = dict(obs.get("mhbench_instruction") or {})
        return encoded

    def _mhbench_chunks(self, per_policy_obs: dict[str, dict]) -> list[dict]:
        """Run every policy once and fold the chunks into
        `mhbench_raw_action` steps."""
        if self.allow_dummy_policy:
            zero = {
                robot: _pack_single_robot_action(np.zeros(35, dtype=np.float32))
                for robot in ("robot_a", "robot_b")
            }
            return [{"mhbench_raw_action": dict(zero)} for _ in range(self.replan_steps)]

        wire = per_policy_obs.get(INSTRUCTIONS_KEY) or {}
        chunks = {}
        for target, policy in self._policies.items():
            chunk = np.asarray(
                policy._infer_action_chunk(
                    per_policy_obs[target], self._mhbench_instruction_for(target, wire)
                ),
                dtype=np.float32,
            )
            if chunk.ndim == 1:
                chunk = chunk[None, :]
            chunks[target] = chunk[: self.replan_steps]

        steps = min(chunk.shape[0] for chunk in chunks.values())
        if self._decentralized:
            return [
                {
                    "mhbench_raw_action": {
                        robot: _pack_single_robot_action(chunks[robot][t])
                        for robot in ("robot_a", "robot_b")
                    }
                }
                for t in range(steps)
            ]
        return [{"mhbench_raw_action": _pack_dual_arm_action(chunks["duo"][t])} for t in range(steps)]

    # ------------------------------------------------------------------
    # RoboTwin/RoboDojo path (unchanged behaviour)
    # ------------------------------------------------------------------

    def _encode_obs_for_fastwam(self, obs: dict) -> dict:
        vision = obs["vision"]
        adapted = {
            "observation": {
                "head_camera": {"rgb": _standardize_rgb(vision["cam_head"]["color"])},
                "left_camera": {"rgb": _standardize_rgb(vision["cam_left_wrist"]["color"])},
                "right_camera": {"rgb": _standardize_rgb(vision["cam_right_wrist"]["color"])},
            },
            "joint_action": {
                "vector": pack_robot_state(
                    obs,
                    self.action_type,
                    self.robot_action_dim_info,
                    source_type="obs",
                    state_type="state",
                ).astype(np.float32)
            },
        }
        return adapted

    def update_obs(self, obs):
        if self._mhbench:
            self.last_obs = self._encode_mhbench(obs)
            return
        self.last_obs = self._encode_obs_for_fastwam(obs)
        self.last_instruction = _get_instruction(obs, self.default_instruction)

    def update_obs_batch(self, obs_list):
        if not obs_list:
            raise ValueError("update_obs_batch received an empty observation list.")
        if self._mhbench:
            self._batch = {int(obs["env_idx"]): self._encode_mhbench(obs) for obs in obs_list}
            return
        self._batch_obs = {}
        self._batch_instruction = {}
        for obs in obs_list:
            env_idx = int(obs["env_idx"])
            self._batch_obs[env_idx] = self._encode_obs_for_fastwam(obs)
            self._batch_instruction[env_idx] = _get_instruction(obs, self.default_instruction)

    def _zero_actions(self):
        dim = sum(self.robot_action_dim_info["arm_dim"]) + sum(self.robot_action_dim_info["ee_dim"])
        zeros = np.zeros((self.replan_steps, dim), dtype=np.float32)
        return unpack_robot_state(zeros, self.action_type, self.robot_action_dim_info, source_type="obs")

    def _infer_actions(self, obs, instruction):
        if self.allow_dummy_policy:
            return self._zero_actions()
        if obs is None:
            raise ValueError("No observation is available. Call update_obs() before get_action().")
        action_chunk = self.model._infer_action_chunk(obs, instruction)
        action_chunk = np.asarray(action_chunk, dtype=np.float32)
        if action_chunk.ndim == 1:
            action_chunk = action_chunk[None, :]
        n_exec = min(self.replan_steps, action_chunk.shape[0])
        action_chunk = action_chunk[:n_exec]
        return unpack_robot_state(action_chunk, self.action_type, self.robot_action_dim_info, source_type="obs")

    def get_action(self):
        if self._mhbench:
            if self.last_obs is None:
                raise ValueError("No observation is available. Call update_obs() before get_action().")
            return self._mhbench_chunks(self.last_obs)
        return self._infer_actions(self.last_obs, self.last_instruction)

    def get_action_batch(self, env_idx_list):
        if self._mhbench:
            if not self._batch:
                raise ValueError("No batch observation is available. Call update_obs_batch() first.")
            return [self._mhbench_chunks(self._batch[int(env_idx)]) for env_idx in env_idx_list]
        if not hasattr(self, "_batch_obs"):
            raise ValueError("No batch observation is available. Call update_obs_batch() first.")
        return [
            self._infer_actions(self._batch_obs[int(env_idx)], self._batch_instruction[int(env_idx)])
            for env_idx in env_idx_list
        ]

    def reset(self):
        self.last_obs = None
        self.last_instruction = self.default_instruction
        if self._mhbench:
            self._batch = {}
            for policy in self._policies.values():
                policy.reset()
            return
        if self.model is not None:
            self.model.reset()
