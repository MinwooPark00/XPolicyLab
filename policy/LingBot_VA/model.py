import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_CUR_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CUR_DIR.parents[2]

for _path in (str(_REPO_ROOT),):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)

from .prepare_merged_ckpt import build_merged_ckpt
from .lingbot_va.evaluation.robotwin.websocket_client_policy import WebsocketClientPolicy
from .lingbot_va.wan_va.configs import VA_CONFIGS

DEFAULT_VA_SERVER_HOST = "127.0.0.1"
DEFAULT_VA_SERVER_PORT = 10001
DEFAULT_CONFIG_NAME = "robotwin30_train"

# ---------------------------------------------------------------------------
# MHBench
#
# Two Unitree G1s, one shared multitask policy: the same weights drive both
# agents and only the instruction tells them apart. LingBot-VA rolls out
# autoregressively, so each agent needs its own KV cache and its own video
# history -- the server keys those by `session` (wan_va_server.SESSION_STATE),
# and this adapter drives one session per agent through the same protocol the
# official RoboTwin eval uses (reset(prompt) -> infer -> compute_kv_cache).
#
# Camera slots are historical: the env sends ego_a as cam_left_wrist and ego_b
# as cam_right_wrist (mhbench_xpolicylab_env.py's _VISION_SLOT), which is how
# every mhbench adapter reads them.
# ---------------------------------------------------------------------------
MHBENCH_CAMERA_SLOT = {"robot_a": "cam_left_wrist", "robot_b": "cam_right_wrist"}
MHBENCH_AGENTS = ("robot_a", "robot_b")
MHBENCH_CONFIG_NAME = "mhbench"
MHBENCH_ACTION_DIM = 35


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, str) and value.strip().lower() in {"", "none", "null"}


def _pack_single_robot_action(flat_action: np.ndarray) -> dict:
    """One robot's 35D action -> MHBenchTaskEnv.take_action's
    {joint_targets, height, base_vel} (the mhbench_keys.ACTION_KEYS layout)."""
    flat_action = np.asarray(flat_action, dtype=np.float32)
    assert flat_action.shape[-1] == MHBENCH_ACTION_DIM, \
        f"expected 35D per-robot action, got {flat_action.shape}"
    return {
        "joint_targets": flat_action[0:31],
        "height": flat_action[31:32],
        "base_vel": flat_action[32:35],
    }


class _Rollout:
    """One agent's client-side half of the closed loop."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.first_obs = None
        self.skip_leading = True
        self.latest_raw_chunk = None
        self.exec_buffer = []
        self.last_exec_steps = 0

# 30-dim LingBot layout -> 14-dim RoboDojo joint.
JOINT_CONTROL_INDICES = np.array([
    14, 15, 16, 17, 18, 19,
    28,
    21, 22, 23, 24, 25, 26,
    29,
], dtype=np.int64)


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


def ensure_hwc_uint8(image):

    image = np.asarray(image)

    if image.ndim != 3:
        raise ValueError(f"Expected image ndim=3, got shape {image.shape}")

    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    if image.shape[-1] in (1, 3):
        return image
    if image.shape[0] in (1, 3):
        return np.transpose(image, (1, 2, 0))
    raise ValueError(f"Unsupported image shape: {image.shape}")


def encode_obs(observation, action_type, robot_action_dim_info, default_prompt):
    images = {
        "cam_high": ensure_hwc_uint8(
            extract_image(observation, ["cam_high", "cam_head", "head_camera", "top_camera"])
        ),
        "cam_left_wrist": ensure_hwc_uint8(
            extract_image(observation, ["cam_left_wrist", "left_camera", "left_wrist", "wrist_left"])
        ),
        "cam_right_wrist": ensure_hwc_uint8(
            extract_image(observation, ["cam_right_wrist", "right_camera", "right_wrist", "wrist_right"])
        ),
    }

    if robot_action_dim_info is None:
        state = np.zeros((1,), dtype=np.float32)
    else:
        state = pack_robot_state(
            observation,
            action_type,
            robot_action_dim_info,
            source_type="obs",
        ).astype(np.float32)

    prompt = observation.get("instruction") or observation.get("prompt") or default_prompt

    return {
        "observation.images.cam_high": images["cam_high"],
        "observation.images.cam_left_wrist": images["cam_left_wrist"],
        "observation.images.cam_right_wrist": images["cam_right_wrist"],
        "observation.state": state,
        "task": prompt,
    }


class Model(ModelTemplate):
    """RoboDojo bridge client -> official wan_va_server (WebsocketPolicyServer)."""

    def __init__(self, model_cfg) -> None:
        self.model_cfg = dict(model_cfg)

        self.task_name = self.model_cfg.get("task_name", "default_task")
        self.action_type = self.model_cfg["action_type"]
        # Explicitly-configured prompt (deploy.yml) is authoritative and overrides
        # any env-injected obs["instruction"]; None when not configured.
        self._prompt_override = self.model_cfg.get("prompt")
        self.default_prompt = self._prompt_override or (
            "Cover the blocks from left to right, remember their colors, then uncover them in the order: red, green, and blue."
        )

        self._mhbench = str(self.model_cfg.get("bench_name") or "") == "mhbench"
        config_name = self.model_cfg.get("config_name", DEFAULT_CONFIG_NAME)
        if self._mhbench and (_is_none_like(config_name) or str(config_name).startswith("robotwin")):
            config_name = MHBENCH_CONFIG_NAME
        self.job_config = VA_CONFIGS[config_name]
        self.obs_cam_keys = list(self.job_config.obs_cam_keys)
        self.action_dim = int(self.job_config.action_dim)
        self.action_per_frame = int(self.job_config.action_per_frame)

        env_cfg = self.model_cfg.get("env_cfg")
        if env_cfg is None:
            self.robot_action_dim_info = None
        else:
            try:
                self.robot_action_dim_info = get_robot_action_dim_info(env_cfg)
            except FileNotFoundError:
                print(f"[WARN] env_cfg '{env_cfg}' not found, fallback to raw action mode.")
                self.robot_action_dim_info = None

        self.observation_window: list[dict[str, Any]] | None = None
        self._latest_env_idx_list: list[int] = [0]
        self._skip_leading_chunk_on_next_action = True
        self._backend: subprocess.Popen | None = None
        self._first_observation: dict[str, Any] | None = None
        self._latest_raw_action_chunk: np.ndarray | None = None
        self._exec_obs_buffer: list[dict[str, Any]] = []
        self._last_exec_step_count = 0
        self.rollout_mode = str(self.model_cfg.get("rollout_mode", "closed_loop"))
        self.keyframe_stride = self.model_cfg.get("keyframe_stride")
        self.chunk_exec_steps = self.model_cfg.get("chunk_exec_steps")
        self.initial_action_skip = self.model_cfg.get("initial_action_skip")

        # Imagined-video saving: when enabled, ask the server to decode predicted
        # latents each chunk and persist them under result_dir/imagined/.
        self.save_imagined_video = bool(self.model_cfg.get("save_imagined_video", False))
        self._imagined_dir: Path | None = None
        self._chunk_seq = 0
        if self.save_imagined_video:
            result_dir = Path(self.model_cfg.get("result_dir", "./results/LingBot_VA"))
            self._imagined_dir = result_dir / "imagined"
            self._imagined_dir.mkdir(parents=True, exist_ok=True)

        self.va_server_host = self.model_cfg.get("va_server_host", DEFAULT_VA_SERVER_HOST)
        self.va_server_port = int(self.model_cfg.get("va_server_port", DEFAULT_VA_SERVER_PORT))

        print(f"[LingBot_VA] rollout_mode={self.rollout_mode}", flush=True)
        print(
            f"[LingBot_VA] va_server=ws://{self.va_server_host}:{self.va_server_port}, "
            f"config_name={config_name}",
            flush=True,
        )
        print(f"[LingBot_VA] save_imagined_video={self.save_imagined_video}, "
              f"model_cfg keys={list(self.model_cfg.keys())[:10]}", flush=True)

        if self._mhbench:
            self._init_mhbench()
        self._maybe_launch_backend()
        self._ws = self._connect_client()
        print(
            f"[LingBot_VA] keyframe_stride={self._resolve_keyframe_stride()}, "
            f"chunk_exec_steps={self._resolve_chunk_exec_steps()}, "
            f"initial_action_skip={self._resolve_initial_action_skip()}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # MHBench
    # ------------------------------------------------------------------

    def _init_mhbench(self):
        env_cfg_type = str(self.model_cfg.get("env_cfg_type") or "")
        if env_cfg_type and env_cfg_type not in ("unitree_g1x2_decentralized",):
            raise ValueError(
                "LingBot_VA serves MHBench decentralized only (one shared policy per "
                f"agent); got env_cfg_type={env_cfg_type!r}. There is no 70D centralized "
                "checkpoint for this model.")
        style = str(self.model_cfg.get("mhbench_decentralized_style") or "shared")
        if style != "shared":
            raise ValueError(
                f"LingBot_VA has only the shared multitask checkpoint; got style={style!r}")
        if int(self.job_config.action_dim) != MHBENCH_ACTION_DIM:
            raise ValueError(
                f"config {self.job_config.__name__} has action_dim="
                f"{self.job_config.action_dim}, MHBench needs {MHBENCH_ACTION_DIM}")
        if getattr(self.job_config, "norm_stat_source", None) == "":
            raise ValueError(
                "no action normalisation: the checkpoint's lingbot_norm_stat.json was "
                "not found. serve/LingBot_VA.sh exports LINGBOT_NORM_STAT from the "
                "checkpoint directory; a checkpoint copied without it needs the value "
                "from the dataset it was trained on.")
        self.action_dim = MHBENCH_ACTION_DIM
        self._rollouts = {agent: _Rollout() for agent in MHBENCH_AGENTS}
        self._last_encoded: dict[str, dict] | None = None
        print(f"[LingBot_VA][mhbench] shared multitask policy, sessions="
              f"{list(MHBENCH_AGENTS)}, action_per_frame={self.action_per_frame}",
              flush=True)

    def _maybe_launch_backend(self):
        """Start the wan_va backend if nothing is answering on its port.

        MHBench's runner starts one policy server per job and knows nothing
        about LingBot's second process, so the adapter brings it up itself --
        the same thing eval.sh does outside the runner.
        """
        if not self._mhbench or os.environ.get("LINGBOT_VA_NO_AUTOLAUNCH") == "1":
            return
        import socket

        def answering():
            with socket.socket() as sock:
                sock.settimeout(1.0)
                return sock.connect_ex((self.va_server_host, self.va_server_port)) == 0

        if answering():
            print("[LingBot_VA][mhbench] wan_va backend already up", flush=True)
            return

        checkpoint = self.model_cfg.get("model_dir") or self.model_cfg.get("checkpoint_path")
        if _is_none_like(checkpoint):
            raise ValueError("no checkpoint to serve: set model_dir in the deploy overrides")
        base = self.model_cfg.get("base_model_path") or os.environ.get("LINGBOT_VA_BASE_MODEL_PATH")
        if _is_none_like(base):
            raise ValueError(
                "no base_model_path: the served transformer needs the base vae/"
                "text_encoder/tokenizer beside it (LINGBOT_VA_BASE_MODEL_PATH)")
        # One merged dir per server process: eval shards run concurrently and
        # a shared .merged_ckpt would have them relinking each other's.
        merged = str(build_merged_ckpt(
            str(checkpoint), str(base),
            Path(os.environ.get("TMPDIR", "/tmp")) / f"lingbot_merged_{os.getpid()}"))

        env = dict(os.environ)
        env["LINGBOT_VA_BASE_MODEL_PATH"] = merged
        env.setdefault("MASTER_ADDR", "127.0.0.1")
        env.setdefault("MASTER_PORT", str(self.va_server_port + 1000))
        cmd = [sys.executable, "-m", "wan_va.wan_va_server",
               "--config-name", MHBENCH_CONFIG_NAME,
               "--port", str(self.va_server_port),
               "--save_root", str(Path(self.model_cfg.get("result_dir", "./results/LingBot_VA")) / "wan_va")]
        print(f"[LingBot_VA][mhbench] launching backend: {' '.join(cmd)}", flush=True)
        self._backend = subprocess.Popen(cmd, cwd=str(_CUR_DIR / "lingbot_va"), env=env)

        deadline = time.time() + float(os.environ.get("LINGBOT_VA_BACKEND_TIMEOUT_S", 1800))
        while time.time() < deadline:
            if self._backend.poll() is not None:
                raise RuntimeError(
                    f"wan_va backend exited with {self._backend.returncode} before serving")
            if answering():
                print("[LingBot_VA][mhbench] backend is up", flush=True)
                return
            time.sleep(2.0)
        raise TimeoutError("wan_va backend did not open its port in time")

    def _encode_mhbench(self, obs: dict) -> dict[str, dict]:
        """One env observation -> one encoded observation per agent."""
        vision = obs["vision"]
        wire = dict(obs.get("mhbench_instruction") or {})
        encoded = {}
        for agent in MHBENCH_AGENTS:
            slot = MHBENCH_CAMERA_SLOT[agent]
            if slot not in vision:
                raise KeyError(f"observation has no {slot} view for {agent}")
            frame = vision[slot]
            frame = frame["color"] if isinstance(frame, dict) else frame
            sentence = wire.get(agent) or self._prompt_override or self.default_prompt
            encoded[agent] = {
                "observation.images.ego": ensure_hwc_uint8(frame),
                "task": sentence,
            }
        return encoded

    def _mhbench_predict(self, agent: str) -> np.ndarray:
        """One agent's next chunk, driving its own server session."""
        roll = self._rollouts[agent]
        observation = self._last_encoded[agent]

        if roll.first_obs is None:
            self._ws.infer({"session": agent, "reset": True, "prompt": observation["task"]})
            print(f"[LingBot_VA][mhbench] {agent} instruction: {observation['task']!r}",
                  flush=True)
            roll.first_obs = observation
            roll.skip_leading = True
        else:
            self._mhbench_commit(agent)

        payload = self._to_engine_obs(roll.first_obs)
        payload["session"] = agent
        result = self._ws.infer(payload)
        roll.latest_raw_chunk = np.asarray(result["action"])

        chunk = self._format_action_chunk(result["action"])
        if chunk.shape[1] != MHBENCH_ACTION_DIM:
            raise ValueError(f"{agent}: server returned {chunk.shape[1]}D actions, "
                             f"expected {MHBENCH_ACTION_DIM}")
        if roll.skip_leading:
            skip = self._resolve_initial_action_skip()
            if skip >= chunk.shape[0]:
                raise ValueError(f"{agent}: initial skip {skip} >= chunk {chunk.shape[0]}")
            chunk = chunk[skip:]
            roll.skip_leading = False
        chunk = self._maybe_truncate_action_chunk(chunk)
        roll.last_exec_steps = int(chunk.shape[0])
        roll.exec_buffer = []
        return chunk

    def _mhbench_commit(self, agent: str):
        """Feed the frames actually executed back into the agent's KV cache."""
        roll = self._rollouts[agent]
        if roll.latest_raw_chunk is None:
            return
        stride = self._resolve_keyframe_stride()
        exec_steps = roll.last_exec_steps or len(roll.exec_buffer)
        usable = min(len(roll.exec_buffer), max(exec_steps - 1, 0))
        key_frames = [roll.exec_buffer[i] for i in range(stride - 1, usable, stride)]
        if len(key_frames) < exec_steps // stride and self._last_encoded is not None:
            key_frames.append(self._last_encoded[agent])
        if not key_frames:
            key_frames = list(roll.exec_buffer)
        if not key_frames:
            return
        payload = self._to_engine_obs_batch(key_frames, state=roll.latest_raw_chunk)
        payload["session"] = agent
        payload["compute_kv_cache"] = True
        self._ws.infer(payload)

    def _mhbench_chunks(self) -> list[dict]:
        if self._last_encoded is None:
            raise ValueError("No observation is available. Call update_obs() before get_action().")
        chunks = {agent: self._mhbench_predict(agent) for agent in MHBENCH_AGENTS}
        steps = min(chunk.shape[0] for chunk in chunks.values())
        return [
            {"mhbench_raw_action": {
                agent: _pack_single_robot_action(chunks[agent][t]) for agent in MHBENCH_AGENTS}}
            for t in range(steps)
        ]

    def _connect_client(self) -> WebsocketClientPolicy:
        return WebsocketClientPolicy(
            host=self.va_server_host,
            port=self.va_server_port,
        )

    def _soft_reset_inference_state(self):
        """Clear server KV/VAE state (receding-horizon replan)."""
        self._ws.infer({"reset": True, "prompt": self.default_prompt})
        self._skip_leading_chunk_on_next_action = True
        self._first_observation = None
        self._latest_raw_action_chunk = None
        self._exec_obs_buffer = []

    def _to_engine_obs(self, observation):
        # Match deploy_policy.py: infer only sends {obs, prompt}, not state.
        # The server _infer() ignores obs['state']; only compute_kv_cache uses
        # state (passed separately as the full action chunk).
        image_dict = {key: observation[key] for key in self.obs_cam_keys}
        return {
            "obs": [image_dict],
            "prompt": observation["task"],
        }

    def _to_engine_obs_batch(self, observation_list, state=None):
        obs_batch = []
        prompt = self.default_prompt

        for observation in observation_list:
            image_dict = {key: observation[key] for key in self.obs_cam_keys}
            obs_batch.append(image_dict)
            prompt = observation.get("task", prompt)

        payload = {
            "obs": obs_batch,
            "prompt": prompt,
        }
        if state is not None:
            payload["state"] = state
        return payload

    def _format_action_chunk(self, action):
        action = np.asarray(action)

        if action.ndim == 3:
            action = np.transpose(action, (1, 2, 0)).reshape(-1, action.shape[0])
        elif action.ndim == 2 and action.shape[0] == self.action_dim:
            action = action.T

        return action

    def _convert_to_joint_control_chunk(self, action_chunk):
        action_chunk = np.asarray(action_chunk)

        if action_chunk.ndim != 2:
            raise ValueError(f"Expected action chunk with ndim=2, got shape {action_chunk.shape}.")

        if action_chunk.shape[1] == len(JOINT_CONTROL_INDICES):
            return action_chunk

        if action_chunk.shape[1] != 30:
            raise ValueError(
                "LingBot_VA joint-control conversion expects raw action dim 30 or already-converted dim 14, "
                f"got {action_chunk.shape[1]}."
            )

        return action_chunk[:, JOINT_CONTROL_INDICES]

    def _resolve_keyframe_stride(self) -> int:
        if self.keyframe_stride is not None:
            return max(1, int(self.keyframe_stride))
        if self.action_per_frame <= 0:
            return 1
        return max(1, self.action_per_frame // 4)

    def _resolve_initial_action_skip(self) -> int:
        if self.initial_action_skip is not None:
            return max(0, int(self.initial_action_skip))
        return self.action_per_frame

    def _resolve_chunk_exec_steps(self) -> int | None:
        if self.chunk_exec_steps is None:
            return None
        steps = int(self.chunk_exec_steps)
        return steps if steps > 0 else None

    def _maybe_trim_initial_action_chunk(self, action_chunk):
        action_chunk = np.asarray(action_chunk)

        if not self._skip_leading_chunk_on_next_action:
            return action_chunk

        skip_count = self._resolve_initial_action_skip()
        if skip_count <= 0:
            return action_chunk

        if action_chunk.shape[0] <= skip_count:
            raise ValueError(
                "Initial-action trimming would remove the whole chunk: "
                f"chunk_len={action_chunk.shape[0]}, skip_count={skip_count}."
            )

        self._skip_leading_chunk_on_next_action = False
        return action_chunk[skip_count:]

    def _maybe_truncate_action_chunk(self, action_chunk):
        exec_steps = self._resolve_chunk_exec_steps()
        if exec_steps is None or action_chunk.shape[0] <= exec_steps:
            return action_chunk
        return action_chunk[:exec_steps]

    def _predict_chunk(self, observation):
        payload = self._to_engine_obs(observation)
        if self.save_imagined_video:
            payload["save_visualization"] = True
        result = self._ws.infer(payload)
        self._latest_raw_action_chunk = np.asarray(result["action"])
        if self.save_imagined_video and result.get("video") is not None:
            self._save_imagined_chunk(np.asarray(result["video"]))
        action = self._format_action_chunk(result["action"])
        action = self._convert_to_joint_control_chunk(action)
        action = self._maybe_trim_initial_action_chunk(action)
        action = self._maybe_truncate_action_chunk(action)
        # Remember how many env steps this chunk will actually execute so that
        # _commit_executed_frames samples keyframes over exactly that span
        # (matching the official robotwin eval cadence).
        self._last_exec_step_count = int(action.shape[0])
        return action

    def _save_imagined_chunk(self, video: np.ndarray) -> None:
        """Persist one chunk's imagined frames: npz + a tiled jpg preview."""
        if self._imagined_dir is None:
            return
        seq = self._chunk_seq
        self._chunk_seq += 1
        np.savez_compressed(self._imagined_dir / f"chunk_{seq:05d}.npz", video=video)
        try:
            from PIL import Image

            frames = np.asarray(video)
            if frames.ndim == 4:
                if frames.shape[-1] not in (1, 3) and frames.shape[1] in (1, 3):
                    frames = np.transpose(frames, (0, 2, 3, 1))
                if frames.shape[-1] == 1:
                    frames = np.repeat(frames, 3, axis=-1)
                frames = frames.astype(np.float32)
                frames = np.clip(frames, 0.0, 1.0) if frames.max() <= 1.0 else np.clip(frames, 0, 255)
                frames = (frames * 255.0).astype(np.uint8) if frames.max() <= 1.0 else frames.astype(np.uint8)
                n, h, w, _ = frames.shape
                cols = min(n, 4)
                rows = int(np.ceil(n / cols))
                tile = Image.new("RGB", (w * cols, h * rows), (0, 0, 0))
                for i, frame in enumerate(frames):
                    tile.paste(Image.fromarray(frame.astype(np.uint8)), ((i % cols) * w, (i // cols) * h))
                tile.save(self._imagined_dir / f"chunk_{seq:05d}.jpg", quality=85)
        except Exception as exc:
            print(f"[LingBot_VA] failed to save imagined jpg: {exc}", flush=True)

    def _commit_executed_frames(self):
        if self._latest_raw_action_chunk is None:
            return
        stride = self._resolve_keyframe_stride()
        exec_steps = int(getattr(self, "_last_exec_step_count", 0)) or len(self._exec_obs_buffer)
        # deploy_policy.py samples keyframes at (j+1)%action_per_frame==0, i.e. at
        # steps stride, 2*stride, ..., exec_steps (1-indexed, inclusive of last).
        # The RoboDojo harness calls update_obs after every take_action EXCEPT the
        # last (it breaks before update_obs), so the buffer holds exec_steps-1
        # frames. We sample from the buffer and append observation_window[0]
        # (which IS the post-last-step observation set by the next update_obs)
        # to match deploy_policy's inclusive last keyframe.
        n_obs = len(self._exec_obs_buffer)
        usable = min(n_obs, exec_steps - 1)  # buffer has at most exec_steps-1
        # Indices (0-based) of keyframes in buffer: stride-1, 2*stride-1, ...
        indices = list(range(stride - 1, usable, stride))
        key_frames = [self._exec_obs_buffer[i] for i in indices if i < n_obs]

        # Append the "last step" keyframe from observation_window if available.
        # observation_window[0] is the obs AFTER the last take_action of this
        # chunk (set by the harness's next update_obs before get_action).
        if self.observation_window is not None and len(key_frames) < (exec_steps // stride):
            key_frames.append(self.observation_window[0])

        if not key_frames:
            key_frames = self._exec_obs_buffer[:]

        cache_obs = self._to_engine_obs_batch(key_frames, state=self._latest_raw_action_chunk)
        cache_obs["compute_kv_cache"] = True
        self._ws.infer(cache_obs)
        self._exec_obs_buffer = []

    def infer(self, observation):
        if observation.get("reset"):
            self.reset(
                checkpoint_path=observation.get("checkpoint_path"),
            )
            return dict(action=None)
        return dict(action=self._predict_chunk(observation))

    def update_obs(self, obs):
        if self._mhbench:
            encoded = self._encode_mhbench(obs)
            if self._last_encoded is not None:
                for agent in MHBENCH_AGENTS:
                    self._rollouts[agent].exec_buffer.append(encoded[agent])
            self._last_encoded = encoded
            return
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        self._latest_env_idx_list = [obs.get("env_idx", index) for index, obs in enumerate(obs_list)]
        encoded_obs_list = [
            encode_obs(obs, self.action_type, self.robot_action_dim_info, self.default_prompt)
            for obs in obs_list
        ]
        self.observation_window = encoded_obs_list
        if self._first_observation is not None:
            self._exec_obs_buffer.append(encoded_obs_list[0])

    def _reset_server_with_instruction(self, observation):
        """Reset the server with the episode instruction before the first chunk.

        The server encodes the prompt embedding only at reset; matching the
        official robotwin eval (reset(prompt=instruction) -> infer(first_obs)).
        """
        # obs["instruction"] (carried as encoded "task") wins over the deploy.yml
        # prompt override; fall back to configured prompt, then default_prompt.
        prompt = observation.get("task") or self._prompt_override or self.default_prompt
        print(f"[LingBot_VA] reset server with instruction: {prompt!r}", flush=True)
        self._ws.infer({"reset": True, "prompt": prompt})
        self._skip_leading_chunk_on_next_action = True
        self._latest_raw_action_chunk = None
        self._exec_obs_buffer = []
        self._chunk_seq = 0

    def get_action(self, **kwargs):
        if self._mhbench:
            return self._mhbench_chunks()
        if self.observation_window is None:
            raise AssertionError("update_obs or update_obs_batch first!")

        if self.rollout_mode == "receding_horizon":
            self._soft_reset_inference_state()
            action_chunk = self._predict_chunk(self.observation_window[0])
        else:
            if self._first_observation is None:
                self._reset_server_with_instruction(self.observation_window[0])
                self._first_observation = self.observation_window[0]
            else:
                self._commit_executed_frames()
            action_chunk = self._predict_chunk(self._first_observation)

        if self.robot_action_dim_info is None:
            return action_chunk

        return unpack_robot_state(
            action_chunk,
            self.action_type,
            self.robot_action_dim_info,
            source_type="obs",
        )

    def get_action_batch(self, env_idx_list=None, **kwargs):
        # Required ModelTemplate hook. The previous implementation inferred env0
        # once and copied that chunk to every env; do not restore that behavior.
        raise NotImplementedError(
            "LingBot_VA wan_va_server keeps one global KV/VAE cache and cannot "
            "evaluate multiple envs in one process. Keep eval_batch: false in deploy.yml."
        )

    def get_action_per_frame(self):
        return self.action_per_frame

    def get_keyframe_interval(self):
        return self._resolve_keyframe_stride()

    def reset(self, checkpoint_path=None) -> None:
        if self._mhbench:
            # Sessions are re-reset with the episode's instruction on the next
            # get_action; nothing to say to the server here.
            for roll in self._rollouts.values():
                roll.reset()
            self._last_encoded = None
            return
        if checkpoint_path is not None:
            print(
                "[WARN] checkpoint_path reload is not supported in websocket bridge mode; "
                "restart wan_va_server with the new checkpoint.",
                flush=True,
            )
        self._ws.infer({"reset": True, "prompt": self.default_prompt})
        self.observation_window = None
        self._latest_env_idx_list = [0]
        self._skip_leading_chunk_on_next_action = True
        self._first_observation = None
        self._latest_raw_action_chunk = None
        self._exec_obs_buffer = []
        self._chunk_seq = 0

    def close(self) -> None:
        """Stop the backend this adapter started, if it started one."""
        if getattr(self, "_backend", None) is None:
            return
        if self._backend.poll() is None:
            self._backend.terminate()
            try:
                self._backend.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._backend.kill()
        self._backend = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
