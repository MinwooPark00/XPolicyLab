"""
DreamZero G1 SIMPLE HTTP policy server.

Serves the DreamZero G1 wholebody LoRA checkpoint (14B Wan2.1 backbone +
wan_flow_matching_action_tf head, trained by scripts/train/g1_simple_training_lora.sh)
over an HTTP endpoint that matches USC PSI Lab's SIMPLE `HttpActionClient`
protocol — so SIMPLE's baseline wrapper (`src/simple/baselines/dreamzero.py`,
via `simple.baselines.client.HttpActionClient`) can query it just like it
queries Psi0 today.

Protocol (/act endpoint, matching `simple/baselines/client.py:RequestMessage`):

    request = {
        "image": {"rgb_head_stereo_left": np.ndarray(H, W, 3)},
        "instruction": "pick up the cracker box",
        "history": {"session_id": str, "episode_index": int,
                    "step_index": int, "reset": Optional[bool]},
        "state": {"states": np.ndarray(1, 32)},         # Psi0-flat state
        "condition": {},
        "gt_action": [],
        "dataset_name": "dreamzero_g1_simple",
        "timestamp": str,
    }

    response = {
        "action":     np.ndarray(chunk_len, 36),   # Psi0-layout action chunk
        "err":        float,                        # 0.0 on success, or error string
        "traj_image": np.ndarray(1, 1, 3),          # placeholder, server does not render
    }

All numpy arrays on the wire are serialized via the SIMPLE convention:

    {"__numpy__": base64, "dtype": ..., "shape": ...}

Usage:

    # Single-GPU
    python eval_utils/serve_dreamzero_g1_simple.py \
        --model-path /workspace/checkpoints/dreamzero_g1_simple_mixture_lora \
        --port 22085 --host 0.0.0.0

    # lora checkpoint
    DEBUG=1 python eval_utils/serve_dreamzero_g1_simple.py \
        --model-path checkpoints/dreamzero_g1_simple_G1WholebodyXMoveBendPickTeleop-v0_lora/checkpoint-30000 \
        --pretrained-base checkpoints/DreamZero-AgiBot --prompt-cache /data/dreamzero/data/prompt_cache_g1_simple.pt

    # full finetuned checkpoint
    DEBUG=1 python eval_utils/serve_dreamzero_g1_simple.py \
        --model-path checkpoints/g1simple_G1WholebodyLocomotionPickBetweenTablesTeleop-v0_full_mc4nf33_20260503_1010/checkpoint-4000 \
        --prompt-cache /data/dreamzero/data/prompt_cache_g1_simple.pt

    # If the 14B model doesn't fit on one GPU, launch with torchrun (the
    # underlying GrootSimPolicy supports a device_mesh and you can set
    # --world-size 2 or more). For a first smoke test on an A100-80GB, a
    # 14B bf16 + LoRA inference typically fits in ~30 GB.

NOTE: This file is a standalone draft — it does NOT touch the existing
      serve_dreamzero_wan22.py, socket_test_optimized_AR.py, or any other
      server code. Another agent may be working in this repo, so feel free
      to move / rename this file if it conflicts with ongoing work.
"""

from __future__ import annotations

import argparse
import base64
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import datetime

import imageio
import numpy as np
import torch
import torch._dynamo
from einops import rearrange
# Disable torch.compile for inference — the model was compiled for training
# shapes and recompiles endlessly on inference-shaped inputs.
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.cache_size_limit = 256
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from numpy.lib.format import descr_to_dtype, dtype_to_descr
from pydantic import BaseModel
from tianshou.data import Batch

# Repo root on path so `groot.*` imports work regardless of where you cd.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dreamzero.groot.vla.data.schema import EmbodimentTag
from dreamzero.groot.vla.model.n1_5.sim_policy import GrootSimPolicy  # noqa: E402

logger = logging.getLogger("dreamzero-g1-server")

# --------------------------------------------------------------------------- #
#   SIMPLE numpy-in-JSON (de)serialization                                    #
# --------------------------------------------------------------------------- #
#   These mirror `simple/baselines/client.py:numpy_serialize/deserialize` so  #
#   the wire format is identical and SIMPLE's HttpActionClient can talk to    #
#   this server with no changes.                                              #


def numpy_serialize(o: Any) -> dict:
    if isinstance(o, (np.ndarray, np.generic)):
        data = o.data if o.flags["C_CONTIGUOUS"] else o.tobytes()
        return {
            "__numpy__": base64.b64encode(bytes(data)).decode(),
            "dtype": dtype_to_descr(o.dtype),
            "shape": list(o.shape),
        }
    raise TypeError(f"Object of type {type(o).__name__} is not numpy")


def numpy_deserialize(dct: dict) -> Any:
    if "__numpy__" in dct:
        arr = np.frombuffer(
            base64.b64decode(dct["__numpy__"]), descr_to_dtype(dct["dtype"])
        )
        shape = tuple(dct["shape"])
        return arr.reshape(shape) if shape else arr[0]
    return dct


def convert_numpy_in_dict(data, func):
    if isinstance(data, dict):
        if "__numpy__" in data:
            return func(data)
        return {k: convert_numpy_in_dict(v, func) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_numpy_in_dict(v, func) for v in data]
    if isinstance(data, (np.ndarray, np.generic)):
        return func(data)
    return data


# --------------------------------------------------------------------------- #
#   Config — keep in sync with scripts/train/g1_simple_training_lora.sh       #
# --------------------------------------------------------------------------- #

# Must match training image_resolution_{width,height}.
DEFAULT_IMAGE_HEIGHT = 176
DEFAULT_IMAGE_WIDTH = 320

# max_state_dim from training (Psi0-flat 32D state + 12 pad slots).
MODEL_STATE_DIM = 44

# max_action_dim from training. Model outputs (action_horizon=48, action_dim=36).
MODEL_ACTION_DIM = 36

# Video frames per chunk: training subsamples 8 raw frames per chunk at
# stride = action_steps_per_chunk // 8. The client should collect observations
# at this stride during action execution and send all 8 frames per /act call.
VIDEO_FRAMES_PER_CHUNK = 8

# DISABLE_KV_CACHE mode: every /act call is independent. Server only needs the
# current frame as an anchor each call — no streaming chunk follow-up. We
# advertise this to the client via /config so it sends 1 frame per call.
DISABLE_KV_CACHE = os.environ.get("DISABLE_KV_CACHE", "false").lower() == "true"


# --------------------------------------------------------------------------- #
#   Policy wrapper                                                            #
# --------------------------------------------------------------------------- #


class DreamZeroG1SimplePolicy:
    """Thin wrapper around GrootSimPolicy that:

    1. Takes a SIMPLE-style request dict.
    2. Packs it into the Batch format expected by the g1_simple training
       modality config:
           video.egocentric:                              (T, H, W, 3) uint8
           state.joint_position:                           (1, 32)     float64  -> padded
           annotation.language.language_instruction:      str
    3. Runs `policy.lazy_joint_forward_causal(batch)`.
    4. Extracts `action.joint_position` (shape (24, 36)) from the result.
    5. Handles session / KV-cache resets (on `reset=True` in request.history).
    """

    def __init__(
        self,
        groot_policy: GrootSimPolicy,
        image_height: int = DEFAULT_IMAGE_HEIGHT,
        image_width: int = DEFAULT_IMAGE_WIDTH,
        video_output_dir: str | None = None,
        model_path: str | None = None,
    ) -> None:
        self._policy = groot_policy
        self._h = image_height
        self._w = image_width
        self._frame_buffer: list[np.ndarray] = []
        self._is_first_call = True
        self._current_session_id: str | None = None
        # Video logging: accumulate predicted video latents per session and
        # decode+save as MP4 on session reset.
        self._video_output_dir = video_output_dir
        self._video_latents: list[torch.Tensor] = []
        # Bottom-banner metadata burned into every saved frame. Use last
        # two components (run_id / ckpt_step) so the run name is visible.
        if model_path:
            norm = os.path.normpath(model_path).rstrip(os.sep)
            parts = norm.split(os.sep)
            self._model_path_basename = (
                "/".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
            )
        else:
            self._model_path_basename = "?"
        if os.environ.get("DISABLE_KV_CACHE", "false").lower() == "true":
            self._cache_mode = "no_cache"
        elif os.environ.get("ROLLING_KV_CACHE", "false").lower() == "true":
            self._cache_mode = "rolling_cache"
        else:
            self._cache_mode = "vanilla"
        if video_output_dir:
            os.makedirs(video_output_dir, exist_ok=True)
            logger.info("Video logging enabled → %s", video_output_dir)

    def _save_accumulated_video(self) -> None:
        """Decode accumulated video latents and save as MP4.

        Called on session reset so each episode gets one video file showing
        the model's predicted future frames across all /act calls.
        """
        if not self._video_output_dir or not self._video_latents:
            return
        try:
            head = self._policy.trained_model.action_head
            num_frame_per_block = head.num_frame_per_block

            # Drop the first latent of every chunk after the first that carries
            # a re-anchor prepend (T > num_frame_per_block). That re-anchor
            # latent is a fresh VAE-encoding of the current observation and
            # temporally overlaps the previous chunk's last denoised latent —
            # concatenating it would insert a discontinuous single-frame latent
            # into the middle of the joint sequence. The first chunk keeps its
            # prepend: it is the initial KV-cache anchor, no overlap.
            trimmed = [self._video_latents[0]]
            for l in self._video_latents[1:]:
                if l.shape[2] > num_frame_per_block:
                    trimmed.append(l[:, :, 1:])
                else:
                    trimmed.append(l)

            # Map each pixel frame back to its source chunk using the causal
            # VAE attribution: latent 0 of the full sequence contributes 1
            # pixel frame (causal edge), every subsequent latent contributes 4
            # pixel frames. So chunk 0 gets (1 + 4*(T0-1)) pixels and every
            # later chunk gets 4*Tk pixels.
            chunk_pixel_counts: list[int] = []
            for ci, l in enumerate(trimmed):
                t = l.shape[2]
                chunk_pixel_counts.append(1 + 4 * (t - 1) if ci == 0 else 4 * t)
            chunk_ids: list[int] = []
            for ci, n in enumerate(chunk_pixel_counts):
                chunk_ids.extend([ci] * n)

            latents_cat = torch.cat(trimmed, dim=2)
            with torch.no_grad():
                decoded = head.vae.decode(
                    latents_cat,
                    tiled=head.tiled,
                    tile_size=(head.tile_size_height, head.tile_size_width),
                    tile_stride=(head.tile_stride_height, head.tile_stride_width),
                )
            frames = rearrange(decoded, "B C T H W -> B T H W C")[0]
            frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
            assert len(chunk_ids) == len(frames), (
                f"chunk_ids ({len(chunk_ids)}) != frames ({len(frames)})"
            )

            # Overlay "cM fK gG" on every frame (top-left).
            # Bottom-left banner: model basename / cache mode / level_ep extracted
            # from current session_id (e.g. "dreamzero-dwbc-rolling_cache_level1_ep5"
            # → "level1_ep5"). Self-documents which (model, cache, rollout) the
            # mp4 came from, so it's identifiable even when filenames are
            # shortened or files are moved.
            import re
            m = re.search(r"level\d+_ep\d+", self._current_session_id or "")
            level_ep = m.group(0) if m else (self._current_session_id or "?")
            banner_text = f"{self._model_path_basename} | {self._cache_mode} | {level_ep}"

            from PIL import Image, ImageDraw
            annotated = []
            chunk_local_idx: dict[int, int] = {}
            for gi, (frame, cid) in enumerate(zip(frames, chunk_ids)):
                k = chunk_local_idx.get(cid, 0)
                chunk_local_idx[cid] = k + 1
                img = Image.fromarray(frame)
                draw = ImageDraw.Draw(img)
                # top-left chunk/frame indicator
                text = f"c{cid} f{k}  g{gi}"
                bbox = draw.textbbox((4, 4), text)
                draw.rectangle(bbox, fill=(0, 0, 0))
                draw.text((4, 4), text, fill=(255, 255, 255))
                # bottom-left banner: model | cache | level_ep
                W, H = img.size
                bbox2 = draw.textbbox((4, H - 16), banner_text)
                draw.rectangle(bbox2, fill=(0, 0, 0))
                draw.text((4, H - 16), banner_text, fill=(255, 255, 255))
                annotated.append(np.array(img))

            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            sid = self._current_session_id or "unknown"
            n_chunks = len(self._video_latents)
            path = os.path.join(
                self._video_output_dir,
                f"dreamzero-{ts}-{sid}_n{n_chunks}.mp4",
            )
            imageio.mimsave(path, annotated, fps=5, codec="libx264")
            logger.info("[video] saved %d frames (%d chunks) → %s", len(annotated), n_chunks, path)
        except Exception:
            logger.exception("[video] failed to save accumulated video")
        finally:
            self._video_latents = []

    def _maybe_reset(self, session_id: str | None, reset: bool) -> None:
        if reset or (
            session_id is not None and session_id != self._current_session_id
        ):
            # Save predicted video from the previous session before clearing.
            self._save_accumulated_video()
            logger.info(
                "[reset] session %s -> %s (explicit reset=%s)",
                self._current_session_id,
                session_id,
                reset,
            )
            self._frame_buffer = []
            self._is_first_call = True
            self._current_session_id = session_id
            head = getattr(self._policy.trained_model, "action_head", None)
            if head is not None and hasattr(head, "current_start_frame"):
                head.current_start_frame = 0

    def _pack_state(self, states: np.ndarray) -> np.ndarray:
        """Client sends 32D Psi0-flat state as shape (1, 32).

        Do NOT pre-pad to max_state_dim=44 here — the model's internal
        transforms normalise first (using 32D q99 stats), then the
        ConcatTransform pads to max_state_dim. Pre-padding would cause a
        shape mismatch between the 32D normalisation mask and a 44D tensor.
        """
        if states.ndim == 1:
            states = states.reshape(1, -1)
        return states.astype(np.float64)

    def _build_batch(self, request: dict) -> Batch:
        """Convert a SIMPLE RequestMessage dict to a tianshou Batch the
        G1 policy can consume."""
        image_dict = request.get("image", {})
        instruction = request.get("instruction", "") or ""
        state_dict = request.get("state", {}) or {}
        history = request.get("history", {}) or {}

        # Cache reset if needed.
        self._maybe_reset(history.get("session_id"), bool(history.get("reset", False)))

        # Accept observation frames from client.
        # Preferred: client sends (T, H, W, 3) with T = VIDEO_FRAMES_PER_CHUNK
        # Backward compat: client sends (H, W, 3) single frame, accumulated in buffer.
        img = image_dict.get("rgb_head_stereo_left")
        if img is None and image_dict:
            img = next(iter(image_dict.values()))
        if img is None:
            raise HTTPException(400, detail="request.image.rgb_head_stereo_left missing")
        if not isinstance(img, np.ndarray):
            img = np.asarray(img)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        if img.ndim == 4:
            # Client sent (T, H, W, 3) — use directly
            video = img
            if (
                not self._is_first_call
                and not DISABLE_KV_CACHE
                and video.shape[0] != VIDEO_FRAMES_PER_CHUNK
            ):
                # Streaming VAE assumes exactly 4*nfpb=8 NEW frames per /act
                # call (no overlap with prior calls). Reject anything else.
                # In DISABLE_KV_CACHE mode every call is a fresh first-call
                # so we accept any T (the head uses videos[:, :, -1:] as
                # anchor and discards the rest).
                raise HTTPException(
                    400,
                    detail=(
                        f"non-first call must send exactly {VIDEO_FRAMES_PER_CHUNK} "
                        f"new frames; got {video.shape[0]}"
                    ),
                )
        else:
            # Client sent (H, W, 3) — backward compat: accumulate in buffer
            self._frame_buffer.append(img)
            if self._is_first_call:
                video = img[np.newaxis]  # (1, H, W, 3)
            else:
                # Streaming VAE assumes exactly 4*nfpb=8 NEW frames per /act
                # call (no overlap with prior calls). Reject anything else.
                raise HTTPException(
                    400,
                    detail=(
                        f"non-first call must send exactly {VIDEO_FRAMES_PER_CHUNK} "
                        f"new frames; got {img.shape}"
                    ),
                )
                num_frames = min(VIDEO_FRAMES_PER_CHUNK, len(self._frame_buffer))
                if num_frames < VIDEO_FRAMES_PER_CHUNK:
                    logger.warning(
                        "[_build_batch] only %d frames in buffer but model expects %d. "
                        "Client should send %d subsampled frames per /act call.",
                        num_frames, VIDEO_FRAMES_PER_CHUNK, VIDEO_FRAMES_PER_CHUNK,
                    )
                video = np.stack(self._frame_buffer[-num_frames:], axis=0)

        # State: client sends {"states": np.ndarray(1, 32)}.
        states = state_dict.get("states")
        if states is None:
            raise HTTPException(400, detail="request.state.states missing")
        if not isinstance(states, np.ndarray):
            states = np.asarray(states, dtype=np.float64)
        states = self._pack_state(states)

        # Assemble the model batch. Keys MUST match the modality_config_g1_psi0
        # block in groot/vla/configs/data/dreamzero/g1_simple.yaml.
        obs = {
            "video.egocentric":                            video,
            "state.joint_position":                        states,
            "annotation.language.language_instruction":    instruction,
        }
        return Batch(obs=obs)

    def _extract_action(self, result_batch: Batch) -> np.ndarray:
        """Pull the 36D action chunk out of the model's Batch output.

        The train config writes the action under `action.joint_position`
        (see g1_simple.yaml). GrootSimPolicy exposes it as an attribute on
        `result_batch.act` with the same key.
        """
        action_obj = result_batch.act
        for key in dir(action_obj):
            if not key.startswith("action."):
                continue
            if "joint_position" not in key:
                continue
            val = getattr(action_obj, key)
            if isinstance(val, torch.Tensor):
                val = val.detach().cpu().numpy()
            elif not isinstance(val, np.ndarray):
                val = np.asarray(val)
            if val.ndim == 1:
                val = val.reshape(1, -1)
            if val.shape[-1] != MODEL_ACTION_DIM:
                logger.warning(
                    "action width %d != MODEL_ACTION_DIM=%d (key=%s)",
                    val.shape[-1], MODEL_ACTION_DIM, key,
                )
            return val.astype(np.float32)
        raise HTTPException(
            500,
            detail=f"model output did not contain action.*joint_position key; "
                   f"available: {[k for k in dir(action_obj) if k.startswith('action.')]}",
        )

    def infer(self, request: dict) -> np.ndarray:
        batch = self._build_batch(request)
        t0 = time.time()
        with torch.no_grad():
            result_batch, video_pred = self._policy.lazy_joint_forward_causal(batch)
        # Accumulate predicted video latents for later decode+save.
        if video_pred is not None and self._video_output_dir:
            self._video_latents.append(video_pred.detach())
        # DISABLE_KV_CACHE: keep _is_first_call=True so the streaming-frame
        # gate above stays relaxed and downstream sees every call as fresh.
        if self._is_first_call and not DISABLE_KV_CACHE:
            self._is_first_call = False
        action = self._extract_action(result_batch)
        logger.info(
            "[infer] session=%s step=%s chunk=%s latency=%.3fs",
            self._current_session_id,
            request.get("history", {}).get("step_index"),
            tuple(action.shape),
            time.time() - t0,
        )
        return action


# --------------------------------------------------------------------------- #
#   FastAPI app                                                               #
# --------------------------------------------------------------------------- #


class _EmptyModel(BaseModel):
    model_config = {"extra": "allow"}


def make_app(policy: DreamZeroG1SimplePolicy) -> FastAPI:
    app = FastAPI(title="dreamzero-g1-simple")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": "dreamzero-g1-simple"}

    @app.get("/config")
    def config():
        """Return model config so clients can adapt frame collection and action execution."""
        head = policy._policy.trained_model.action_head
        action_horizon = head.action_horizon
        # DISABLE_KV_CACHE mode wants 1 frame per call (anchor only, run as
        # first-call path). Other modes pull the standard streaming chunk.
        frames_per_chunk = 1 if DISABLE_KV_CACHE else VIDEO_FRAMES_PER_CHUNK
        video_stride = action_horizon // VIDEO_FRAMES_PER_CHUNK
        model_config =  {
            "action_horizon": action_horizon,
            "video_frames_per_chunk": frames_per_chunk,
            "video_stride": video_stride,
            "num_frame_per_block": head.num_frame_per_block,
            "action_dim": MODEL_ACTION_DIM,
        }
        if os.environ.get("ROLLING_KV_CACHE", "false").lower() == "true":
            model_config["rolling_kv_cache"] = True
        if DISABLE_KV_CACHE:
            model_config["disable_kv_cache"] = True
        return model_config

    @app.post("/flush")
    def flush():
        """Save any accumulated video latents from the last session."""
        policy._save_accumulated_video()
        return {"status": "flushed"}

    @app.post("/act")
    async def act(request_body: dict):
        try:
            request = convert_numpy_in_dict(request_body, numpy_deserialize)
            action = policy.infer(request)
            response = {
                "action":     action,
                "err":        0.0,
                "traj_image": np.zeros((1, 1, 3), dtype=np.uint8),
            }
            return JSONResponse(convert_numpy_in_dict(response, numpy_serialize))
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("[/act] unhandled error")
            return JSONResponse(
                convert_numpy_in_dict(
                    {
                        "action":     np.zeros((1, MODEL_ACTION_DIM), dtype=np.float32),
                        "err":        repr(e),
                        "traj_image": np.zeros((1, 1, 3), dtype=np.uint8),
                    },
                    numpy_serialize,
                ),
                status_code=500,
            )

    return app


# --------------------------------------------------------------------------- #
#   Entry point                                                               #
# --------------------------------------------------------------------------- #


def _init_mesh() -> tuple:
    """Initialize distributed process group and device mesh.

    When launched via torchrun (world_size > 1), creates a real multi-GPU mesh
    for tensor-parallel inference. When launched standalone (world_size == 1),
    falls back to a single-GPU setup.

    Returns (device_mesh, rank, signal_group).
    """
    if not dist.is_initialized():
        # Support both torchrun and standalone launch.
        if "RANK" in os.environ:
            dist.init_process_group("nccl")
        else:
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "29500")
            dist.init_process_group("nccl", rank=0, world_size=1)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    logger.info("Rank %d/%d (PID %d) on cuda:%d", rank, world_size, os.getpid(), rank)

    mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=("ip",))

    # Gloo-based signal group for lightweight rank-0 → worker signaling
    # (does not block NCCL traffic).
    import datetime
    signal_group = dist.new_group(backend="gloo",
                                  timeout=datetime.timedelta(hours=24))
    return mesh, rank, signal_group


def _worker_loop(
    policy: DreamZeroG1SimplePolicy,
    signal_group: dist.ProcessGroup,
) -> None:
    """Non-rank-0 worker: wait for signals and participate in distributed
    forward passes so NCCL collectives don't hang.

    Protocol (mirrors socket_test_optimized_AR.py):
      signal 0 → inference request incoming, participate in forward()
      signal 1 → shutdown
      signal 2 → idle / next-client, loop back
    """
    import pickle

    rank = dist.get_rank()
    logger.info("Rank %d entering worker loop", rank)
    signal_tensor = torch.zeros(1, dtype=torch.int32, device="cpu")

    while True:
        try:
            dist.broadcast(signal_tensor, src=0, group=signal_group)
            sig = signal_tensor.item()

            if sig == 1:
                logger.info("Rank %d received shutdown signal", rank)
                break
            if sig == 2:
                continue

            # sig == 0 → inference request. Receive the observation batch.
            size_tensor = torch.zeros(1, dtype=torch.int64, device="cuda")
            dist.broadcast(size_tensor, src=0)
            data_tensor = torch.zeros(int(size_tensor.item()), dtype=torch.uint8, device="cuda")
            dist.broadcast(data_tensor, src=0)
            obs = pickle.loads(data_tensor.cpu().numpy().tobytes())
            batch = Batch(obs=obs)

            # Participate in the distributed forward pass.
            dist.barrier()
            with torch.no_grad():
                policy._policy.lazy_joint_forward_causal(batch)
            dist.barrier()

        except Exception:
            logger.exception("Worker loop error on rank %d", rank)
            break


def _broadcast_obs_to_workers(obs: dict) -> None:
    """Rank-0 helper: serialize and broadcast the observation dict so all
    worker ranks can participate in the same forward pass."""
    import pickle

    serialized = pickle.dumps(obs)
    size_tensor = torch.tensor([len(serialized)], dtype=torch.int64, device="cuda")
    dist.broadcast(size_tensor, src=0)
    data_tensor = torch.frombuffer(serialized, dtype=torch.uint8).cuda()
    dist.broadcast(data_tensor, src=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="DreamZero G1 SIMPLE HTTP server")
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to the fine-tuned G1 SIMPLE LoRA checkpoint "
             "(e.g. ./checkpoints/dreamzero_g1_simple_mixture_lora).",
    )
    parser.add_argument("--port", type=int, default=22085)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument(
        "--pretrained-base",
        type=str,
        default=None,
        help="Path to the pretrained base checkpoint (e.g. DreamZero-AgiBot) "
             "that the LoRA was fine-tuned on top of. Without this, the LoRA "
             "operates on the raw backbone and pretrained knowledge is lost.",
    )
    parser.add_argument(
        "--image-height", type=int, default=DEFAULT_IMAGE_HEIGHT,
        help=f"Video frame height after resize. Must match training "
             f"image_resolution_height ({DEFAULT_IMAGE_HEIGHT}).",
    )
    parser.add_argument(
        "--image-width", type=int, default=DEFAULT_IMAGE_WIDTH,
        help=f"Video frame width after resize. Must match training "
             f"image_resolution_width ({DEFAULT_IMAGE_WIDTH}).",
    )
    parser.add_argument("--enable-dit-cache", type=bool, default=True,
                        help="Enable DiT KV-cache for faster autoregressive inference.")
    parser.add_argument(
        "--video-output-dir", type=str, default="auto",
        help="Directory to save predicted video MP4s. 'auto' (default) saves "
             "to <model-path>/eval_videos/. Set to 'none' to disable.",
    )
    parser.add_argument(
        "--prompt-cache", type=str, default=None,
        help="Path to precomputed prompt embeddings (.pt). When provided, "
             "the text encoder is skipped for cached prompts and only loaded "
             "on cache miss. Generate with scripts/precompute_text_embeddings.py.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )

    # Match the original server's torch.compile config.
    os.environ["ENABLE_DIT_CACHE"] = "true" if args.enable_dit_cache else "false"
    os.environ.setdefault("ATTENTION_BACKEND", "TE")
    torch._dynamo.config.recompile_limit = 800

    device_mesh, rank, signal_group = _init_mesh()

    logger.info("Loading GrootSimPolicy (embodiment=simple) from %s", args.model_path)
    if args.pretrained_base:
        logger.info("Using pretrained base: %s", args.pretrained_base)
    if args.prompt_cache:
        logger.info("Using prompt cache: %s", args.prompt_cache)
    groot_policy = GrootSimPolicy(
        embodiment_tag=EmbodimentTag("simple"),
        model_path=args.model_path,
        tokenizer_path_override=args.tokenizer_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        device_mesh=device_mesh,
        pretrained_base_path=args.pretrained_base,
        prompt_cache_path=args.prompt_cache,
    )

    # Resolve video output directory
    video_output_dir = args.video_output_dir
    if video_output_dir == "auto":
        video_output_dir = os.path.join(args.model_path, "eval_videos")
    elif video_output_dir == "none":
        video_output_dir = None

    wrapper = DreamZeroG1SimplePolicy(
        groot_policy=groot_policy,
        image_height=args.image_height,
        image_width=args.image_width,
        video_output_dir=video_output_dir,
        model_path=args.model_path,
    )

    if rank == 0:
        # Patch the infer() method to signal workers before each forward pass.
        _original_infer = wrapper.infer

        def _distributed_infer(request: dict) -> np.ndarray:
            signal_tensor = torch.zeros(1, dtype=torch.int32, device="cpu")
            dist.broadcast(signal_tensor, src=0, group=signal_group)

            batch = wrapper._build_batch(request)
            _broadcast_obs_to_workers(batch.obs)

            dist.barrier()
            t0 = time.time()
            with torch.no_grad():
                result_batch, video_pred = wrapper._policy.lazy_joint_forward_causal(batch)
            dist.barrier()

            if video_pred is not None and wrapper._video_output_dir:
                wrapper._video_latents.append(video_pred.detach())
            if wrapper._is_first_call and not DISABLE_KV_CACHE:
                wrapper._is_first_call = False
            action = wrapper._extract_action(result_batch)
            logger.info(
                "[infer] session=%s step=%s chunk=%s latency=%.3fs",
                wrapper._current_session_id,
                request.get("history", {}).get("step_index"),
                tuple(action.shape),
                time.time() - t0,
            )
            return action

        if dist.get_world_size() > 1:
            wrapper.infer = _distributed_infer

        app = make_app(wrapper)
        logger.info("Rank 0: starting HTTP server on %s:%d", args.host, args.port)
        uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level.lower())
    else:
        logger.info("Rank %d: entering worker loop for distributed inference", rank)
        _worker_loop(wrapper, signal_group)


if __name__ == "__main__":
    main()
