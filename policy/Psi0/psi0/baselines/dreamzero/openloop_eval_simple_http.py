"""Open-loop evaluation for the DreamZero G1 policy served over HTTP.

This is the HTTP-client counterpart of ``examples/simple/openloop_eval.py``.
Instead of loading a Psi0 checkpoint and running inference locally, it talks to
the DreamZero G1 SIMPLE server (``serve_dreamzero_g1_simple.py``) over the
SIMPLE ``/act`` protocol: for every sampled frame it sends the current
observation (single RGB frame + 32D raw state + language instruction) and
parses the predicted action chunk out of the response.

Ground-truth actions are read straight from the DreamZero LeRobot dataset
(``action`` column, 36D, raw units) — the same data the server was trained on.
The server's model normalises the state and de-normalises its action output
internally, so both the state we send and the action we receive are in raw
units and no client-side (de)normalisation is required. We then compute the
per-modality L1 error between the predicted and ground-truth action chunks and
plot it exactly like the reference script.

Because the server keeps a streaming KV-cache (a non-first ``/act`` call expects
exactly 8 new frames), every request here carries ``history.reset=True`` so each
frame is evaluated independently as a fresh first call (single-frame anchor).

Usage:
    .venv-dreamzero/bin/python baselines/dreamzero/openloop_eval_simple_http.py \
        --host 127.0.0.1 --port 8014 \
        --dataset-path /hfm/data/simple-latest/dreamzero/G1WholebodyLocomotionPickBetweenTablesTeleop-v0 \
        --eps-idx 0 --stride 32 --output-dir baselines/dreamzero/openloop_out
"""

from __future__ import annotations

import argparse
import base64
import json
import pathlib
import time
import urllib.request
import uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import imageio.v3 as iio
from numpy.lib.format import descr_to_dtype, dtype_to_descr
from tqdm.auto import tqdm

# --------------------------------------------------------------------------- #
#   DreamZero G1 36-dim action layout (authoritative source:                  #
#   simple/baselines/dreamzero.py). NOTE: dims 31..35 DIFFER from the Psi0    #
#   layout in examples/simple/openloop_eval.py — do not reuse those splits.   #
#       [0:14]  hand_joints                                                   #
#       [14:28] arm_joints                                                    #
#       [28]    torsor_roll   (waist roll)                                    #
#       [29]    torsor_pitch  (waist pitch)                                   #
#       [30]    torsor_yaw    (waist yaw)                                     #
#       [31]    base_height   (target base height)                            #
#       [32]    vx                                                           #
#       [33]    vy                                                           #
#       [34]    torso_vyaw    (turning flag)                                  #
#       [35]    target_yaw                                                    #
# --------------------------------------------------------------------------- #
LABELS = [
    "hand_joints",   # [0:14]
    "arm_joints",    # [14:28]
    "torsor_roll",   # [28:29]
    "torsor_pitch",  # [29:30]
    "torsor_yaw",    # [30:31]
    "base_height",   # [31:32]
    "vx",            # [32:33]
    "vy",            # [33:34]
    "torso_vyaw",    # [34:35]
    "target_yaw",    # [35:36]
]
# Boundaries that carve the 36-dim vector into the 10 LABELS above. Used for
# both the printed summary and the plotted curves so they stay consistent.
SPLITS = [14, 28, 29, 30, 31, 32, 33, 34, 35]

VIDEO_KEY = "observation.images.egocentric"


# --------------------------------------------------------------------------- #
#   SIMPLE numpy-in-JSON (de)serialization — mirrors the server's wire format #
#   (serve_dreamzero_g1_simple.py:numpy_serialize / numpy_deserialize).       #
# --------------------------------------------------------------------------- #
def numpy_serialize(o):
    if isinstance(o, (np.ndarray, np.generic)):
        data = o.data if o.flags["C_CONTIGUOUS"] else o.tobytes()
        return {
            "__numpy__": base64.b64encode(bytes(data)).decode(),
            "dtype": dtype_to_descr(o.dtype),
            "shape": list(o.shape),
        }
    raise TypeError(f"Object of type {type(o).__name__} is not numpy")


def numpy_deserialize(dct):
    if "__numpy__" in dct:
        arr = np.frombuffer(base64.b64decode(dct["__numpy__"]), descr_to_dtype(dct["dtype"]))
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
#   HTTP helpers                                                              #
# --------------------------------------------------------------------------- #
def get_config(host: str, port: int) -> dict:
    url = f"http://{host}:{port}/config"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode())


def request_action(
    host: str,
    port: int,
    frame: np.ndarray,
    state: np.ndarray,
    instruction: str,
    session_id: str,
    episode_index: int,
    step_index: int,
) -> np.ndarray:
    """Send one observation to /act and return the predicted action chunk.

    ``history.reset=True`` makes the server drop its KV-cache and treat this
    call as a fresh first call, so a single anchor frame is accepted and the
    prediction depends only on the current observation (open-loop).
    """
    request = {
        "image": {"rgb_head_stereo_left": frame},           # (H, W, 3) uint8
        "instruction": instruction,
        "history": {
            "session_id": session_id,
            "episode_index": int(episode_index),
            "step_index": int(step_index),
            "reset": True,
        },
        "state": {"states": state.astype(np.float64).reshape(1, -1)},  # (1, 32)
        "condition": {},
        "gt_action": [],
        "dataset_name": "dreamzero_g1_simple",
        "timestamp": str(time.time()),
    }
    payload = json.dumps(convert_numpy_in_dict(request, numpy_serialize)).encode()
    url = f"http://{host}:{port}/act"
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read().decode())

    body = convert_numpy_in_dict(body, numpy_deserialize)
    err = body.get("err", 0.0)
    if err not in (0.0, 0, None) and not (isinstance(err, float) and err == 0.0):
        raise RuntimeError(f"server returned err={err!r}")
    action = np.asarray(body["action"], dtype=np.float32)
    if action.ndim == 1:
        action = action.reshape(1, -1)
    return action  # (Tp, 36)


# --------------------------------------------------------------------------- #
#   Dataset helpers                                                           #
# --------------------------------------------------------------------------- #
def load_meta(dataset_root: pathlib.Path) -> dict:
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    tasks: dict[int, str] = {}
    tasks_path = dataset_root / "meta" / "tasks.jsonl"
    if tasks_path.exists():
        for line in tasks_path.read_text().splitlines():
            line = line.strip()
            if line:
                rec = json.loads(line)
                tasks[int(rec["task_index"])] = rec["task"]
    return {"info": info, "tasks": tasks}


def load_episode(dataset_root: pathlib.Path, info: dict, ep_idx: int):
    chunks_size = int(info.get("chunks_size", 1000))
    chunk_id = ep_idx // chunks_size
    data_fmt = info["data_path"]
    video_fmt = info["video_path"]

    parquet_path = dataset_root / data_fmt.format(episode_chunk=chunk_id, episode_index=ep_idx)
    df = pq.read_table(parquet_path).to_pandas()

    states = np.stack(df["observation.state"].to_numpy()).astype(np.float32)   # (N, 32)
    actions = np.stack(df["action"].to_numpy()).astype(np.float32)             # (N, 36)
    task_index = int(df["task_index"].iloc[0]) if "task_index" in df.columns else 0

    video_path = dataset_root / video_fmt.format(
        episode_chunk=chunk_id, video_key=VIDEO_KEY, episode_index=ep_idx
    )
    frames = iio.imread(video_path)  # (N, H, W, 3) uint8
    return states, actions, frames, task_index


# --------------------------------------------------------------------------- #
#   Main                                                                      #
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description="Open-loop HTTP eval for DreamZero G1 server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8014)
    p.add_argument(
        "--dataset-path",
        type=pathlib.Path,
        default=pathlib.Path(
            "/hfm/data/simple-latest/dreamzero/G1WholebodyLocomotionPickBetweenTablesTeleop-v0"
        ),
        help="Root of the DreamZero LeRobot dataset (matches the served checkpoint).",
    )
    p.add_argument("--eps-idx", type=int, default=None, help="Episode index (random if unset)")
    p.add_argument("--stride", type=int, default=32, help="Frame stride within the episode")
    p.add_argument("--instruction", type=str, default=None,
                   help="Override the language instruction (defaults to the episode's task).")
    p.add_argument("--output-dir", type=str, default="baselines/dreamzero/openloop_out")
    return p.parse_args()


def main():
    args = parse_args()
    dataset_root = args.dataset_path.resolve()

    meta = load_meta(dataset_root)
    info = meta["info"]
    total_episodes = int(info["total_episodes"])

    eps_idx = args.eps_idx if args.eps_idx is not None else np.random.randint(0, total_episodes)
    print(f"Server: http://{args.host}:{args.port}   config={get_config(args.host, args.port)}")
    print(f"Dataset: {dataset_root}")
    print(f"Episode {eps_idx} / {total_episodes}")

    states, actions, frames, task_index = load_episode(dataset_root, info, eps_idx)
    n_frames = min(len(states), len(actions), len(frames))
    instruction = args.instruction or meta["tasks"].get(task_index, "")
    print(f"Frames: {n_frames}  |  instruction: {instruction!r}")

    session_id = f"openloop-{uuid.uuid4().hex[:8]}-ep{eps_idx}"

    # ------------------------------------------------------------------
    # Eval loop — one independent /act request per sampled frame.
    # ------------------------------------------------------------------
    per_frame_errors = []
    for i in tqdm(range(0, n_frames, args.stride), desc="Evaluating frames", unit="frame"):
        pred = request_action(
            host=args.host, port=args.port,
            frame=frames[i], state=states[i], instruction=instruction,
            session_id=session_id, episode_index=eps_idx, step_index=i,
        )  # (Tp, 36)

        # Ground-truth chunk: the future actions starting at frame i, matched
        # to the predicted horizon and truncated at the episode boundary.
        horizon = min(pred.shape[0], n_frames - i)
        gt_chunk = actions[i:i + horizon]                 # (L, 36)
        pred_chunk = pred[:horizon]                        # (L, 36)

        error_l1 = np.abs(pred_chunk - gt_chunk)           # (L, 36)
        per_frame_errors.append(error_l1.mean(axis=0))     # (36,)

    per_frame_errors = np.stack(per_frame_errors, axis=0)  # (T, 36)
    mean_error = per_frame_errors.mean(axis=0)             # (36,)

    # ------------------------------------------------------------------
    # Print per-modality summary: mean L1 absolute error over the group's dims.
    # ------------------------------------------------------------------
    print("\n---------------------------\n")
    for i, seg in enumerate(np.split(mean_error, SPLITS)):
        print(f"err_l1_{LABELS[i]} {seg.shape}:  {seg.mean():.6f}")

    # ------------------------------------------------------------------
    # Plot per-step error curves (5 panels, same grouping as reference).
    # Each curve is the mean L1 absolute error across the group's dims.
    # ------------------------------------------------------------------
    error_groups  = np.split(per_frame_errors, SPLITS, axis=-1)
    per_label_mae = [g.mean(axis=-1) for g in error_groups]
    curve_map     = dict(zip(LABELS, per_label_mae))

    plot_groups = [
        ("hand_joints + arm_joints (rad)",   ["hand_joints", "arm_joints"]),
        ("torso rpy (rad)",                  ["torsor_roll", "torsor_pitch", "torsor_yaw"]),
        ("base_height (m)",                  ["base_height"]),
        ("vx + vy + torso_vyaw (m/s, rad)",  ["vx", "vy", "torso_vyaw"]),
        ("target_yaw (rad)",                 ["target_yaw"]),
    ]

    fig, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=True)
    for ax, (title, keys) in zip(axes, plot_groups):
        for key in keys:
            if key in curve_map:
                curve = curve_map[key]
                (line,) = ax.plot(curve, label=key)
                # Horizontal line at the curve's average, colour-matched to it.
                avg = float(np.mean(curve))
                ax.axhline(
                    avg,
                    color=line.get_color(),
                    linestyle="--",
                    linewidth=1,
                    alpha=0.7,
                    label=f"{key} avg={avg:.4f}",
                )
        ax.set_title(title)
        ax.set_ylabel("Mean L1 abs error")
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel(f"Sample step in episode (stride={args.stride})")
    plt.suptitle(f"DreamZero G1 (HTTP)  |  Episode {eps_idx}", y=1.01)
    plt.tight_layout()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"openloop_eval_http_eps{eps_idx}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
