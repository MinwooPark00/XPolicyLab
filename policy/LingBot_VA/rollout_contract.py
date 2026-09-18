"""Pure helpers for keeping LingBot rollout actions and video history aligned."""

from __future__ import annotations

import numpy as np


def cache_action_chunk(
    raw_action: np.ndarray,
    *,
    skipped_steps: int,
    executed_steps: int,
    action_per_frame: int,
    include_skipped_prefix: bool,
) -> np.ndarray:
    """Return only action frames represented by the real observations cached.

    The backend emits ``[channel, latent_frame, action_per_frame]``.  On the
    first call its leading frame is the synthetic zero-action condition, so it
    must accompany the initial image in the first cache update.  Predictions
    after the executed horizon are discarded and must not enter the KV cache.
    """
    raw = np.asarray(raw_action)
    if raw.ndim != 3:
        raise ValueError(f"expected [channel, frame, action] tensor, got {raw.shape}")
    if action_per_frame <= 0 or raw.shape[2] != action_per_frame:
        raise ValueError(
            f"action_per_frame={action_per_frame} disagrees with raw shape {raw.shape}")
    if skipped_steps < 0 or executed_steps <= 0:
        raise ValueError(
            f"invalid skipped/executed steps: {skipped_steps}/{executed_steps}")
    if skipped_steps % action_per_frame or executed_steps % action_per_frame:
        raise ValueError(
            "LingBot cache updates must use complete action/video frames: "
            f"skip={skipped_steps}, execute={executed_steps}, "
            f"action_per_frame={action_per_frame}")

    chronological = raw.transpose(1, 2, 0).reshape(-1, raw.shape[0])
    start = 0 if include_skipped_prefix else skipped_steps
    stop = skipped_steps + executed_steps
    if stop > chronological.shape[0]:
        raise ValueError(
            f"cache span [{start}:{stop}] exceeds {chronological.shape[0]} actions")
    selected = chronological[start:stop]
    frames = selected.shape[0] // action_per_frame
    return selected.reshape(frames, action_per_frame, raw.shape[0]).transpose(2, 0, 1)


def cache_keyframes(
    executed_observations: list,
    latest_observation,
    *,
    executed_steps: int,
    keyframe_stride: int,
) -> list:
    """Select the four real images represented by each action/video frame.

    MHBench records the observation after each action except the final action
    in a returned chunk.  ``latest_observation`` is that missing final image,
    supplied by the next ``update_obs`` immediately before replanning.
    """
    if executed_steps <= 0 or keyframe_stride <= 0:
        raise ValueError(
            f"invalid executed_steps/keyframe_stride: {executed_steps}/{keyframe_stride}")
    if executed_steps % keyframe_stride:
        raise ValueError(
            "executed horizon must be divisible by keyframe stride: "
            f"{executed_steps} % {keyframe_stride} != 0")

    expected = executed_steps // keyframe_stride
    usable = min(len(executed_observations), executed_steps - 1)
    result = [
        executed_observations[index]
        for index in range(keyframe_stride - 1, usable, keyframe_stride)
    ]
    if len(result) < expected and latest_observation is not None:
        result.append(latest_observation)
    if len(result) != expected:
        raise ValueError(
            f"expected {expected} real cache keyframes, collected {len(result)}")
    return result
