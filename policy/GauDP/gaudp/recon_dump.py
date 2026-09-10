"""Keyframe selection and PNG dumping for NoPoSplat reconstruction eval.

`eval_gaussian.py` already renders every validation frame -- `reconstruction_loss`
splats the predicted Gaussians and keeps only the RGB/depth scalars. This module
is what turns a handful of those renders back into something you can look at.

Only a handful, deliberately. NoPoSplat is per-timestep feed-forward: it reads
the two ego views of one frame and predicts that frame's Gaussians, with no
temporal accumulation, so skipping frames costs nothing in the quality of the
frames that are kept. Rendering all ~31k frames of a task would cost hours and
tens of GB of PNGs to show the same thing 500 times per episode.

Episode starts alone are not enough either: every MHBench episode begins from
the same reset pose (identical camera baseline and depth extent across
episodes), so the first frame of 60 episodes is close to one frame of
information -- and it is the frame with no contact and no occlusion, which is
exactly what a two-view reconstruction handles best. Hence fractions through
the episode rather than a prefix of it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

DEFAULT_FRACTIONS = (0.0, 0.25, 0.5, 0.75)

COLUMNS = ("rgb (gt)", "rgb (recon)", "depth (gt)", "depth (recon)")
"""Column order of a dumped grid. Rows are the context views, in camera order."""

_LABEL_HEIGHT = 14


@dataclass(frozen=True)
class KeyFrame:
    """One frame chosen for dumping, addressed inside a `GaussianFrameDataset`."""

    position: int
    """Index into the dataset (not the global frame index of the HDF5 file)."""
    episode_id: int
    frame: int
    """Frame offset inside its episode."""
    length: int
    """Episode length, so `frame/length` recovers the fraction that chose it."""

    @property
    def name(self) -> str:
        return f"ep{self.episode_id:04d}_f{self.frame:05d}"


def parse_fractions(value: str | Iterable[float]) -> tuple[float, ...]:
    """Parse `'0,0.25,0.5,0.75'` into sorted, de-duplicated fractions in [0, 1)."""
    if isinstance(value, str):
        parts = [part for part in value.replace(",", " ").split() if part]
    else:
        parts = list(value)
    if not parts:
        raise ValueError("at least one fraction is required")
    fractions = []
    for part in parts:
        fraction = float(part)
        if not 0.0 <= fraction < 1.0:
            raise ValueError(f"fraction must be in [0, 1), got {fraction}")
        fractions.append(fraction)
    return tuple(sorted(set(fractions)))


def select_keyframes(
    episode_ranges: Sequence[tuple[int, int]],
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    episode_ids: Sequence[int] | None = None,
) -> list[KeyFrame]:
    """Pick `fractions` through each episode, as dataset positions.

    `episode_ranges` are the global `(start, end)` pairs the dataset selected,
    in the order it concatenated them; positions are the running offset into
    that concatenation, which is what indexing the dataset expects. Two
    fractions that land on the same frame of a short episode yield one keyframe.
    """
    keyframes: list[KeyFrame] = []
    offset = 0
    for order, (start, end) in enumerate(episode_ranges):
        length = int(end) - int(start)
        if length <= 0:
            continue
        episode_id = order if episode_ids is None else int(episode_ids[order])
        seen: set[int] = set()
        for fraction in fractions:
            frame = min(int(fraction * length), length - 1)
            if frame in seen:
                continue
            seen.add(frame)
            keyframes.append(KeyFrame(offset + frame, episode_id, frame, length))
        offset += length
    return keyframes


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """`[3,H,W]` float in [0, 1] -> `[H,W,3]` uint8."""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 3 or array.shape[0] != 3:
        raise ValueError(f"expected [3,H,W], got {array.shape}")
    return (np.clip(array, 0.0, 1.0).transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)


def _depth_to_uint8(depth: np.ndarray, near: float, far: float) -> np.ndarray:
    """`[H,W]` metric depth -> `[H,W,3]` grey, near bright, invalid black.

    Ground truth and render share one `near`/`far`, so the two depth columns are
    directly comparable; without that the render's own range would hide a scale
    error. Invalid ground-truth pixels get 0, which the valid range never
    reaches (it starts at 20).
    """
    array = np.asarray(depth, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"expected [H,W], got {array.shape}")
    valid = np.isfinite(array) & (array > 0)
    span = max(float(far) - float(near), 1e-6)
    normalized = np.clip((array - float(near)) / span, 0.0, 1.0)
    grey = np.where(valid, 20.0 + (1.0 - normalized) * 235.0, 0.0)
    return np.repeat(grey.astype(np.uint8)[:, :, None], 3, axis=2)


def _label_strip(width: int, tiles: int) -> np.ndarray:
    strip = np.zeros((_LABEL_HEIGHT, width, 3), dtype=np.uint8)
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return strip
    canvas = Image.fromarray(strip)
    draw = ImageDraw.Draw(canvas)
    tile_width = width // max(1, tiles)
    for index, title in enumerate(COLUMNS[:tiles]):
        draw.text((index * tile_width + 4, 2), title, fill=(220, 220, 220))
    return np.asarray(canvas)


def render_grid(
    *,
    images: np.ndarray,
    depth: np.ndarray,
    rendered_rgb: np.ndarray,
    rendered_depth: np.ndarray,
    near: float,
    far: float,
    labels: bool = True,
) -> np.ndarray:
    """One frame's `[V,4]` grid of tiles as a single `[H,W,3]` uint8 image."""
    views = images.shape[0]
    rows = [
        np.hstack(
            (
                _to_uint8(images[view]),
                _to_uint8(rendered_rgb[view]),
                _depth_to_uint8(depth[view], near, far),
                _depth_to_uint8(rendered_depth[view], near, far),
            )
        )
        for view in range(views)
    ]
    grid = np.vstack(rows)
    if labels:
        grid = np.vstack((_label_strip(grid.shape[1], len(COLUMNS)), grid))
    return grid


def frame_psnr(rendered_rgb: np.ndarray, images: np.ndarray) -> tuple[float, float]:
    """`(rgb_mse, psnr)` for one frame, over all of its views."""
    error = np.asarray(rendered_rgb, dtype=np.float64) - np.asarray(images, dtype=np.float64)
    mse = float(np.mean(error**2))
    return mse, float(-10.0 * np.log10(max(mse, 1e-10)))


def save_keyframe(
    directory: Path,
    keyframe: KeyFrame,
    *,
    images: np.ndarray,
    depth: np.ndarray,
    rendered_rgb: np.ndarray,
    rendered_depth: np.ndarray,
    near: float,
    far: float,
    labels: bool = True,
) -> dict:
    """Write `<name>.png` and return the manifest record describing it."""
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - Pillow ships with the env
        raise RuntimeError("Pillow is required to dump reconstruction images") from error

    directory.mkdir(parents=True, exist_ok=True)
    grid = render_grid(
        images=images,
        depth=depth,
        rendered_rgb=rendered_rgb,
        rendered_depth=rendered_depth,
        near=near,
        far=far,
        labels=labels,
    )
    path = directory / f"{keyframe.name}.png"
    Image.fromarray(grid).save(path)
    mse, psnr = frame_psnr(rendered_rgb, images)
    return {
        **asdict(keyframe),
        "fraction": keyframe.frame / max(1, keyframe.length),
        "near": float(near),
        "far": float(far),
        "rgb_mse": mse,
        "psnr": psnr,
        "file": path.name,
    }


def write_manifest(directory: Path, records: Sequence[dict]) -> Path:
    """Record what was dumped, so the worst frames can be found without eyes."""
    path = directory / "manifest.jsonl"
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path
