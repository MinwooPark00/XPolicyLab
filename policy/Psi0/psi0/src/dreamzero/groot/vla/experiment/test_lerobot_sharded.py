import json
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import decord
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from psi.utils import dreamzero_instantiate as instantiate
from dreamzero.groot.vla.utils.action_args_override_utils import apply_action_overrides


@hydra.main(config_path="../configs", config_name="conf", version_base=None)
def test_padding_and_alignment(cfg):
    cfg = apply_action_overrides(cfg)

    # Derived parameters
    num_latent_per_block: int = cfg.num_frame_per_block    # 2
    action_steps_per_chunk: int = cfg.train_dataset.dataset_kwargs.action_steps_per_chunk
    video_frames_per_chunk: int = num_latent_per_block * 4  # 8 (VAE temporal factor = 4)
    video_stride: int = action_steps_per_chunk // video_frames_per_chunk
    max_chunk_size: int = cfg.max_chunk_size
    max_action_dim: int = cfg.max_action_dim
    max_state_dim: int = cfg.max_state_dim

    max_video_frames = 1 + max_chunk_size * video_frames_per_chunk   # 33
    max_action_steps = max_chunk_size * action_steps_per_chunk        # 320
    max_state_steps  = max_chunk_size                                  # 4 (one per chunk)

    print(f"\n{'='*70}")
    print("DATASET SANITY CHECK")
    print(f"  num_latent_per_block={num_latent_per_block}, "
          f"video_frames_per_chunk={video_frames_per_chunk}, "
          f"video_stride={video_stride}")
    print(f"  action_steps_per_chunk={action_steps_per_chunk}, "
          f"max_chunk_size={max_chunk_size}")
    print(f"  expected video frames : {max_video_frames}  "
          f"(1 obs + {max_chunk_size}X{video_frames_per_chunk})")
    print(f"  expected action steps : {max_action_steps}  "
          f"({max_chunk_size}X{action_steps_per_chunk})")
    print(f"  expected state steps  : {max_state_steps}  "
          f"(1 per chunk, shape=({max_state_steps},{max_state_dim}))")
    print('='*70)

    train_dataset = instantiate(cfg.train_dataset)

    i = 0
    for sample in train_dataset:
        # if i < 2:
        #     i += 1 
        #     continue

        traj_id  = sample.get('_debug_trajectory_id', 'N/A')
        step_idx = sample.get('_debug_step_index', 'N/A')
        print(f"\nSample  episode={traj_id}  start_step={step_idx}")

        # ── 1. Print all keys / shapes ──────────────────────────────────────
        print("\n─── Transformed keys ───────────────────────────────────────────")
        for k in sorted(sample.keys()):
            if k.startswith('_debug'):
                continue
            v = sample[k]
            if isinstance(v, np.ndarray):
                vmin = f"{v.min():.3f}" if np.issubdtype(v.dtype, np.floating) else str(v.min())
                vmax = f"{v.max():.3f}" if np.issubdtype(v.dtype, np.floating) else str(v.max())
                print(f"  {k:40s} shape={str(v.shape):20s} dtype={v.dtype}  [{vmin}, {vmax}]")
            elif hasattr(v, 'shape'):
                print(f"  {k:40s} shape={str(v.shape):20s} dtype={v.dtype}")
            elif isinstance(v, (int, float, bool, np.bool_, np.integer)):
                print(f"  {k:40s} = {v}")
            elif isinstance(v, str):
                print(f"  {k:40s} = '{v[:80]}'")
            elif isinstance(v, list):
                print(f"  {k:40s} list[{len(v)}] = {str(v[:2])[:60]}...")
            else:
                print(f"  {k:40s} {type(v).__name__}")

        # ── 2. Chunk alignment on transformed data ───────────────────────────
        print("\n─── Chunk alignment (transformed) ──────────────────────────────")
        # action = sample.get('action')
        # state  = sample.get('state')
        # NOTE turn off all transforms to test this
        action = sample.get("action.joint_position")
        state  = sample.get("state.joint_position")
        checks: list[tuple[str, bool]] = []

        n_chunks_action = None
        n_chunks_state  = None

        chunk_size = action.shape[0] // action_steps_per_chunk 

        if action is not None:
            n = action.shape[0]
            ok = (n % action_steps_per_chunk == 0)
            n_chunks_action = n // action_steps_per_chunk
            checks.append(("action steps divisible by chunk size", ok))
            mark = "✓" if ok else "✗"
            print(f"  action : {n:4d} steps / {action_steps_per_chunk} = "
                  f"{n_chunks_action} chunks  {mark}")

        if state is not None:
            n = state.shape[0]
            n_chunks_state = n  # one state token per chunk
            ok = (n == n_chunks_action)
            checks.append(("state tokens == n_chunks_action", ok))
            mark = "✓" if ok else "✗"
            print(f"  state  : {n:4d} tokens  expected={n_chunks_action}  {mark}")

        if n_chunks_action is not None and n_chunks_state is not None:
            ok = (n_chunks_action == n_chunks_state)
            checks.append(("action chunks == state tokens", ok))
            mark = "✓" if ok else "✗"
            print(f"  action chunks == state tokens: "
                  f"{n_chunks_action} == {n_chunks_state}  {mark}")

        # ── 3. Raw data: read parquet + mp4 directly ────────────────────────
        print("\n─── Raw data (parquet + mp4) ────────────────────────────────────")

        if traj_id == 'N/A' or step_idx == 'N/A':
            print("  [no debug info — skipping raw inspection]")
            raw = None
        else:
            raw = _load_raw(
                cfg, int(traj_id), int(step_idx),
                action_steps_per_chunk, action.shape[0] // action_steps_per_chunk if action is not None else max_chunk_size,
                video_frames_per_chunk, video_stride,
            )
            print("=" * 50)
            print(f" traj len={raw['traj_len']} steps, step_idx={step_idx}, traj_idx={traj_id}")
            print("=" * 50)

            if raw is None:
                print("  [_load_raw returned None]")
            else:
                for k, v in raw.items():
                    if isinstance(v, np.ndarray):
                        print(f"    {k}: shape={v.shape}")

                raw_video  = raw.get("video")   # (T, H, W, C)
                raw_action = raw.get("action")  # (total_steps, action_dim)

                # Video frame count check
                if raw_video is not None:
                    n_vf = raw_video.shape[0]
                    n_chunks_video = (n_vf - 1) / video_frames_per_chunk
                    ok_int = (n_vf - 1) % video_frames_per_chunk == 0
                    ok_exp = n_vf == (1 + n_chunks_action * video_frames_per_chunk)
                    checks.append(("raw video frame count", ok_int and ok_exp))
                    print(f"\n  video frames: {n_vf}  "
                          f"expected={max_video_frames}  "
                          f"{'✓' if ok_exp else '✗'}")
                    print(f"  (frames-1)/{video_frames_per_chunk} = {n_chunks_video:.2f} chunks  "
                          f"{'✓' if ok_int else '✗ NOT INTEGER'}")

                    if n_chunks_action is not None:
                        ok = int(n_chunks_video) == n_chunks_action
                        checks.append(("video/action chunk count match", ok))
                        mark = "✓" if ok else "✗"
                        print(f"  video chunks == action chunks: "
                              f"{int(n_chunks_video)} == {n_chunks_action}  {mark}")

                    # Temporal layout
                    print(f"\n  Temporal layout (video_stride={video_stride}):")
                    print(f"    Frame  0        → t=step+0  (observation)")
                    actual_n_chunks = int(n_chunks_video)
                    for c in range(actual_n_chunks):
                        f0 = 1 + c * video_frames_per_chunk
                        f1 = f0 + video_frames_per_chunk - 1
                        t0 = (c * video_frames_per_chunk + 1) * video_stride
                        t1 = (c + 1) * video_frames_per_chunk * video_stride
                        a0 = c * action_steps_per_chunk
                        a1 = (c + 1) * action_steps_per_chunk - 1
                        print(f"    Chunk {c}: video frames [{f0:2d},{f1:2d}] "
                              f"→ t=step+[{t0},{t1}]  |  "
                              f"action steps [{a0},{a1}] "
                              f"→ t=step+[{a0},{a1}]")

                    # Visualize
                    out = _visualize_video(
                        raw_video, int(traj_id), int(step_idx), video_frames_per_chunk
                    )
                    print(f"\n  Video grid saved → {out}")

                if raw_action is not None:
                    out = _visualize_action(
                        raw_action, int(traj_id), int(step_idx), action_steps_per_chunk
                    )
                    print(f"  Action plot saved → {out}")

                # ── Compare raw data with dataset output ─────────────────────
                print("\n─── Raw vs dataset comparison ───────────────────────────────────")
                # With IdentityTransform, dataset yields per-modality keys unchanged.
                # With simple_pad_freeze_action=False, no freeze padding is applied.
                # dataset action key (IdentityTransform preserves original key names)
                dataset_action = sample.get("action.joint_position")
                dataset_state  = sample.get("state.joint_position")

                if dataset_action is not None and raw_action is not None:
                    if dataset_action.shape == raw_action.shape:
                        match = np.allclose(dataset_action, raw_action, rtol=1e-5, atol=1e-5)
                        mark = "✓" if match else "✗"
                        print(f"  {mark} action.joint_position  shape={dataset_action.shape}  match={match}")
                        if not match:
                            diff = np.abs(dataset_action.astype(np.float64) - raw_action.astype(np.float64))
                            first_bad = int(np.argmax(diff.max(axis=-1)))
                            print(f"      max_diff={diff.max():.6f}  mean_diff={diff.mean():.6f}"
                                  f"  first_mismatch_row={first_bad}")
                            # Check: mismatch should start exactly at the padding boundary.
                            # pad_start is the first action row that overshoots the trajectory
                            # (i.e. step_idx + row >= traj_len).
                            raw_traj_len = raw.get("traj_len")
                            if raw_traj_len is not None:
                                pad_start = raw_traj_len - int(step_idx)
                                at_boundary = (first_bad == pad_start)
                                bmark = "✓" if at_boundary else "✗"
                                print(f"      {bmark} mismatch at pad boundary: "
                                      f"first_bad={first_bad}  pad_start={pad_start}"
                                      f"  (step_idx={int(step_idx)}, traj_len={raw_traj_len})")
                                checks.append(("mismatch starts at pad boundary", at_boundary))
                        checks.append(("action matches raw parquet", match))
                    else:
                        print(f"  ✗ action shape mismatch: dataset={dataset_action.shape}  raw={raw_action.shape}")
                        checks.append(("action matches raw parquet", False))
                else:
                    missing = []
                    if dataset_action is None:
                        missing.append("dataset action (IdentityTransform not active?)")
                    if raw_action is None:
                        missing.append("raw action")
                    print(f"  [skipped — missing: {', '.join(missing)}]")

                # State: dataset yields one value per chunk at anchor indices
                # (step_idx, step_idx+chunk_size, ...). raw["state"] is sequential,
                # so we extract the same anchor rows for comparison.
                raw_state = raw.get("state")
                if dataset_state is not None and raw_state is not None:
                    n_chunks = dataset_state.shape[0]
                    anchor_rows = np.arange(n_chunks) * action_steps_per_chunk
                    raw_state_at_anchors = raw_state[
                        np.minimum(anchor_rows, len(raw_state) - 1)
                    ]
                    if dataset_state.shape == raw_state_at_anchors.shape:
                        smatch = np.allclose(
                            dataset_state, raw_state_at_anchors, rtol=1e-5, atol=1e-5
                        )
                        smark = "✓" if smatch else "✗"
                        print(f"  {smark} state.joint_position   shape={dataset_state.shape}  "
                              f"(compared at chunk anchors)  match={smatch}")
                        if not smatch:
                            sdiff = np.abs(
                                dataset_state.astype(np.float64)
                                - raw_state_at_anchors.astype(np.float64)
                            )
                            sfirst_bad = int(np.argmax(sdiff.max(axis=-1)))
                            print(f"      max_diff={sdiff.max():.6f}  mean_diff={sdiff.mean():.6f}"
                                  f"  first_mismatch_chunk={sfirst_bad}")
                            raw_traj_len = raw.get("traj_len")
                            if raw_traj_len is not None:
                                # first chunk whose anchor overshoots the trajectory
                                remaining = raw_traj_len - int(step_idx)
                                pad_start_chunk = (
                                    remaining + action_steps_per_chunk - 1
                                ) // action_steps_per_chunk
                                at_boundary = (sfirst_bad >= pad_start_chunk)
                                bmark = "✓" if at_boundary else "✗"
                                print(f"      {bmark} mismatch at pad boundary: "
                                      f"first_bad_chunk={sfirst_bad}  pad_start_chunk={pad_start_chunk}"
                                      f"  (step_idx={int(step_idx)}, traj_len={raw_traj_len})")
                                checks.append(("state mismatch starts at pad boundary", at_boundary))
                        checks.append(("state matches raw parquet (at anchors)", smatch))
                    else:
                        print(f"  ✗ state shape mismatch: "
                              f"dataset={dataset_state.shape}  raw_anchors={raw_state_at_anchors.shape}")
                        checks.append(("state matches raw parquet (at anchors)", False))
                else:
                    missing = []
                    if dataset_state is None:
                        missing.append("dataset state (IdentityTransform not active?)")
                    if raw_state is None:
                        missing.append("raw state")
                    print(f"  [skipped state — missing: {', '.join(missing)}]")

                # Video: dataset yields (T, H, W, C) uint8 frames loaded via decord.
                # _load_raw also uses decord directly on the same file, so pixel values
                # should be identical when the frame indices agree.
                # The pad boundary here is the first frame index that overshoots traj_len.
                dataset_video = sample.get("video.egocentric")
                raw_video = raw.get("video")
                if dataset_video is not None and raw_video is not None:
                    if dataset_video.shape == raw_video.shape:
                        vmatch = np.array_equal(dataset_video, raw_video)
                        vmark = "✓" if vmatch else "✗"
                        print(f"  {vmark} video.egocentric       shape={dataset_video.shape}  match={vmatch}")
                        if not vmatch:
                            # Per-frame max pixel difference
                            vdiff = np.abs(
                                dataset_video.astype(np.int32) - raw_video.astype(np.int32)
                            )  # (T, H, W, C)
                            per_frame_max = vdiff.max(axis=(1, 2, 3))  # (T,)
                            vfirst_bad = int(np.argmax(per_frame_max > 0))
                            print(f"      max_pixel_diff={vdiff.max()}  "
                                  f"first_mismatch_frame={vfirst_bad}  "
                                  f"(per-frame max: {per_frame_max.tolist()})")
                            mismatch_out = _visualize_mismatched_frames(
                                dataset_video, raw_video, per_frame_max,
                                int(traj_id), int(step_idx), video_stride,
                            )
                            print(f"  Mismatch grid saved → {mismatch_out}")
                            raw_traj_len = raw.get("traj_len")
                            if raw_traj_len is not None:
                                # Frame 0 is obs (step_idx), frames 1.. are prediction.
                                # The k-th prediction frame lands at step_idx + k*video_stride.
                                # pad_start_frame: first frame index (0-based in T) where that
                                # trajectory position >= traj_len.
                                remaining = raw_traj_len - int(step_idx)
                                # frame k (k>=1) is at trajectory offset k * video_stride
                                pad_start_frame = (
                                    (remaining + video_stride - 1) // video_stride
                                )  # first k where k*video_stride >= remaining
                                at_boundary = (vfirst_bad >= pad_start_frame)
                                bmark = "✓" if at_boundary else "✗"
                                print(f"      {bmark} mismatch at pad boundary: "
                                      f"first_bad_frame={vfirst_bad}  pad_start_frame={pad_start_frame}"
                                      f"  (step_idx={int(step_idx)}, traj_len={raw_traj_len},"
                                      f" video_stride={video_stride})")
                                checks.append(("video mismatch starts at pad boundary", at_boundary))
                        checks.append(("video matches raw decode", vmatch))
                    else:
                        print(f"  ✗ video shape mismatch: "
                              f"dataset={dataset_video.shape}  raw={raw_video.shape}")
                        checks.append(("video matches raw decode", False))
                else:
                    missing = []
                    if dataset_video is None:
                        missing.append("dataset video (IdentityTransform not active?)")
                    if raw_video is None:
                        missing.append("raw video")
                    print(f"  [skipped video — missing: {', '.join(missing)}]")

                # ── Temporal alignment at trajectory boundary ────────────────
                traj_len = raw.get("traj_len")
                if traj_len is not None:
                    s = int(step_idx)
                    print(f"\n─── Temporal alignment  (traj_len={traj_len}, start={s}) ─────────")
                    print(f"  {'Chunk':5} | {'action t=[first,last]':24} | {'clamped':7} "
                          f"| {'video  t=[first,last]':24} | {'clamped':7} | ok")
                    print(f"  {'-'*5}-+-{'-'*24}-+-{'-'*7}-+-{'-'*24}-+-{'-'*7}-+----")
                    all_aligned = True
                    for c in range(n_chunks_action if n_chunks_action is not None else max_chunk_size):
                        # Action: consecutive steps, stride 1
                        a_first = s + c * action_steps_per_chunk
                        a_last  = s + (c + 1) * action_steps_per_chunk - 1
                        a_clamp = a_last >= traj_len

                        # Video: stride = video_stride; first frame of chunk at
                        #   s + c*vfpc*vs + 1*vs, last at s + (c+1)*vfpc*vs
                        v_first = s + c * video_frames_per_chunk * video_stride + video_stride
                        v_last  = s + (c + 1) * video_frames_per_chunk * video_stride
                        v_clamp = v_last >= traj_len

                        # They should agree: both clamped or both in-bounds
                        ok = (a_clamp == v_clamp)
                        if not ok:
                            all_aligned = False
                        mark = "✓" if ok else "✗"
                        print(f"  {c:5} | t=[{a_first:5d},{a_last:5d}] "
                              f"| {'CLAMP' if a_clamp else 'ok   ':5} "
                              f"| t=[{v_first:5d},{v_last:5d}] "
                              f"| {'CLAMP' if v_clamp else 'ok   ':5} | {mark}")
                    checks.append(("action/video clamping chunk-consistent", all_aligned))

        # ── 4. Summary ───────────────────────────────────────────────────────
        print(f"\n─── Check summary ───────────────────────────────────────────────")
        all_passed = True
        for name, ok in checks:
            mark = "✓" if ok else "✗"
            print(f"  {mark}  {name}")
            if not ok:
                all_passed = False
        print()
        print("ALL CHECKS PASSED ✓" if all_passed else "SOME CHECKS FAILED ✗")
        print('='*70)
        i+=1
        # break


def _load_raw(cfg, traj_id: int, step_idx: int,
              action_steps_per_chunk: int, max_chunk_size: int,
              video_frames_per_chunk: int, video_stride: int) -> dict | None:
    """Read action/state from parquet and video frames from mp4 directly."""
    dataset_root = Path(list(cfg.train_dataset.mixture_spec[0].dataset_path.simple)[0])
    info = json.loads((dataset_root / "meta" / "info.json").read_text())
    chunks_size: int = info["chunks_size"]
    episode_chunk = traj_id // chunks_size

    # ── parquet ──────────────────────────────────────────────────────────────
    parquet_path = dataset_root / info["data_path"].format(
        episode_chunk=episode_chunk, episode_index=traj_id
    )
    df = pd.read_parquet(parquet_path)
    traj_len = len(df)
    print(f"  Parquet: {parquet_path.name}  rows={traj_len}")

    # Collect all action steps across chunks
    total_steps = max_chunk_size * action_steps_per_chunk
    action_rows = []
    state_rows = []
    for i in range(total_steps):
        row_idx = min(step_idx + i, traj_len - 1)
        action_rows.append(np.array(df.iloc[row_idx]["action"], dtype=np.float32))
        state_rows.append(np.array(df.iloc[row_idx]["observation.state"], dtype=np.float32))

    raw_action = np.stack(action_rows)   # (total_steps, action_dim)
    raw_state  = np.stack(state_rows)    # (total_steps, state_dim)

    # ── mp4 ──────────────────────────────────────────────────────────────────
    video_key = list(info["video_path"].split("{video_key}")[0].split("/")[-1:])[0]
    # determine the actual video key from features
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    assert video_keys, "No video features found in info.json"
    vk = video_keys[0]
    video_path = dataset_root / info["video_path"].format(
        episode_chunk=episode_chunk, episode_index=traj_id, video_key=vk
    )
    print(f"  Video:  {video_path.name}  key={vk}")

    vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
    # Initial frame (observation) + video_frames_per_chunk frames per chunk
    frame_indices_raw = [step_idx]  # observation frame
    for c in range(max_chunk_size):
        for f in range(1, video_frames_per_chunk + 1):
            fi = step_idx + c * video_frames_per_chunk * video_stride + f * video_stride
            frame_indices_raw.append(min(fi, len(vr) - 1))

    frames = vr.get_batch(frame_indices_raw).asnumpy()   # (T, H, W, C)

    language_col = "annotation.language.language_instruction"
    language = df[language_col].iloc[step_idx] if language_col in df.columns else None

    print(f"  raw_action : {raw_action.shape}  raw_state : {raw_state.shape}  "
          f"video_frames : {frames.shape}")
    if language is not None:
        print(f"  language   : {language}")

    return {"action": raw_action, "state": raw_state, "video": frames, "traj_len": traj_len}


def _visualize_video(
    frames: np.ndarray,
    traj_id: int,
    step_idx: int,
    video_frames_per_chunk: int,
) -> str:
    """Save all video frames as a labelled grid PNG."""
    n_frames = len(frames)
    n_cols = video_frames_per_chunk + 1  # one extra column for the obs frame
    n_rows = (n_frames + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2.2))
    axes = np.array(axes).reshape(n_rows, n_cols)

    for i in range(n_frames):
        r, c = i // n_cols, i % n_cols
        ax = axes[r, c]
        frame = frames[i]
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        ax.imshow(frame)
        if i == 0:
            label = "obs (t=0)"
            ax.set_title(label, fontsize=7, color='darkgreen', fontweight='bold')
        else:
            chunk = (i - 1) // video_frames_per_chunk
            frame_in_chunk = (i - 1) % video_frames_per_chunk
            ax.set_title(f"c{chunk}/f{frame_in_chunk}", fontsize=7)
        ax.axis('off')

    for i in range(n_frames, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].axis('off')

    plt.suptitle(
        f"Video  ep={traj_id}  start_step={step_idx}  ({n_frames} frames)",
        fontsize=9,
    )
    plt.tight_layout()
    out = f"debug_video_ep{traj_id}_step{step_idx}.png"
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close()
    return out


def _visualize_action(
    actions: np.ndarray,
    traj_id: int,
    step_idx: int,
    action_steps_per_chunk: int,
) -> str:
    """Plot action dimensions across all chunks with chunk-boundary markers."""
    n_steps, n_dims = actions.shape
    n_plot = min(n_dims, 12)

    fig, ax = plt.subplots(figsize=(14, 4))
    for d in range(n_plot):
        ax.plot(actions[:, d], alpha=0.75, lw=1.0, label=f"dim{d}")

    n_chunks = n_steps // action_steps_per_chunk
    for c in range(1, n_chunks + 1):
        ax.axvline(
            c * action_steps_per_chunk - 0.5,
            color='red', linestyle='--', alpha=0.6, lw=0.9,
            label='chunk boundary' if c == 1 else None,
        )

    ax.set_xlabel("Action step")
    ax.set_ylabel("Value (raw)")
    ax.set_title(
        f"Action  ep={traj_id}  start_step={step_idx}  "
        f"({n_steps} steps, {n_chunks} chunks X {action_steps_per_chunk})"
    )
    ax.legend(fontsize=6, ncol=7, loc='upper right')

    out = f"debug_action_ep{traj_id}_step{step_idx}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=100, bbox_inches='tight')
    plt.close()
    return out


def _visualize_mismatched_frames(
    dataset_frames: np.ndarray,
    raw_frames: np.ndarray,
    per_frame_max: np.ndarray,
    traj_id: int,
    step_idx: int,
    video_stride: int,
) -> str:
    """Save a side-by-side grid of mismatched video frames (dataset | raw | diff).

    Each column is one mismatched frame.  The column title shows the frame index
    and the absolute trajectory step in large bold text.  Row labels are burned
    into the image corners so they survive axis('off').
    """
    bad_idx = np.where(per_frame_max > 0)[0]
    n_bad = len(bad_idx)
    if n_bad == 0:
        return ""

    n_rows, n_cols = 3, n_bad
    col_w, row_h = 4.0, 3.5
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * col_w, n_rows * row_h),
                             squeeze=False)

    row_tags = ["dataset", "raw", "diff ×8"]

    for col, k in enumerate(bad_idx):
        traj_step = step_idx + int(k) * video_stride

        df_frame  = dataset_frames[k]
        rw_frame  = raw_frames[k]
        diff_vis  = np.clip(
            np.abs(df_frame.astype(np.int32) - rw_frame.astype(np.int32)) * 8,
            0, 255,
        ).astype(np.uint8)

        for row, (img, tag) in enumerate(zip([df_frame, rw_frame, diff_vis], row_tags)):
            ax = axes[row, col]
            ax.imshow(img if img.dtype == np.uint8 else img.astype(np.uint8))
            ax.axis("off")

            # Row label burned into top-left corner of each image
            ax.text(
                0.01, 0.97, tag,
                transform=ax.transAxes,
                fontsize=11, fontweight="bold", color="yellow",
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.55),
            )

            # Column title (big font) only on the top row
            if row == 0:
                ax.set_title(
                    f"frame {k}   t = {traj_step}",
                    fontsize=16, fontweight="bold", pad=6,
                )

    plt.suptitle(
        f"Video mismatch  ep={traj_id}  start={step_idx}"
        f"   {n_bad}/{len(per_frame_max)} frames differ"
        f" (if diffs happen at padding frames, its fine.) ",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    out = f"debug_mismatch_video_ep{traj_id}_step{step_idx}.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    plt.close()
    return out


if __name__ == "__main__":
    # a smoke test to test action/state/video padding/truncation and chunk alignment
    test_padding_and_alignment()
