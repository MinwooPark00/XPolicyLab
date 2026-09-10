#!/usr/bin/env python3
"""Final verdict: run the DP-comparable diag2 probe on every A/B arm.

Uses the SAME probe construction as diag2_action_parity.py (80 val frames,
8 per episode, roll within an episode) so the numbers sit on the same scale as
DP's 184.8%. Honors each checkpoint's recorded `fusion_bypass`. Read-only.
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import h5py, numpy as np, torch
from torch import nn

XPL = Path(os.environ.get("MHBENCH_XPL") or Path(__file__).resolve().parents[2])
for p in (str(XPL.parent), str(XPL)):
    if p not in sys.path:
        sys.path.insert(0, p)
from XPolicyLab.policy.GauDP.gaudp.policy import GauDPPolicy  # noqa: E402
from XPolicyLab.policy.GauDP.gaudp.dataset import _resize_chw  # noqa: E402

JT = [k for r in (0, 1) for k in range(r * 35, r * 35 + 31)]
GAU = XPL / "policy/GauDP"
DATA = GAU / "data/mhbench-handover-unitree_g1x2_centralized-joint.hdf5"
FEAT = GAU / "checkpoints/mhbench-handover_easy-handover_easy-ee-0/gaussian/features.hdf5"


def rms(x):
    x = np.asarray(x, np.float64)
    return float(np.sqrt((x ** 2).mean())) if x.size else float("nan")


def evaluate(ckpt: Path, f, g, probes, dev, batch=8, seed=1234):
    payload = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    cfg = dict(payload["config"])
    bypass = bool(payload.get("fusion_bypass", False))
    pol = GauDPPolicy(**cfg, gaussian_encoder=nn.Identity())
    pol.load_state_dict(payload["state_dict"], strict=False)
    if bypass:
        pol.gaussian_fusion.forward = lambda gau, img: img
    pol.to(dev).eval().requires_grad_(False)
    H, NA = pol.horizon, pol.n_action_steps

    def cond(images, state, gauss):
        B, T, V, C, Hh, Ww = images.shape
        st = pol.normalizer.normalize_state(state)
        fused = pol.gaussian_fusion(gauss, images)
        enc = pol.obs_encoder(fused.reshape(B * T, V, 3, Hh, Ww), st.reshape(B * T, 86))
        return enc.reshape(B, T * pol.obs_encoder.output_dim)

    def sample(c, noise):
        tr = noise.clone()
        gen = torch.Generator(device=dev).manual_seed(seed)
        pol.noise_scheduler.set_timesteps(pol.num_inference_steps, device=dev)
        for t in pol.noise_scheduler.timesteps:
            tr = pol.noise_scheduler.step(pol.diffusion(tr, t, global_cond=c), t, tr,
                                          generator=gen).prev_sample
        return pol.normalizer.unnormalize_action(tr)[:, :NA]

    acts = {k: [] for k in ("base", "roll_rgb", "roll_gauss", "roll_state")}
    GT, COS = [], []
    for bi in range(0, len(probes), batch):
        ts = probes[bi:bi + batch]
        ims, gau, sts, gts = [], [], [], []
        for t in ts:
            im = [_resize_chw(torch.from_numpy(np.asarray(f[f"rgb_{c}"][t], np.uint8))
                              .permute(2, 0, 1).float().div_(255.)) for c in (0, 1)]
            ims.append(torch.stack(im))
            gau.append(torch.from_numpy(np.asarray(g["gaussian_features"][t])).float())
            sts.append(torch.from_numpy(np.asarray(f["state"][t], np.float32)))
            gts.append(np.stack([np.asarray(f["action"][t + k], np.float32) for k in range(NA)]))
        images = torch.stack(ims)[:, None].to(dev)
        gauss = torch.stack(gau)[:, None].to(dev)
        state = torch.stack(sts)[:, None].to(dev)
        B = images.shape[0]
        roll = max(1, B // 2)
        noise = torch.randn((B, H, 70), device=dev,
                            generator=torch.Generator(device=dev).manual_seed(seed))
        base_c = cond(images, state, gauss)
        COS.append(base_c[:, :1024].float().cpu().numpy())
        for name, c in (("base", base_c),
                        ("roll_rgb", cond(torch.roll(images, roll, 0), state, gauss)),
                        ("roll_gauss", cond(images, state, torch.roll(gauss, roll, 0))),
                        ("roll_state", cond(images, torch.roll(state, roll, 0), gauss))):
            acts[name].append(sample(c, noise).float().cpu().numpy())
        GT.append(np.stack(gts))
    GT = np.concatenate(GT)
    A = {k: np.concatenate(v) for k, v in acts.items()}
    V = np.concatenate(COS)
    n = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    iu = np.triu_indices(len(n), 1)
    motion = rms(GT[:, :, JT] - GT[:, :1, JT])
    return {
        "checkpoint": str(ckpt), "epoch": payload.get("epoch"), "arm": payload.get("arm"),
        "fusion_bypass": bypass, "ema": bool(payload.get("ema", False)),
        "probes": len(probes),
        "gt_chunk_motion_rad": motion,
        "open_loop_error_rad": rms(A["base"][:, :, JT] - GT[:, :, JT]),
        "roll_rgb_rad": rms(A["roll_rgb"][:, :, JT] - A["base"][:, :, JT]),
        "roll_gauss_rad": rms(A["roll_gauss"][:, :, JT] - A["base"][:, :, JT]),
        "roll_state_rad": rms(A["roll_state"][:, :, JT] - A["base"][:, :, JT]),
        "roll_rgb_pct": rms(A["roll_rgb"][:, :, JT] - A["base"][:, :, JT]) / motion * 100,
        "roll_gauss_pct": rms(A["roll_gauss"][:, :, JT] - A["base"][:, :, JT]) / motion * 100,
        "roll_state_pct": rms(A["roll_state"][:, :, JT] - A["base"][:, :, JT]) / motion * 100,
        "visual_pairwise_cos": float((n @ n.T)[iu].mean()),
        "visual_l2": float(np.linalg.norm(V, axis=1).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A,B,C,D")
    ap.add_argument("--which", default="last,best")
    ap.add_argument("--root", type=Path, required=True, help="dir holding <ARM>/last.ckpt")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--per-episode", type=int, default=8)
    args = ap.parse_args()
    dev = torch.device("cuda")
    f = h5py.File(str(DATA), "r"); g = h5py.File(str(FEAT), "r")
    ends = np.asarray(f["episode_ends"], np.int64); starts = np.concatenate(([0], ends[:-1]))
    val = np.flatnonzero(np.asarray(f["val_mask"]))
    probes = [int(t) for ep in val for t in
              np.linspace(int(starts[ep]) + 20, int(ends[ep]) - 8 - 6 - 2, args.per_episode).astype(int)]
    out = []
    for arm in args.arms.split(","):
        for which in args.which.split(","):
            ck = args.root / arm / f"{which}.ckpt"
            if not ck.is_file():
                print(f"[skip] {ck} missing", flush=True); continue
            r = evaluate(ck, f, g, probes, dev)
            r["label"] = f"{arm}/{which}"
            out.append(r)
            print(json.dumps(r), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print("\n=== DP-comparable scale (roll within episode, 80 val frames) ===")
    print(f"{'arm':10} {'ep':>4} {'openloop':>9} {'roll_rgb':>9} {'roll_gau':>9} {'roll_state':>11} {'viscos':>8}")
    for r in out:
        print(f"{r['label']:10} {str(r['epoch']):>4} {r['open_loop_error_rad']:>9.5f} "
              f"{r['roll_rgb_pct']:>8.1f}% {r['roll_gauss_pct']:>8.1f}% {r['roll_state_pct']:>10.1f}% "
              f"{r['visual_pairwise_cos']:>8.4f}")
    print("  reference: DP roll_rgb 184.8% viscos 0.203 | shipped GauDP best 34.4% viscos 0.989")


if __name__ == "__main__":
    main()
