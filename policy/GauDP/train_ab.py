#!/usr/bin/env python3
"""GauDP vision-collapse A/B. Standalone: imports the library, edits nothing.

Arms (see --arm):
  A  baseline, identical to the shipped recipe (control)
  B  DP-aligned recipe: EMA + lr 3e-4 + 500-step warmup + normalizer range_eps
     + best selected on val action x0 MAE instead of epsilon MSE
  C  B + proprioception noise during training (breaks the state->action shortcut)
  D  A + fusion bypass: the ResNet sees raw RGB (DP-equivalent inside GauDP)

Readout is the vision-dependence probe logged every epoch, not rollout success.
"""
from __future__ import annotations

import argparse, copy, json, math, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

# policy/GauDP/train_ab.py -> XPolicyLab. MHBENCH_XPL overrides for an out-of-tree copy.
XPL = Path(os.environ.get("MHBENCH_XPL") or Path(__file__).resolve().parents[2])
for p in (str(XPL.parent), str(XPL)):
    if p not in sys.path:
        sys.path.insert(0, p)

from XPolicyLab.policy.GauDP.gaudp.dataset import GauDPSequenceDataset  # noqa: E402
from XPolicyLab.policy.GauDP.gaudp.policy import GauDPPolicy, policy_checkpoint_payload  # noqa: E402
from XPolicyLab.policy.GauDP.gaudp.experiment_logger import ExperimentLogger, parse_wandb_tags  # noqa: E402

JT = [k for r in (0, 1) for k in range(r * 35, r * 35 + 31)]

ARMS = {
    "A": dict(lr=1e-4, ema=False, warmup=0, range_eps=0.0, state_noise=0.0, bypass=False,
              best_on="val/loss"),
    "B": dict(lr=3e-4, ema=True, warmup=500, range_eps=1e-4, state_noise=0.0, bypass=False,
              best_on="val/action/x0_clipped_mae"),
    "C": dict(lr=3e-4, ema=True, warmup=500, range_eps=1e-4, state_noise=0.1, bypass=False,
              best_on="val/action/x0_clipped_mae"),
    "D": dict(lr=1e-4, ema=False, warmup=0, range_eps=0.0, state_noise=0.0, bypass=True,
              best_on="val/loss"),
    # sigma sweep on top of the B recipe. B (sigma=0) lands at rgb/state 0.054 and
    # C (sigma=0.1) at 0.338; DP sits at 0.184, so the useful range is between them.
    "S02": dict(lr=3e-4, ema=True, warmup=500, range_eps=1e-4, state_noise=0.02, bypass=False,
                best_on="val/action/x0_clipped_mae"),
    "S04": dict(lr=3e-4, ema=True, warmup=500, range_eps=1e-4, state_noise=0.04, bypass=False,
                best_on="val/action/x0_clipped_mae"),
    "S07": dict(lr=3e-4, ema=True, warmup=500, range_eps=1e-4, state_noise=0.07, bypass=False,
                best_on="val/action/x0_clipped_mae"),
    # S04 with the gaussian branch bypassed: differs from S04 in bypass alone, so a
    # matching closed-loop score says the gaussians contribute nothing and sigma is
    # what fixed the blindness.
    "DS04": dict(lr=3e-4, ema=True, warmup=500, range_eps=1e-4, state_noise=0.04, bypass=True,
                 best_on="val/action/x0_clipped_mae"),
    # S02 with the gaussian branch bypassed. Differs from S02 in bypass alone, so a
    # matching closed-loop score says the gaussians contribute nothing.
    "DS02": dict(lr=3e-4, ema=True, warmup=500, range_eps=1e-4, state_noise=0.02, bypass=True,
                 best_on="val/action/x0_clipped_mae"),
}

# Closed-loop on handover (50 episodes, grasp_lift / transfer / place):
#   sigma 0     4/50   0  0      sigma 0.04  21/50   0  0
#   sigma 0.02 48/50  16  1      sigma 0.07   9/50   0  0
# DP for reference: 21/50, 7, 0. The peak is sharp and the offline vision probes
# rank it wrong (they preferred 0.04), so re-tune sigma per task by rollout.
DEFAULT_ARM = "S02"


class EMA:
    """diffusion_policy's EMAModel schedule (inv_gamma 1.0, power 0.75, max 0.9999)."""

    def __init__(self, model, inv_gamma=1.0, power=0.75, max_value=0.9999):
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()
                       if not k.startswith("gaussian_encoder.")}
        self.inv_gamma, self.power, self.max_value = inv_gamma, power, max_value
        self.step = 0

    def decay(self):
        s = max(0, self.step - 1)
        value = 1 - (1 + s / self.inv_gamma) ** -self.power
        return 0.0 if s <= 0 else min(value, self.max_value)

    @torch.no_grad()
    def update(self, model):
        d = self.decay()
        self.step += 1
        for k, v in model.state_dict().items():
            if k not in self.shadow:
                continue
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1 - d)
            else:
                self.shadow[k].copy_(v.detach().float())

    def state(self, reference):
        return {k: self.shadow[k].to(reference[k].dtype) for k in self.shadow}


def apply_range_eps(normalizer, eps: float) -> int:
    """Neutralize near-constant state dims, the way DP's LinearNormalizer does.

    GauDP's min-max maps a dim whose demonstrations span 9e-6 rad onto the full
    [-1,1], a gain of 2.2e5 per radian on what is effectively sensor noise.
    Re-centering those dims on a unit-gain window makes them carry ~0 instead.
    """
    lo, hi = normalizer.state_min, normalizer.state_max
    span = hi - lo
    bad = span < eps
    if not bool(bad.any()):
        return 0
    centre = (hi + lo) * 0.5
    normalizer.state_min = torch.where(bad, centre - 1.0, lo)
    normalizer.state_max = torch.where(bad, centre + 1.0, hi)
    return int(bad.sum())


def build_probe(val_data, device, count=16):
    """A fixed set of val frames, one batch, used for the per-epoch probe."""
    step = max(1, len(val_data) // count)
    items = [val_data[i * step] for i in range(count)]
    out = {}
    for key in ("images", "state", "action", "gaussian_features"):
        # the cache is float16; the fusion runs in the images' dtype, exactly as
        # GauDPPolicy._global_condition casts it.
        out[key] = torch.stack([it[key] for it in items]).to(device=device, dtype=torch.float32)
    return out


@torch.no_grad()
def vision_probe(policy, probe, bypass, full=False, seed=1234):
    """Is the visual conditioning discriminative, and does it move the action?"""
    policy.eval()
    dev = probe["images"].device
    images, state, gauss = probe["images"], probe["state"], probe["gaussian_features"]
    B, T, V, C, H, W = images.shape
    roll = max(1, B // 2)

    def cond(im, ga):
        st = policy.normalizer.normalize_state(state[:, : policy.n_obs_steps])
        fused = policy.gaussian_fusion(ga, im)
        enc = policy.obs_encoder(fused.reshape(B * T, V, 3, H, W), st.reshape(B * T, 86))
        return enc.reshape(B, T * policy.obs_encoder.output_dim)

    base = cond(images, gauss)
    vis = base[:, :1024].float().cpu().numpy()
    n = vis / (np.linalg.norm(vis, axis=1, keepdims=True) + 1e-12)
    iu = np.triu_indices(len(n), 1)
    out = {
        "probe/visual_pairwise_cos": float((n @ n.T)[iu].mean()),
        "probe/visual_l2": float(np.linalg.norm(vis, axis=1).mean()),
        "probe/visual_rel_variation": float(vis.std(0).mean() / (np.abs(vis).mean() + 1e-12)),
    }
    if not full:
        return out

    def sample(c):
        g = torch.Generator(device=dev).manual_seed(seed)
        tr = torch.randn((B, policy.horizon, 70), device=dev, generator=g)
        policy.noise_scheduler.set_timesteps(policy.num_inference_steps, device=dev)
        for t in policy.noise_scheduler.timesteps:
            tr = policy.noise_scheduler.step(
                policy.diffusion(tr, t, global_cond=c), t, tr, generator=g).prev_sample
        return policy.normalizer.unnormalize_action(tr)[:, : policy.n_action_steps]

    a0 = sample(base).float().cpu().numpy()
    a_rgb = sample(cond(torch.roll(images, roll, 0), gauss)).float().cpu().numpy()
    a_gau = sample(cond(images, torch.roll(gauss, roll, 0))).float().cpu().numpy()
    gt = probe["action"][:, : policy.n_action_steps].float().cpu().numpy()
    rms = lambda x: float(np.sqrt((np.asarray(x, np.float64) ** 2).mean()))
    motion = rms(gt[:, :, JT] - gt[:, :1, JT])
    out["probe/roll_rgb_rad"] = rms(a_rgb[:, :, JT] - a0[:, :, JT])
    out["probe/roll_gauss_rad"] = rms(a_gau[:, :, JT] - a0[:, :, JT]) if not bypass else 0.0
    out["probe/gt_chunk_motion_rad"] = motion
    out["probe/roll_rgb_ratio"] = out["probe/roll_rgb_rad"] / max(motion, 1e-9)
    out["probe/roll_gauss_ratio"] = out["probe/roll_gauss_rad"] / max(motion, 1e-9)
    out["probe/open_loop_rad"] = rms(a0[:, :, JT] - gt[:, :, JT])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), default=DEFAULT_ARM,
                    help=f"training recipe (default {DEFAULT_ARM})")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--gaussian", type=Path, required=True)
    ap.add_argument("--gaussian-features", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--obs-steps", type=int, default=1)
    ap.add_argument("--action-steps", type=int, default=6)
    ap.add_argument("--inference-steps", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--full-probe-every", type=int, default=10)
    ap.add_argument("--max-batches", type=int, default=0, help="smoke test: cap train/val batches per epoch")
    ap.add_argument("--wandb-mode", default="online")
    ap.add_argument("--wandb-project", default="MHBench-GauDP")
    args = ap.parse_args()
    cfgA = ARMS[args.arm]
    args.output.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    train_data = GauDPSequenceDataset(args.data, True, args.horizon, args.obs_steps, args.gaussian_features)
    val_data = GauDPSequenceDataset(args.data, False, args.horizon, args.obs_steps, args.gaussian_features)
    policy = GauDPPolicy(
        num_views=len(train_data.camera_order), horizon=args.horizon,
        n_obs_steps=args.obs_steps, n_action_steps=args.action_steps,
        num_inference_steps=args.inference_steps, crop_shape=(216, 288),
        image_norm="imagenet", group_norm_divisor=16, gaussian_encoder=nn.Identity(),
    )
    states, actions = train_data.normalization_arrays()
    policy.normalizer.fit(states, actions)
    neutralized = apply_range_eps(policy.normalizer, cfgA["range_eps"]) if cfgA["range_eps"] else 0
    if cfgA["bypass"]:
        # Keep the module (and its keys) so the checkpoint stays loadable; only
        # the forward path changes, so the ResNet reads raw RGB.
        policy.gaussian_fusion.forward = lambda gau, img: img
    policy.to(device)
    # per-dim raw-space sigma matching `state_noise` in normalized units
    span = (policy.normalizer.state_max - policy.normalizer.state_min).clamp_min(1e-6)
    noise_scale = (cfgA["state_noise"] * span * 0.5).to(device)

    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfgA["lr"], betas=(0.95, 0.999), weight_decay=1e-6)
    # persistent workers: each worker holds the 125 GB feature cache open, so
    # respawning them every epoch would pay that open 150 times over.
    keep = args.num_workers > 0
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              persistent_workers=keep)
    val_loader = DataLoader(val_data, args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True,
                            persistent_workers=keep)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warm = cfgA["warmup"]

    def lr_at(step):
        if warm and step < warm:
            return (step + 1) / warm
        p = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)
    ema = EMA(policy) if cfgA["ema"] else None
    probe = build_probe(val_data, device)

    print(f"[ab][{args.arm}] {json.dumps(cfgA)} neutralized_state_dims={neutralized} "
          f"train_batches={steps_per_epoch} epochs={args.epochs} out={args.output}", flush=True)

    def to_dev(batch):
        return {k: v.to(device, non_blocking=True) for k, v in batch.items()}

    def save(path, metrics, epoch):
        payload = policy_checkpoint_payload(
            policy, epoch=epoch, metrics=metrics, arm=args.arm, arm_config=cfgA,
            gaussian_checkpoint=str(args.gaussian.resolve()),
            camera_order=list(train_data.camera_order),
            fusion_bypass=bool(cfgA["bypass"]),
        )
        if ema is not None:
            payload["state_dict"] = ema.state(payload["state_dict"])
            payload["ema"] = True
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)

    best = math.inf
    best_key = cfgA["best_on"]
    gstep = 0
    with ExperimentLogger(
        args.output, config=vars(args) | {"arm_config": cfgA},
        wandb_mode=args.wandb_mode, wandb_project=args.wandb_project,
        wandb_run_name=f"vision-ab-{args.arm}", wandb_entity=None,
        wandb_group="vision-ab", wandb_tags=parse_wandb_tags(f"vision-ab,arm-{args.arm},handover"),
    ) as logger:
        for epoch in range(args.epochs):
            t0 = time.monotonic()
            policy.train()
            sums, count = {}, 0
            for bi, batch in enumerate(train_loader):
                if args.max_batches and bi >= args.max_batches:
                    break
                batch = to_dev(batch)
                if cfgA["state_noise"]:
                    batch["state"] = batch["state"] + torch.randn_like(batch["state"]) * noise_scale
                optimizer.zero_grad(set_to_none=True)
                loss, m = policy.compute_loss(batch, return_metrics=True)
                loss.backward()
                m["optimization/grad_norm"] = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
                optimizer.step(); scheduler.step()
                if ema is not None:
                    ema.update(policy)
                for k, v in m.items():
                    sums[k] = sums.get(k, 0.0) + float(v)
                count += 1; gstep += 1
            policy.eval()
            vsums, vcount = {}, 0
            with torch.no_grad():
                for bi, batch in enumerate(val_loader):
                    if args.max_batches and bi >= args.max_batches:
                        break
                    _, m = policy.compute_loss(to_dev(batch), return_metrics=True)
                    for k, v in m.items():
                        vsums[k] = vsums.get(k, 0.0) + float(v)
                    vcount += 1
            metrics = {
                "record_type": "epoch", "epoch": epoch, "arm": args.arm,
                "lr": optimizer.param_groups[0]["lr"],
                "performance/epoch_seconds": time.monotonic() - t0,
                "train/loss": sums.get("diffusion/noise_mse", 0.0) / max(1, count),
                "val/loss": vsums.get("diffusion/noise_mse", 0.0) / max(1, vcount),
                **{f"train/{k}": v / max(1, count) for k, v in sums.items()},
                **{f"val/{k}": v / max(1, vcount) for k, v in vsums.items()},
            }
            # EMA weights are what would be served, so probe those.
            if ema is not None:
                backup = copy.deepcopy({k: v for k, v in policy.state_dict().items()
                                        if not k.startswith("gaussian_encoder.")})
                policy.load_state_dict(ema.state(backup), strict=False)
            full = (epoch % args.full_probe_every == 0) or (epoch == args.epochs - 1)
            metrics.update(vision_probe(policy, probe, cfgA["bypass"], full=full))
            if ema is not None:
                policy.load_state_dict(backup, strict=False)
            logger.log(metrics, step=gstep)
            line = (f"[ab][{args.arm}] ep {epoch + 1}/{args.epochs} "
                    f"train {metrics['train/loss']:.5f} val {metrics['val/loss']:.5f} "
                    f"x0mae {metrics.get('val/action/x0_clipped_mae', float('nan')):.5f} "
                    f"viscos {metrics['probe/visual_pairwise_cos']:.4f} "
                    f"visl2 {metrics['probe/visual_l2']:.4f}")
            if full:
                line += (f" ROLLRGB {metrics['probe/roll_rgb_ratio'] * 100:.1f}% "
                         f"rollgau {metrics['probe/roll_gauss_ratio'] * 100:.1f}% "
                         f"openloop {metrics['probe/open_loop_rad']:.5f}")
            line += f" ({metrics['performance/epoch_seconds']:.0f}s)"
            print(line, flush=True)
            if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
                save(args.output / "last.ckpt", metrics, epoch)
            score = metrics.get(best_key, math.inf)
            if score < best:
                best = score
                save(args.output / "best.ckpt", metrics, epoch)
    print(f"[ab][{args.arm}] done, best {best_key}={best:.6f}", flush=True)


if __name__ == "__main__":
    main()
