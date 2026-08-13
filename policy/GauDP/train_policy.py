#!/usr/bin/env python3
"""Train the centralized GauDP diffusion policy with a frozen Gaussian encoder."""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path

_XPL_ROOT = Path(__file__).resolve().parents[2]
if str(_XPL_ROOT) not in sys.path:
    sys.path.insert(0, str(_XPL_ROOT))

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def _save(path: Path, policy: GauDPPolicy, optimizer, scheduler, epoch, metrics, gaussian_checkpoint: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        policy_checkpoint_payload(
            policy,
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict(),
            epoch=epoch,
            metrics=metrics,
            gaussian_checkpoint=gaussian_checkpoint.name,
        ),
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gaussian", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--obs-steps", type=int, default=3)
    parser.add_argument("--action-steps", type=int, default=6)
    parser.add_argument("--inference-steps", type=int, default=100)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=os.environ.get("GAUDP_WANDB_MODE", "offline"),
        help="W&B mode; JSONL is always written regardless of this setting",
    )
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "MHBench-GauDP"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", default="", help="comma-separated W&B tags")
    parser.add_argument("--debug", action="store_true", help="run one train and validation batch")
    args = parser.parse_args()

    global np, torch, DataLoader
    global GauDPSequenceDataset, freeze_gaussian_encoder, load_gaussian_checkpoint
    global GauDPPolicy, policy_checkpoint_payload
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from XPolicyLab.policy.GauDP.gaudp.dataset import GauDPSequenceDataset
    from XPolicyLab.policy.GauDP.gaudp.gaussian import freeze_gaussian_encoder, load_gaussian_checkpoint
    from XPolicyLab.policy.GauDP.gaudp.experiment_logger import ExperimentLogger, parse_wandb_tags
    from XPolicyLab.policy.GauDP.gaudp.policy import GauDPPolicy, policy_checkpoint_payload

    _seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("GauDP policy training requires CUDA for the frozen NoPoSplat encoder")
    train_data = GauDPSequenceDataset(args.data, True, args.horizon, args.obs_steps)
    val_data = GauDPSequenceDataset(args.data, False, args.horizon, args.obs_steps)
    policy = GauDPPolicy(
        num_views=len(train_data.camera_order),
        horizon=args.horizon,
        n_obs_steps=args.obs_steps,
        n_action_steps=args.action_steps,
        num_inference_steps=args.inference_steps,
    )
    missing, unexpected = load_gaussian_checkpoint(policy.gaussian_encoder, args.gaussian, strict=False)
    if missing or unexpected:
        print(f"[GauDP] Gaussian load: missing={len(missing)}, unexpected={len(unexpected)}")
    freeze_gaussian_encoder(policy.gaussian_encoder)
    states, actions = train_data.normalization_arrays()
    policy.normalizer.fit(states, actions)
    policy.to(device)

    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.95, 0.999), weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, 1 if args.debug else args.epochs))
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_data, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    best = math.inf
    epochs = 1 if args.debug else args.epochs
    global_step = 0
    run_name = args.wandb_run_name or f"{args.output.parent.name}-policy"
    with ExperimentLogger(
        args.output,
        config=vars(args),
        wandb_mode=args.wandb_mode,
        wandb_project=args.wandb_project,
        wandb_run_name=run_name,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_tags=parse_wandb_tags(args.wandb_tags),
    ) as logger:
        for epoch in range(epochs):
            policy.train()
            train_total, train_count = 0.0, 0
            for batch in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss = policy.compute_loss(_to_device(batch, device))
                loss.backward()
                if any(parameter.grad is not None for parameter in policy.gaussian_encoder.parameters()):
                    raise RuntimeError("frozen Gaussian encoder unexpectedly received gradients")
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                train_total += float(loss.detach())
                train_count += 1
                global_step += 1
                if args.debug:
                    break
            scheduler.step()

            policy.eval()
            val_total, val_count = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    val_total += float(policy.compute_loss(_to_device(batch, device)))
                    val_count += 1
                    if args.debug:
                        break
            metrics = {
                "epoch": epoch,
                "lr": scheduler.get_last_lr()[0],
                "train/loss": train_total / max(1, train_count),
                "val/loss": val_total / max(1, val_count),
            }
            _save(args.output / "last.ckpt", policy, optimizer, scheduler, epoch, metrics, args.gaussian)
            if metrics["val/loss"] < best:
                best = metrics["val/loss"]
                _save(args.output / "best.ckpt", policy, optimizer, scheduler, epoch, metrics, args.gaussian)
            logger.log(metrics, step=global_step)
            print(f"[GauDP][policy] step={global_step} {metrics}")


if __name__ == "__main__":
    main()
