# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import os
import shutil
import sys
import time
from pathlib import Path
import wandb

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler, Subset
from tqdm import tqdm
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from safetensors.torch import save_file, load_file
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.fsdp import shard_model, ptd_checkpoint_wrapper
from distributed.util import (
    _configure_model, 
    init_distributed, 
    dist_mean, 
    dist_max
)
from einops import rearrange
from modules.utils import (
    load_transformer,
    load_vae,
)
from diffusers.video_processor import VideoProcessor
from modules.lora import apply_lora, merged_state_dict
from utils import (
    init_logger, 
    logger, 
    get_mesh_id, 
    sample_timestep_id,
    data_seq_to_patch,
    warmup_constant_lambda,
    FlowMatchScheduler
)

from dataset import MultiLatentLeRobotDataset
import gc

# Reinitialised because MHBench's action is 35D and the pretrained one is 30D;
# baselines/scripts/prepare_lingbot_init_ckpt.py drops exactly these.
ACTION_WIDTH_KEYS = {
    "action_embedder.weight",
    "action_embedder.bias",
    "action_proj_out.weight",
    "action_proj_out.bias",
}


class Trainer:
    def __init__(self, config):
        # Telemetry must not be able to end a multi-day run: a wandb outage, an
        # expired key or a compute node without egress logs a warning and the
        # training carries on. `enable_wandb` then reads False everywhere below.
        if config.enable_wandb and config.rank == 0:
            try:
                if os.environ.get("WANDB_BASE_URL") and os.environ.get("WANDB_API_KEY"):
                    wandb.login(host=os.environ["WANDB_BASE_URL"],
                                key=os.environ["WANDB_API_KEY"])
                self.wandb = wandb
                self.wandb.init(
                    entity=os.environ.get("WANDB_TEAM_NAME") or None,
                    project=os.getenv("WANDB_PROJECT", "va_robotwin"),
                    config=config,
                    mode="online",
                    name=os.environ.get("WANDB_NAME") or None,
                    id=os.environ.get("WANDB_RUN_ID") or None,
                    resume="allow",
                )
                # A Slurm segment can run beyond its last durable checkpoint.
                # On resume, optimizer_step legitimately moves backwards even
                # though W&B's internal append-only step cannot. Use an
                # explicit x-axis so recovered training is recorded instead
                # of being rejected as non-monotonic.
                self.wandb.define_metric("optimizer_step")
                self.wandb.define_metric("loss_metrics/*", step_metric="optimizer_step")
                self.wandb.define_metric("val/*", step_metric="optimizer_step")
                self.wandb.define_metric("grad_norm", step_metric="optimizer_step")
                self.wandb.define_metric("lr", step_metric="optimizer_step")
                logger.info("WandB logging enabled")
            except Exception as exc:
                logger.warning(f"WandB disabled: {exc}")
                config.enable_wandb = False
        self.step = 0
        self.config = config
        self.device = torch.device(f"cuda:{config.local_rank}")
        self.dtype = config.param_dtype
        self.patch_size = config.patch_size

        # Load models
        logger.info("Loading models...")

        # Load and shard transformer with FSDP
        logger.info("Loading transformer...")

        self.save_dir = Path(config.save_root) / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.lora_rank = int(getattr(config, 'lora_rank', 0) or 0)
        if self.lora_rank and config.world_size > 1:
            raise ValueError(
                "the LoRA arm is the single-card arm; use lora_rank=0 for a sharded "
                "full fine-tune, or run it on one GPU")
        # Full fine-tuning needs FSDP even on one card (16 bytes/param of
        # optimizer state); the LoRA arm keeps the frozen base in bf16 and
        # trains ~80M parameters, which fits without sharding.
        self.use_fsdp = not self.lora_rank

        if getattr(config, 'norm_stat_source', None) == "":
            raise ValueError(
                "no action normalisation: set LINGBOT_NORM_STAT, or point "
                "LINGBOT_VA_DATASET_PATH at a dataset whose meta/ holds "
                "lingbot_norm_stat.json (prepare_lingbot_latents.sbatch writes it). "
                "Training against the placeholder quantiles would learn the wrong scale.")

        base_dir = config.wan22_pretrained_model_name_or_path
        self.resume_state_path = self._resolve_resume_state(config)
        if self.lora_rank:
            # The base never moves under LoRA: the adapters carry the delta and
            # are restored from the training state, so resuming reloads the base.
            transformer_path = os.path.join(base_dir, 'transformer')
        else:
            resume_ckpt = self._latest_checkpoint() if self.resume_state_path else None
            transformer_path = os.path.join(resume_ckpt or base_dir, 'transformer')
        if config.rank == 0:
            logger.info(f"Transformer weights: {transformer_path}")

        self.transformer, loading_info = load_transformer(
            transformer_path,
            torch_dtype=torch.bfloat16 if self.lora_rank else torch.float32,
            torch_device='cpu',
            attn_mode="flex",
            output_loading_info=True,
        )
        missing = set(loading_info.get("missing_keys") or ())
        unexpected = set(loading_info.get("unexpected_keys") or ())
        if not missing <= ACTION_WIDTH_KEYS:
            raise ValueError(
                f"{transformer_path} is not the base this expects: missing "
                f"{sorted(missing - ACTION_WIDTH_KEYS)} beyond the action projections. "
                f"Rebuild it with baselines/scripts/prepare_lingbot_init_ckpt.py.")
        # lingbot-va-base ships a stock-Wan `patch_embedding` Conv3d next to the
        # `patch_embedding_mlp` Linear this model actually uses, so unexpected
        # keys are the release's, not a mismatch. Say what they were and go on.
        if unexpected and config.rank == 0:
            logger.info(f"checkpoint carries {len(unexpected)} unused key(s): {sorted(unexpected)}")
        # diffusers loads on the meta device (low_cpu_mem_usage cannot be
        # turned off while _keep_in_fp32_modules is set), so a key the file
        # does not carry stays a tensor with no data and the first .to(device)
        # raises. Give the two action projections real storage and the
        # initialisation nn.Linear would have given them.
        for module_path in sorted({k.rsplit(".", 1)[0] for k in missing}):
            module = self.transformer.get_submodule(module_path)
            module.to_empty(device="cpu")
            module.reset_parameters()
            if config.rank == 0:
                logger.info(f"reinitialised {module_path}: {tuple(module.weight.shape)}")
        # load_transformer leaves the model where from_pretrained put it (CPU)
        # so the meta parameters above could be materialised first.
        self.transformer = self.transformer.to('cpu')

        if self.lora_rank:
            n_train, n_lora = apply_lora(
                self.transformer,
                rank=self.lora_rank,
                alpha=getattr(config, 'lora_alpha', 16),
                dropout=getattr(config, 'lora_dropout', 0.0),
                train_action_projections=True,
                train_video_projections=getattr(config, 'train_video_projections', False),
                train_action_condition=getattr(config, 'train_action_condition', False),
            )
            if config.rank == 0:
                total = sum(p.numel() for p in self.transformer.parameters())
                logger.info(f"LoRA r{self.lora_rank}: {n_lora/1e6:.1f}M adapter, "
                            f"{n_train/1e6:.1f}M trainable of {total/1e6:.0f}M")
        else:
            self.transformer.requires_grad_(True)

        # Upstream's apply_ac skips the RNG state because its blocks draw no
        # random numbers; LoRA dropout does, and the recompute in backward must
        # redraw the forward's mask or the adapters get another network's gradient.
        preserve_rng = bool(self.lora_rank) and getattr(config, 'lora_dropout', 0.0) > 0
        checkpoint_every = getattr(config, 'activation_checkpoint_every', 1)
        if checkpoint_every < 0:
            raise ValueError("activation_checkpoint_every must be >= 0")
        checkpointed_blocks = 0
        for i, block in enumerate(self.transformer.blocks):
            if checkpoint_every and i % checkpoint_every == 0:
                self.transformer.blocks[i] = ptd_checkpoint_wrapper(
                    block, preserve_rng_state=preserve_rng)
                checkpointed_blocks += 1
        logger.info(
            f"Activation checkpointing: {checkpointed_blocks}/{len(self.transformer.blocks)} "
            f"blocks (every={checkpoint_every}, preserve_rng_state={preserve_rng})")

        if self.use_fsdp:
            logger.info("Setting up FSDP...")
            self.transformer = _configure_model(
                model=self.transformer,
                shard_fn=shard_model,
                param_dtype=self.dtype,
                device=self.device,
                eval_mode=False,
            )
            self.transformer.requires_grad_(True)
        else:
            self.transformer.to(self.device)
        self.transformer.train()

        # Optimizer
        video_projection_params = []
        if self.lora_rank and getattr(config, 'train_video_projections', False):
            video_projection_params = [
                p for name, p in self.transformer.named_parameters()
                if p.requires_grad and name.startswith(
                    ('patch_embedding_mlp.', 'proj_out.'))
            ]
        video_projection_ids = {id(p) for p in video_projection_params}
        main_trainable_params = [
            p for p in self.transformer.parameters()
            if p.requires_grad and id(p) not in video_projection_ids
        ]
        optimizer_groups = [{'params': main_trainable_params,
                             'lr': config.learning_rate}]
        if video_projection_params:
            video_lr = float(config.video_projection_learning_rate)
            optimizer_groups.append({'params': video_projection_params, 'lr': video_lr})
            if config.rank == 0:
                logger.info(
                    f"Video projections: {sum(p.numel() for p in video_projection_params)/1e6:.2f}M "
                    f"trainable at lr={video_lr:.2e}")
        self.optimizer = torch.optim.AdamW(
            optimizer_groups,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
            weight_decay=config.weight_decay,
            fused=True,
            foreach=False,
        )

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, 
            lr_lambda=lambda step: warmup_constant_lambda(step, warmup_steps=config.warmup_steps))

        # Setup dataloaders
        logger.info("Setting up datasets...")
        train_dataset = MultiLatentLeRobotDataset(config=config)
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=config.world_size,
            rank=config.rank,
            shuffle=True,
            seed=42
        ) if config.world_size > 1 else None
        self.train_loader_generator = torch.Generator().manual_seed(42 + config.rank)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=(train_sampler is None), 
            num_workers=config.load_worker,
            sampler=train_sampler,
            # DataLoader otherwise consumes the model's global torch RNG when
            # it creates an iterator. Keep data order independent from the
            # diffusion noise/dropout stream, especially across resumes.
            generator=self.train_loader_generator,
        )

        self.train_scheduler_latent = FlowMatchScheduler(shift=self.config.snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_latent.set_timesteps(1000, training=True)
        self.train_scheduler_action = FlowMatchScheduler(shift=self.config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_action.set_timesteps(1000, training=True)

        # A held-out pass every validation_interval steps, in-process -- not
        # validate_lingbot_checkpoint.py, which reloads a checkpoint from disk
        # in a separate process and only ever runs once per SLURM segment, so
        # it cannot give a mid-run curve. 0 (default) disables this entirely.
        self.validation_interval = int(getattr(config, 'validation_interval', 0) or 0)
        self.validation_samples = int(getattr(config, 'validation_samples', 3) or 3)
        self.val_loader = None
        self.val_loader_iter = None
        if self.validation_interval:
            val_dataset_path = os.environ.get("LINGBOT_VA_VAL_DATASET_PATH") or (
                config.dataset_path + "_val" if getattr(config, 'dataset_path', "") else "")
            if val_dataset_path and os.path.isdir(val_dataset_path):
                from easydict import EasyDict
                val_config = EasyDict(dict(config))
                val_config.dataset_path = val_dataset_path
                val_config.empty_emb_path = os.path.join(val_dataset_path, "empty_emb.pt")
                val_config.cfg_prob = 0.0
                logger.info(f"Loading validation dataset from {val_dataset_path} "
                            f"(every {self.validation_interval} steps)")
                val_dataset = MultiLatentLeRobotDataset(config=val_config)
                # The flattened validation tree is ordered by task and role.
                # Taking its first N whole episodes would measure only the
                # first role of the first task (N=3 in the current run).
                # Spread the fixed diagnostic over the complete held-out set.
                count = min(self.validation_samples, len(val_dataset))
                if count:
                    indices = ([0] if count == 1 else [
                        round(i * (len(val_dataset) - 1) / (count - 1))
                        for i in range(count)
                    ])
                    val_dataset = Subset(val_dataset, indices)
                    logger.info(f"Validation episode indices: {indices}")
                self.val_loader = DataLoader(
                    val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
            else:
                logger.warning(
                    f"LINGBOT_VALIDATION_INTERVAL set but no val dataset at "
                    f"{val_dataset_path!r} -- periodic validation disabled")
                self.validation_interval = 0

        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
        self.train_loader_iter = None
        self._checked_adapter_grads = False
        self._logged_input_images = False
        if self.resume_state_path is not None:
            self._load_training_state(self.resume_state_path)
    
    def _get_next_batch(self):
        """Get next batch from iterator, reset if epoch is finished."""
        if self.train_loader_iter is None:
            self.train_loader_iter = iter(self.train_loader)
        
        try:
            batch = next(self.train_loader_iter)
        except StopIteration:
            # Reset sampler and iterator when epoch finishes
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(self.train_loader.sampler.epoch + 1)
            self.train_loader_iter = iter(self.train_loader)
            batch = next(self.train_loader_iter)

        return batch

    def _get_next_val_batch(self):
        """Same reset-on-exhaustion pattern as _get_next_batch, over val_loader."""
        if self.val_loader_iter is None:
            self.val_loader_iter = iter(self.val_loader)
        try:
            return next(self.val_loader_iter)
        except StopIteration:
            self.val_loader_iter = iter(self.val_loader)
            return next(self.val_loader_iter)

    @torch.no_grad()
    def _run_validation(self):
        """A held-out pass on validation_samples batches, without updating
        weights. Runs on every rank (the forward pass is a collective under
        FSDP), only rank 0 logs. Restores train() mode before returning."""
        self.transformer.eval()
        video_losses, action_losses = [], []
        # Compare the same held-out samples under the same diffusion noise at
        # every checkpoint. fork_rng restores the training stream afterwards,
        # so enabling validation cannot change subsequent optimizer updates.
        self.val_loader_iter = iter(self.val_loader)
        try:
            with torch.random.fork_rng(devices=[self.config.local_rank]):
                torch.manual_seed(12345 + self.config.rank)
                torch.cuda.manual_seed_all(12345 + self.config.rank)
                for _ in range(self.validation_samples):
                    batch = self.convert_input_format(self._get_next_val_batch())
                    input_dict = self._prepare_input_dict(batch)
                    output = self.transformer(input_dict, train_mode=True)
                    video_loss, action_loss = self.compute_loss(input_dict, output)
                    scale = self.gradient_accumulation_steps  # compute_loss divides for accumulation; undo for a true mean
                    video_losses.append(video_loss.detach() * scale)
                    action_losses.append(action_loss.detach() * scale)
        finally:
            self.transformer.train()
        mean_video = dist_mean(torch.stack(video_losses).mean()).item()
        mean_action = dist_mean(torch.stack(action_losses).mean()).item()
        if self.config.rank == 0:
            logger.info(f"validation at step {self.step}: "
                        f"video_loss={mean_video:.4f} action_loss={mean_action:.4f}")
            validation_dir = Path(self.config.save_root) / "validation"
            validation_dir.mkdir(parents=True, exist_ok=True)
            output = validation_dir / f"checkpoint_step_{self.step}.json"
            tmp_output = output.with_suffix(".json.tmp")
            tmp_output.write_text(json.dumps({
                "checkpoint": str(self.save_dir / f"checkpoint_step_{self.step}"),
                "dataset": os.environ.get("LINGBOT_VA_VAL_DATASET_PATH") or
                           f"{self.config.dataset_path}_val",
                "samples": self.validation_samples,
                "mean_video_loss": mean_video,
                "mean_action_loss": mean_action,
            }, indent=2) + "\n")
            os.replace(tmp_output, output)
            if self.config.enable_wandb:
                self.wandb.log({
                    'optimizer_step': self.step,
                    'val/mean_video_loss': mean_video,
                    'val/mean_action_loss': mean_action,
                })

    @torch.no_grad()
    def _add_noise(self, latent, train_scheduler, action_mask=False, action_mode=False, noisy_cond_prob=0.):
        B, C, F, H, W = latent.shape

        timestep_ids = sample_timestep_id(batch_size=F, num_train_timesteps=train_scheduler.num_train_timesteps)
        noise = torch.zeros_like(latent).normal_()
        timesteps = train_scheduler.timesteps[timestep_ids].to(device=self.device)
        noisy_latents =train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        targets =train_scheduler.training_target(latent, noise, timesteps)

        patch_f, patch_h, patch_w = self.patch_size
        if action_mode:
            patch_f = patch_h = patch_w = 1
        
        latent_grid_id = get_mesh_id(
            latent.shape[-3] // patch_f,  # F
            latent.shape[-2] // patch_h,  # H
            latent.shape[-1] // patch_w,  # W
            t=1 if action_mode else 0,  # 1 for action mode (0 for latent), not used
            f_w=1,
            f_shift=0,
            action=action_mode
        ).to(self.device)  # shape: [4, seq_len]
        latent_grid_id = latent_grid_id[None].repeat(B, 1, 1)

        if torch.rand(1).item() < noisy_cond_prob:
            cond_timestep_ids = sample_timestep_id(
                    batch_size=F,
                    min_timestep_bd=0.5, 
                    max_timestep_bd=1.0, 
                    num_train_timesteps=train_scheduler.num_train_timesteps,
                )
            noise = torch.zeros_like(latent).normal_()
            cond_timesteps = train_scheduler.timesteps[cond_timestep_ids].to(device=self.device)
            latent = train_scheduler.add_noise(latent, noise, cond_timesteps, t_dim=2)
        else:
            cond_timesteps = torch.zeros_like(timesteps)

        if action_mask is not None:
            noisy_latents *= action_mask.float()
            targets *= action_mask.float()
            latent *= action_mask.float()

        return dict(
            timesteps=timesteps[None].repeat(B, 1),
            noisy_latents=noisy_latents,
            targets=targets,
            latent=latent,
            cond_timesteps=cond_timesteps[None].repeat(B, 1),
            grid_id=latent_grid_id,
        )

    @torch.no_grad()
    def _prepare_input_dict(self, batch_dict):
        """Prepare input dict following infer code pattern from wan_va_server.py."""
        # Generate grid_id following infer code (no batch dimension yet)
        # For action mode: get_mesh_id(shape[-3], shape[-2], shape[-1], t=1, f_w=1, f_shift, action=True)
        latent_dict = self._add_noise(
            latent=batch_dict['latents'], 
            train_scheduler=self.train_scheduler_latent, 
            action_mask=None, 
            action_mode=False,
            noisy_cond_prob=0.5)
        
        action_dict = self._add_noise(
            latent=batch_dict['actions'], 
            train_scheduler=self.train_scheduler_action, 
            action_mask=batch_dict['actions_mask'], 
            action_mode=True,
            noisy_cond_prob=0.0)

        latent_dict['text_emb'] = batch_dict['text_emb']
        action_dict['text_emb'] = batch_dict['text_emb']
        action_dict['actions_mask'] = batch_dict['actions_mask']

        input_dict = {
            'latent_dict': latent_dict,
            'action_dict': action_dict,
            'chunk_size': torch.randint(1, 5, (1,)).item(),
            'window_size': torch.randint(4, 65, (1,)).item(),
        }
        return input_dict

    def convert_input_format(self, input_dict):
        """Convert input dict to match transformer input format if needed."""
        for key, value in input_dict.items():
            input_dict[key] = value.to(self.device)#.to(self.dtype)
        return input_dict

    def compute_loss(self,
        input_dict,
        pred
    ):
        latent_pred, action_pred = pred
        action_pred = rearrange(action_pred, 'b (f n) c -> b c f n 1', f=input_dict['action_dict']['targets'].shape[-3])
        latent_pred = data_seq_to_patch(
                        self.patch_size, latent_pred,
                        input_dict['latent_dict']['targets'].shape[-3], input_dict['latent_dict']['targets'].shape[-2],
                        input_dict['latent_dict']['targets'].shape[-1], batch_size=latent_pred.shape[0])
        Bn, Fn = input_dict['latent_dict']['timesteps'].shape
        latent_loss_weight = self.train_scheduler_latent.training_weight(input_dict['latent_dict']['timesteps'].flatten()).reshape(Bn, Fn)
        action_loss_weight = self.train_scheduler_action.training_weight(input_dict['action_dict']['timesteps'].flatten()).reshape(Bn, Fn)

        # Frame-wise video loss calculation
        latent_loss = F.mse_loss(latent_pred.float(), input_dict['latent_dict']['targets'].float().detach(), reduction='none')
        latent_loss = latent_loss * latent_loss_weight[:, None, :, None, None]
        # Permute to (B, F, H, W, C) and flatten to (B*F, H*W*C)
        latent_loss = latent_loss.permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        latent_loss = latent_loss.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        # Sum per frame and compute mask per frame
        latent_loss_per_frame = latent_loss.sum(dim=1)  # (B*F,)
        latent_mask_per_frame = torch.ones_like(latent_loss).sum(dim=1)  # (B*F,)
        latent_loss = (latent_loss_per_frame / (latent_mask_per_frame + 1e-6)).mean()

        # Frame-wise action loss calculation
        action_loss = F.mse_loss(action_pred.float(), input_dict['action_dict']['targets'].float().detach(), reduction='none')
        action_loss = action_loss * action_loss_weight[:, None, :, None, None]
        action_loss = action_loss * input_dict['action_dict']['actions_mask'].float()
        # Permute to (B, F, H, W, C) and flatten to (B*F, H*W*C)
        action_loss = action_loss.permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        action_mask = input_dict['action_dict']['actions_mask'].float().permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        action_loss = action_loss.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        action_mask = action_mask.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        # Sum per frame and normalize by mask per frame
        action_loss_per_frame = action_loss.sum(dim=1)  # (B*F,)
        action_mask_per_frame = action_mask.sum(dim=1)  # (B*F,)
        action_loss = (action_loss_per_frame / (action_mask_per_frame + 1e-6)).mean()

        return latent_loss / self.gradient_accumulation_steps, action_loss / self.gradient_accumulation_steps

    @torch.no_grad()
    def _log_input_images(self, batch):
        """Decode the first sample of the very first batch back to pixels and
        log it to wandb, once. A bad data path (wrong camera, stale latents,
        misaligned episode) shows up as a picture here instead of only ever
        showing up as a loss number three days in. The VAE is loaded and
        freed just for this one decode -- it is not needed anywhere else in
        training, which only ever sees precomputed latents."""
        if not (self.config.enable_wandb and self.config.rank == 0):
            return
        vae = None
        try:
            vae_path = os.path.join(self.config.wan22_pretrained_model_name_or_path, 'vae')
            vae = load_vae(vae_path, torch_dtype=torch.bfloat16, torch_device=self.device)
            latents = batch['latents'][:1].to(self.device, vae.dtype)
            latents_mean = torch.tensor(vae.config.latents_mean).view(
                1, vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
            latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(
                1, vae.config.z_dim, 1, 1, 1).to(latents.device, latents.dtype)
            video = vae.decode(latents / latents_std + latents_mean, return_dict=False)[0]
            video_processor = VideoProcessor(vae_scale_factor=1)
            frames = video_processor.postprocess_video(video, output_type='np')[0]
            stride = max(1, len(frames) // 6)
            images = [wandb.Image((frame * 255).clip(0, 255).astype('uint8'))
                      for frame in frames[::stride]]
            self.wandb.log({
                'optimizer_step': self.step,
                'input_images/first_batch': images,
            })
            logger.info(f"logged {len(images)} input-image frame(s) from the first batch to wandb")
        except Exception as exc:
            logger.warning(f"could not log input images: {exc}")
        finally:
            # A decode failure after the VAE reached the card used to leave it
            # resident and make the first real training step OOM.
            del vae
            gc.collect()
            torch.cuda.empty_cache()

    def _train_step(self, batch, batch_idx):
        """Train a single batch, returns losses for logging."""
        # The run's input does not change when a Slurm segment resumes. Avoid
        # loading a second 5B-model VAE at the beginning of every segment.
        if self.step == 0 and batch_idx == 0 and not self._logged_input_images:
            self._logged_input_images = True
            self._log_input_images(batch)
        batch = self.convert_input_format(batch)
        input_dict = self._prepare_input_dict(batch)

        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        
        if hasattr(self.transformer, 'set_requires_gradient_sync'):
            self.transformer.set_requires_gradient_sync(should_sync)

        output = self.transformer(input_dict, train_mode=True)
        latent_loss, action_loss = self.compute_loss(input_dict, output)
        loss = latent_loss + action_loss

        loss.backward()

        losses = {'latent_loss': latent_loss.detach(), 'action_loss': action_loss.detach()}
        
        # Activation checkpointing under a frozen base is the one way this can
        # train on nothing at all: a reentrant wrapper whose inputs need no
        # gradient never runs its backward, and the adapters inside it stay at
        # their initialisation while the loss falls on the video stream alone.
        # Check once, on the first backward, rather than after two days.
        if self.lora_rank and not self._checked_adapter_grads:
            self._checked_adapter_grads = True
            reached = sum(1 for name, p in self.transformer.named_parameters()
                          if p.requires_grad and '.lora_B' in name and p.grad is not None)
            if reached == 0:
                raise RuntimeError(
                    "no adapter received a gradient on the first backward -- the "
                    "activation-checkpoint wrapper is not passing one through. "
                    "Training would leave the LoRA at its initialisation.")
            logger.info(f"gradients reach {reached} adapters")

        # Only update weights after accumulating gradients
        if should_sync:
            total_norm = torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 2.0)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
            losses['total_norm'] = total_norm
            losses['should_log'] = True
        else:
            losses['should_log'] = False

        return losses

    def _latest_checkpoint(self):
        """The newest checkpoint_step_N directory that holds a transformer."""
        candidates = [d for d in self.save_dir.glob("checkpoint_step_*")
                      if (d / "transformer" / "config.json").exists()]
        if not candidates:
            return None
        return str(max(candidates, key=lambda d: int(d.name.rsplit("_", 1)[1])))

    def _resolve_resume_state(self, config):
        """Where to pick training back up, or None for a fresh run.

        `resume_from` is a directory holding training_state.pt; "auto" (the
        default under SLURM --requeue) means this run's own save directory.
        """
        requested = getattr(config, 'resume_from', '') or ''
        if requested and requested != 'auto':
            path = Path(requested)
            path = path if path.name == 'training_state.pt' else path / 'training_state.pt'
            if not path.exists():
                raise FileNotFoundError(f"resume_from has no training_state.pt: {path}")
            return path
        if requested == 'auto':
            path = self.save_dir / 'training_state.pt'
            return path if path.exists() else None
        return None

    def save_checkpoint(self,):
        """Save weights in the pretrained model's format, plus a resume state.

        Two files, for two readers. `transformer/` is what the inference server
        loads: LoRA adapters are folded in, so a checkpoint is keyed exactly
        like a full fine-tune. `training_state.pt` is what a requeued job
        reloads -- optimizer moments, the step counter and, under LoRA, the
        adapters themselves (the base on disk is the unchanged pretrained one).
        Upstream saved neither of the last three, so a requeue silently
        restarted at step 0.
        """
        # get_model_state_dict/get_optimizer_state_dict under FSDP are
        # collectives -- every rank must call them together, so they stay
        # outside the try below and are left to raise straight through a bad
        # rank rather than being caught and only logged by rank 0 (which used
        # to leave the other ranks silently out of step with a dead peer).
        if self.use_fsdp:
            state_dict = get_model_state_dict(
                self.transformer,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
        else:
            state_dict = {k: v.detach().cpu()
                          for k, v in self.transformer.state_dict().items()}
        if self.lora_rank:
            state_dict = merged_state_dict(self.transformer, state_dict)
        state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}

        if self.use_fsdp:
            optim_state = get_optimizer_state_dict(
                self.transformer, self.optimizer,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
        else:
            optim_state = self.optimizer.state_dict()

        # Only rank 0 touches the filesystem below, so only a local
        # try/except is needed here -- but a failure must not just be logged
        # and forgotten (it used to be): a training run with no valid recent
        # checkpoint is one bad Lustre write away from silently wasting every
        # GPU-hour since the last good save. failed_locally is all-reduced
        # below so every rank raises together instead of rank 0 dying alone
        # while the others wait at a barrier it never reaches.
        failed_locally = False
        if self.config.rank == 0:
            checkpoint_dir = self.save_dir / f"checkpoint_step_{self.step}"
            tmp_checkpoint_dir = self.save_dir / f".checkpoint_step_{self.step}.tmp-{os.getpid()}"
            try:
                shutil.rmtree(tmp_checkpoint_dir, ignore_errors=True)
                transformer_dir = tmp_checkpoint_dir / "transformer"
                transformer_dir.mkdir(parents=True, exist_ok=True)

                logger.info(f"Saving transformer to {transformer_dir}")
                save_file(state_dict_bf16, transformer_dir / "diffusion_pytorch_model.safetensors")

                config_dict = dict(self.transformer.config)
                config_dict.pop('_name_or_path', None)
                with open(transformer_dir / "config.json", 'w') as f:
                    json.dump(config_dict, f, indent=2)

                # The action quantiles the head was trained against, beside the
                # weights: serving needs them and must not have to find the
                # training dataset to get them.
                with open(tmp_checkpoint_dir / "lingbot_norm_stat.json", 'w') as f:
                    json.dump({"q01": list(self.config.norm_stat["q01"]),
                               "q99": list(self.config.norm_stat["q99"])}, f)

                # Publish only after weights, config and normalisation are all
                # complete. Resume/eval discovery never sees a half-written
                # checkpoint if Slurm ends the process during a 10 GB write.
                if checkpoint_dir.exists():
                    raise FileExistsError(
                        f"refusing to replace existing checkpoint {checkpoint_dir}")
                os.replace(tmp_checkpoint_dir, checkpoint_dir)

                trainable = {}
                if self.lora_rank:
                    trainable = {k: v.detach().cpu()
                                 for k, v in self.transformer.state_dict().items()
                                 if '.lora_A' in k or '.lora_B' in k or
                                 k.startswith(('action_embedder.', 'action_proj_out.')) or
                                 (getattr(self.config, 'train_video_projections', False) and
                                  k.startswith(('patch_embedding_mlp.', 'proj_out.'))) or
                                 (getattr(self.config, 'train_action_condition', False) and
                                  k.startswith('condition_embedder_action.'))}

                state_path = self.save_dir / "training_state.pt"
                tmp_path = state_path.with_suffix('.pt.tmp')
                torch.save({
                    'step': self.step,
                    'optimizer_state_dict': optim_state,
                    'trainable_state_dict': trainable,
                    'lora_rank': self.lora_rank,
                    # Segment resumes should continue with fresh diffusion
                    # noise/dropout and a fresh data permutation, rather than
                    # replaying the same seeded streams every six hours.
                    'torch_rng_state': torch.get_rng_state(),
                    'cuda_rng_state_all': torch.cuda.get_rng_state_all(),
                    'train_loader_generator_state': self.train_loader_generator.get_state(),
                }, tmp_path)
                os.replace(tmp_path, state_path)

                # Frequent resume points are useful for short Slurm segments,
                # but a merged 5.3B checkpoint is about 10 GB. Keep recent
                # recovery points and optional milestone checkpoints so a
                # long run stays within its storage allocation.
                keep_last = int(getattr(
                    self.config, 'keep_last_checkpoints', 0) or 0)
                keep_every = int(getattr(
                    self.config, 'keep_checkpoint_interval', 0) or 0)
                if keep_last:
                    checkpoints = sorted(
                        self.save_dir.glob("checkpoint_step_*"),
                        key=lambda d: int(d.name.rsplit("_", 1)[1]))
                    recent = set(checkpoints[-keep_last:])
                    for old in checkpoints:
                        old_step = int(old.name.rsplit("_", 1)[1])
                        if old not in recent and not (
                                keep_every and old_step % keep_every == 0):
                            shutil.rmtree(old)
                            logger.info(f"Pruned checkpoint at step {old_step}")

                logger.info(f"Checkpoint saved successfully at step {self.step}")
            except Exception as e:
                failed_locally = True
                shutil.rmtree(tmp_checkpoint_dir, ignore_errors=True)
                logger.error(f"Failed to save checkpoint: {e}")
                import traceback
                logger.error(traceback.format_exc())

        if dist.is_initialized():
            fail_flag = torch.tensor(
                [1 if failed_locally else 0], device=self.device, dtype=torch.int32)
            dist.all_reduce(fail_flag, op=dist.ReduceOp.MAX)
            dist.barrier()
            failed = bool(fail_flag.item())
        else:
            failed = failed_locally

        if failed:
            raise RuntimeError(
                f"checkpoint save failed at step {self.step} -- see the error "
                "logged above. Stopping rather than continuing without a valid "
                "recent checkpoint; resubmit to resume from the last good save.")

    def _load_training_state(self, state_path):
        """Restore optimizer moments, adapters and the step counter."""
        logger.info(f"Loading training state from {state_path}")
        training_state = torch.load(state_path, map_location='cpu', weights_only=False)

        saved_rank = training_state.get('lora_rank', 0)
        if saved_rank != self.lora_rank:
            raise ValueError(
                f"training state was written with lora_rank={saved_rank}, this run "
                f"has {self.lora_rank}")

        trainable = training_state.get('trainable_state_dict') or {}
        if trainable:
            missing, unexpected = self.transformer.load_state_dict(trainable, strict=False)
            if unexpected:
                raise ValueError(f"resume state has unknown keys: {unexpected[:5]}")

        if self.use_fsdp:
            set_optimizer_state_dict(
                self.transformer, self.optimizer,
                optim_state_dict=training_state['optimizer_state_dict'],
                options=StateDictOptions(full_state_dict=True, strict=False)
            )
        else:
            self.optimizer.load_state_dict(training_state['optimizer_state_dict'])
        self.step = training_state.get('step', 0)
        for _ in range(self.step):
            self.lr_scheduler.step()

        if training_state.get('train_loader_generator_state') is not None:
            self.train_loader_generator.set_state(
                training_state['train_loader_generator_state'])
        if training_state.get('torch_rng_state') is not None:
            torch.set_rng_state(training_state['torch_rng_state'])
        if training_state.get('cuda_rng_state_all') is not None:
            torch.cuda.set_rng_state_all(training_state['cuda_rng_state_all'])

        logger.info(f"Training state loaded, resuming from step {self.step}")

        if dist.is_initialized():
            dist.barrier()

    def train(self):
        """Main training loop - train by steps instead of epochs."""
        logger.info(f"Starting training for {self.config.num_steps} steps...")
        self.transformer.train()

        progress_bar = tqdm(
            total=self.config.num_steps,
            desc="Training",
            disable=(self.config.rank != 0),
            leave=True,
            dynamic_ncols=True,
            initial=self.step
        )

        self.optimizer.zero_grad()
        optimizer_step_started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(self.device)
        accumulated_latent_losses = []
        accumulated_action_losses = []
        step_in_accumulation = 0

        while self.step < self.config.num_steps:
            # Get next batch (handles epoch reset automatically)
            batch = self._get_next_batch()
            
            losses = self._train_step(batch, step_in_accumulation)
            
            # Accumulate losses for logging
            accumulated_latent_losses.append(losses['latent_loss'])
            accumulated_action_losses.append(losses['action_loss'])
            step_in_accumulation += 1

            # Log and checkpoint when optimizer steps
            if losses['should_log']:
                lr = self.lr_scheduler.get_last_lr()[0]

                # Average accumulated losses
                latent_loss_show = dist_mean(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                action_loss_show = dist_mean(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()
                max_latent_loss_show = dist_max(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                max_action_loss_show = dist_max(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()

                # Clear accumulated losses
                accumulated_latent_losses = []
                accumulated_action_losses = []
                step_in_accumulation = 0

                torch.cuda.synchronize()
                optimizer_step_seconds = time.perf_counter() - optimizer_step_started
                peak_allocated_gib = torch.cuda.max_memory_allocated(self.device) / 2**30
                peak_reserved_gib = torch.cuda.max_memory_reserved(self.device) / 2**30
                optimizer_step_started = time.perf_counter()
                torch.cuda.reset_peak_memory_stats(self.device)
                if self.step % self.config.gc_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                if self.config.rank == 0:
                    total_norm = losses['total_norm']
                    progress_bar.n += 1
                    progress_bar.set_postfix({
                        'latent_loss': f'{latent_loss_show:.4f}',
                        'action_loss': f'{action_loss_show:.4f}',
                        'step': self.step,
                        'grad_norm': f'{total_norm.item():.2f}',
                        'lr': f'{lr:.2e}',
                        'step_s': f'{optimizer_step_seconds:.2f}',
                        'peak_GiB': f'{peak_allocated_gib:.1f}',
                    })
                    if self.config.enable_wandb:
                        self.wandb.log({
                            'optimizer_step': self.step,
                            'loss_metrics/global_avg_video_loss': latent_loss_show,
                            'loss_metrics/global_avg_action_loss': action_loss_show,
                            'loss_metrics/global_max_video_loss': max_latent_loss_show,
                            'loss_metrics/global_max_action_loss': max_action_loss_show,
                            'grad_norm': total_norm.item(),
                            'lr': lr,
                            'performance/optimizer_step_seconds': optimizer_step_seconds,
                            'performance/peak_allocated_gib': peak_allocated_gib,
                            'performance/peak_reserved_gib': peak_reserved_gib,
                        })
                
                self.step += 1
                
                if self.step % self.config.save_interval == 0:
                    if self.config.rank == 0:
                        logger.info(f"Starting save model at step {self.step}")
                    self.save_checkpoint()

                if self.validation_interval and self.step % self.validation_interval == 0:
                    try:
                        self._run_validation()
                    except Exception as exc:
                        # Validation is diagnostic. A bad held-out sample or a
                        # transient logging failure must not discard days of
                        # otherwise healthy optimizer work.
                        logger.exception(
                            f"validation failed at step {self.step}; training continues: {exc}")
                        self.transformer.train()
                        torch.cuda.empty_cache()

                stop_file = os.environ.get("LINGBOT_STOP_FILE", "")
                stop_requested = bool(stop_file and os.path.exists(stop_file))
                if dist.is_initialized():
                    stop_flag = torch.tensor(
                        [int(stop_requested)], device=self.device, dtype=torch.int32)
                    dist.all_reduce(stop_flag, op=dist.ReduceOp.MAX)
                    stop_requested = bool(stop_flag.item())
                if stop_requested:
                    if self.step % self.config.save_interval:
                        if self.config.rank == 0:
                            logger.info(
                                f"Graceful wall-time stop requested; saving step {self.step}")
                        self.save_checkpoint()
                    if self.config.rank == 0:
                        try:
                            os.unlink(stop_file)
                        except FileNotFoundError:
                            pass
                    if dist.is_initialized():
                        dist.barrier()
                    progress_bar.close()
                    logger.info(f"Stopped cleanly after checkpointing step {self.step}")
                    return

            if dist.is_initialized():
                dist.barrier()

        progress_bar.close()
        logger.info("Training completed!")


def run(args):
    """Main entry point."""
    config = VA_CONFIGS[args.config_name]

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    init_distributed(world_size, local_rank, rank)

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    if args.save_root is not None:
        config.save_root = args.save_root

    if rank == 0:
        logger.info(f"Using config: {args.config_name}")
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")

    trainer = Trainer(config)
    trainer.train()


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Train WAN model for robotics")
    parser.add_argument(
        "--config-name",
        type=str,
        default='robotwin_train',
        help="Config name",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default=None,
        help="Root directory for saving checkpoints",
    )
    # run_va_posttrain.sh has always forwarded --seed; argparse rejected it.
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for torch/numpy/python RNGs",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()
