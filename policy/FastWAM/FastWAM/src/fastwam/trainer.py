import logging
import json
import inspect
import math
import os
import re
from math import ceil
from pathlib import Path
import time

import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig
from PIL import Image
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from .utils.fs import ensure_dir
from .utils import lora as lora_utils
from .utils.logging_config import get_logger, setup_logging
from .utils.pytorch_utils import set_global_seed
from .utils.samplers import ResumableEpochSampler
from .utils.video_io import save_mp4
from .utils.video_metrics import pil_frames_to_video_tensor, video_psnr, video_ssim

logger = get_logger(__name__)


class Wan22Trainer:
    def __init__(self, model, train_dataset, val_dataset=None, *, cfg: DictConfig):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.cfg = cfg
        self.output_dir = str(cfg.output_dir)
        self.learning_rate = float(cfg.learning_rate)
        self.weight_decay = float(cfg.weight_decay)
        self.batch_size = int(cfg.batch_size)
        self.num_workers = int(cfg.num_workers)
        self.num_epochs = int(cfg.num_epochs)
        max_steps = cfg.max_steps
        self.max_steps = int(max_steps) if max_steps is not None else None
        self.log_every = int(cfg.log_every)
        self.save_every = int(cfg.save_every)
        self.eval_every = int(cfg.eval_every)
        self.eval_num_inference_steps = int(cfg.eval_num_inference_steps)
        # How many held-out samples the validation LOSS averages over, and the
        # seed that fixes its noise. Upstream evaluated one randomly drawn
        # sample with freshly drawn flow-matching noise, which makes
        # `eval/val_loss` an n=1 estimate of a quantity whose per-sample
        # variance is dominated by the sampled timestep -- the curve moved far
        # more between neighbouring evaluations than it moved over the whole
        # run. Both knobs below are what make the curve a training signal:
        # a fixed sample set removes the sampling noise, and a fixed seed makes
        # every evaluation score the SAME noise/timestep draws, so successive
        # points differ only by the weights. A loss forward is ~0.05 s/sample
        # here against a 2.3 s training step, so 32 samples cost well under a
        # step and the video/action metrics below still run on one sample.
        self.eval_num_loss_samples = int(cfg.get("eval_num_loss_samples", 32))
        self.eval_noise_seed = int(cfg.get("eval_noise_seed", 12345))
        # Consecutive non-finite optimizer steps tolerated before the run is
        # aborted. 0 disables the check.
        self.max_nonfinite_steps = int(cfg.get("max_nonfinite_steps", 20))
        self._nonfinite_streak = 0
        # Localise the first non-finite tensor instead of only reporting that
        # the loss went nan. By the step after a nan every parameter is nan and
        # the log reads identically whether a batch was bad, the backward
        # overflowed, or the update did -- so the cause has to be caught in the
        # step it happens. Off by default: the check reduces every parameter and
        # every gradient once per micro-batch, which is far too expensive for a
        # real run and cheap for the dozen steps of a probe.
        self.debug_nonfinite = bool(cfg.get("debug_nonfinite", False))
        self._debug_reported = set()
        self.gradient_accumulation_steps = int(cfg.gradient_accumulation_steps)
        self.max_grad_norm = float(cfg.max_grad_norm)
        self.seed = int(cfg.seed)
        
        self.resume = cfg.resume
        # Fine-tuning knob: train the action expert only, leaving the pretrained
        # video world model frozen. Off by default, so full fine-tuning is
        # unchanged.
        self.freeze_video_dit = bool(cfg.get("freeze_video_dit", False))
        # The other parameter-efficient lever: LoRA on the video expert's
        # linear projections, with the action expert still trained in full
        # (its head has to be relearned at the new action width anyway). Rank
        # 0 is off, which leaves full fine-tuning untouched.
        self.lora_rank = int(cfg.get("lora_rank", 0) or 0)
        self.lora_alpha = float(cfg.get("lora_alpha", 16) or 16)
        self.lora_dropout = float(cfg.get("lora_dropout", 0.0) or 0.0)
        if self.lora_rank and self.freeze_video_dit:
            raise ValueError(
                "freeze_video_dit and lora_rank are alternatives: freezing the video "
                "expert outright leaves the adapters with nothing to adapt."
            )
        # Weights-only initialisation from a released checkpoint, as opposed to
        # `resume`, which continues a run of our own. Kept separate because it
        # has to happen BEFORE accelerate/DeepSpeed build the optimizer (see
        # `_load_init_weights`).
        self.init_weights = cfg.get("init_weights", None)
        self.mixed_precision = str(cfg.mixed_precision).strip().lower()
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError(
                f"Unsupported mixed_precision: {cfg.mixed_precision}. "
                "Expected one of: ['no', 'fp16', 'bf16']."
            )
        self.wandb_enabled = bool(cfg.wandb.enabled)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=self.mixed_precision,
            step_scheduler_with_optimizer=False,
        )
        
        logger.info(
            "Accelerate training: distributed_type=%s zero_stage=%s world_size=%d process_index=%d cfg_mixed_precision=%s accelerator_mixed_precision=%s grad_accum=%d grad_clip=%.4f",
            self.accelerator.distributed_type,
            self.accelerator.state.deepspeed_plugin.deepspeed_config.get("zero_optimization", {}).get("stage", "unknown"),
            self.accelerator.num_processes,
            self.accelerator.process_index,
            self.mixed_precision,
            self.accelerator.mixed_precision,
            self.gradient_accumulation_steps,
            self.max_grad_norm,
        )
        logger.info("using accelerator.device=%s", self.accelerator.device)
        worker_init_fn = set_global_seed(self.seed, get_worker_init_fn=True)
        self._assert_dataset_length_consistent(self.train_dataset, "train_dataset")
        if self.val_dataset is not None:
            self._assert_dataset_length_consistent(self.val_dataset, "val_dataset")

        # Both of these must land before `accelerator.prepare`: DeepSpeed copies
        # the bf16 parameters into its fp32 master partition at optimizer
        # construction, so weights written afterwards are silently overwritten
        # by the first step, and adapters added afterwards would have no
        # optimizer state at all.
        self._load_init_weights()
        self._inject_lora()

        # Freeze non-trainable modules before optimizer/deepspeed initialization.
        # This keeps DiT (+ optional proprio encoder) as trainable when ZeRO builds optimizer state.
        self._apply_dit_only_train_mode(
            self.model, freeze_video_dit=self.freeze_video_dit, lora=bool(self.lora_rank)
        )
        trainable_params = [p for p in self.model.dit.parameters() if p.requires_grad]
        proprio_encoder = getattr(self.model, "proprio_encoder", None)
        if proprio_encoder is not None:
            trainable_params.extend(p for p in proprio_encoder.parameters() if p.requires_grad)
        if not trainable_params:
            raise ValueError("No trainable parameters remain after applying freeze settings.")
        logger.info(
            "Trainable parameters: %.3fB of %.3fB (freeze_video_dit=%s lora_rank=%d)",
            sum(p.numel() for p in trainable_params) / 1e9,
            sum(p.numel() for p in self.model.dit.parameters()) / 1e9,
            self.freeze_video_dit,
            self.lora_rank,
        )
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )
        
        self.train_loader = self._build_loader(self.train_dataset, worker_init_fn=worker_init_fn)
        total_train_steps = self._estimate_total_train_steps()
        self.max_steps = total_train_steps
        warmup_steps = int(total_train_steps * 0.05)
        self.scheduler = self._build_scheduler(
            scheduler_type=cfg.lr_scheduler_type,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
        )
        self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0

        self.checkpoint_root = os.path.join(self.output_dir, "checkpoints")
        self.weights_dir = os.path.join(self.checkpoint_root, "weights")
        self.state_dir = os.path.join(self.checkpoint_root, "state")
        self.eval_dir = os.path.join(self.output_dir, "eval")

        ensure_dir(self.output_dir)
        ensure_dir(self.checkpoint_root)
        ensure_dir(self.weights_dir)
        ensure_dir(self.state_dir)
        ensure_dir(self.eval_dir)

        self.model, self.optimizer, self.train_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.scheduler
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.wandb_run = None
        self._init_wandb()
        self._resume_or_load_checkpoint()

        val_size = len(self.val_dataset) if self.val_dataset is not None else len(self.train_dataset)
        logger.info("Train/val dataset size: %d/%d", len(self.train_dataset), val_size)

    def _init_wandb(self):
        if not self.wandb_enabled or not self.accelerator.is_main_process:
            return
        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "wandb logging is enabled in config (`wandb.enabled=true`) but wandb is not installed."
            ) from e

        self.wandb_run = wandb.init(
            entity=self.cfg.wandb.workspace,
            project=self.cfg.wandb.project,
            name=self.cfg.wandb.name,
            group=None if self.cfg.wandb.group in (None, "null", "") else str(self.cfg.wandb.group),
            mode=self.cfg.wandb.mode,
            dir=self.output_dir,
        )
        logger.info(
            "Initialized wandb run: workspace=%s project=%s name=%s",
            self.cfg.wandb.workspace,
            self.cfg.wandb.project,
            self.cfg.wandb.name,
        )

    def _wandb_log(self, payload: dict):
        if self.wandb_run is None:
            return
        self.wandb_run.log(payload, step=self.global_step)

    def _finish_wandb(self):
        if self.wandb_run is None:
            return
        self.wandb_run.finish()
        self.wandb_run = None

    def _build_loader(self, dataset, worker_init_fn=None):
        self.train_sampler = ResumableEpochSampler(
            dataset=dataset,
            seed=self.seed,
            batch_size=self.batch_size,
            num_processes=self.accelerator.num_processes,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=worker_init_fn,
        )

    def _assert_dataset_length_consistent(self, dataset, dataset_name: str):
        if not hasattr(dataset, "__len__"):
            raise TypeError(f"`{dataset_name}` must implement __len__ for rank consistency checks.")

        local_length = len(dataset)
        gathered_lengths = self.accelerator.gather(
            torch.tensor([local_length], device=self.accelerator.device, dtype=torch.int64)
        ).reshape(-1)
        if torch.all(gathered_lengths == gathered_lengths[0]):
            return

        if self.accelerator.is_main_process:
            print(f"[dataset-check] {dataset_name} length mismatch across ranks after initialization:")
            for rank, rank_length in enumerate(gathered_lengths.cpu().tolist()):
                print(f"rank {rank}: {rank_length}")
        self.accelerator.wait_for_everyone()
        raise RuntimeError(
            f"{dataset_name} length mismatch across ranks: {gathered_lengths.cpu().tolist()}"
        )

    def _estimate_total_train_steps(self) -> int:
        if self.max_steps is not None:
            return max(int(self.max_steps), 1)

        if not hasattr(self.train_dataset, "__len__"):
            raise TypeError("`train_dataset` must implement __len__ when `max_steps` is None.")

        num_processes = max(int(self.accelerator.num_processes), 1)
        global_batch_size = max(self.batch_size * num_processes, 1)
        micro_steps_per_epoch = max(ceil(len(self.train_dataset) / global_batch_size), 1)
        opt_steps_per_epoch = max(
            ceil(micro_steps_per_epoch / self.gradient_accumulation_steps),
            1,
        )
        return max(opt_steps_per_epoch * self.num_epochs, 1)

    def _build_scheduler(self, scheduler_type, total_train_steps: int, warmup_steps: int = 0):
        scheduler_type = str(scheduler_type).strip().lower()
        total_train_steps = max(int(total_train_steps), 1)
        warmup_steps = min(max(int(warmup_steps), 0), total_train_steps - 1)

        remaining_steps = max(total_train_steps - warmup_steps, 1)
        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=remaining_steps,
                eta_min=self.learning_rate * 0.01,
            )
        elif scheduler_type == "constant":
            main_scheduler = ConstantLR(self.optimizer, factor=1.0, total_iters=remaining_steps)
        else:
            raise ValueError(
                f"Unsupported lr_scheduler_type: {scheduler_type}. "
                "Expected one of: ['cosine', 'constant']."
            )

        if warmup_steps <= 0:
            return main_scheduler

        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=1.0 / warmup_steps,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
    
    def _estimate_eta(self):
        elapsed = max(time.perf_counter() - self.run_start_time, 1e-6)
        done_steps = max(self.global_step - self.run_start_step, 1)
        steps_per_sec = done_steps / elapsed
        remaining_steps = max(self.max_steps - self.global_step, 0)
        eta_seconds = int(remaining_steps / max(steps_per_sec, 1e-9))
        eta_h, eta_rem = divmod(eta_seconds, 3600)
        eta_m, eta_s = divmod(eta_rem, 60)
        return f"{eta_h:02d}:{eta_m:02d}:{eta_s:02d}", steps_per_sec

    def _load_init_weights(self):
        """Initialise from a released .pt before the optimizer exists.

        `resume` cannot do this job: it runs after `accelerator.prepare`, where
        DeepSpeed has already copied the bf16 parameters into its fp32 master
        partition, so the first optimizer step would write the pre-load values
        straight back over anything loaded there.
        """
        if not self.init_weights:
            return
        path = Path(str(self.init_weights))
        if not path.is_file():
            raise FileNotFoundError(f"init_weights checkpoint not found: {path}")
        logger.info("Initialising weights from %s", path)
        payload = self.model.load_checkpoint(str(path), optimizer=None)
        logger.info(
            "Loaded init weights (source step=%s dtype=%s); optimizer and step start from zero.",
            payload.get("step"), payload.get("torch_dtype"),
        )

    def _inject_lora(self):
        if not self.lora_rank:
            return
        video_dit = self._video_expert(self.model)
        if video_dit is None:
            raise ValueError("lora_rank > 0 requires a model with a video DiT expert.")
        layers, added = lora_utils.inject_lora(
            video_dit, rank=self.lora_rank, alpha=self.lora_alpha, dropout=self.lora_dropout,
        )
        if layers == 0:
            raise ValueError(
                "lora_rank > 0 but no target projection matched; the video expert's block "
                f"layout must have changed (expected {lora_utils.DEFAULT_TARGETS})."
            )
        logger.info(
            "LoRA: wrapped %d projections in the video expert, +%.1fM parameters "
            "(rank=%d alpha=%.1f dropout=%.2f)",
            layers, added / 1e6, self.lora_rank, self.lora_alpha, self.lora_dropout,
        )

    @staticmethod
    def _video_expert(model):
        # nn.ModuleDict has no .get(), so membership-test it.
        mixtures = getattr(model.dit, "mixtures", None)
        if mixtures is not None and "video" in mixtures:
            return mixtures["video"]
        return getattr(model, "video_expert", None)

    def _resume_or_load_checkpoint(self):
        resume = self.resume
        if not resume:
            return
        resume_path = Path(str(resume))
        if resume_path.is_dir():
            logger.info("Resuming full training state from directory: %s", resume)
            self.load_training_state(str(resume_path))
            return
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        logger.info("Loading weight checkpoint only: %s", resume)
        self.accelerator.unwrap_model(self.model).load_checkpoint(str(resume_path), optimizer=None)
        logger.warning("Loaded .pt weights only; optimizer/scheduler/step were not restored under ZeRO2.")

    def _set_dit_only_train_mode(self):
        # Match DiffSynth's freeze_except("dit"): only DiT stays trainable/in-train-mode.
        logger.info("Setting DiT to train mode and freezing other model components.")
        model = self.accelerator.unwrap_model(self.model)
        # Also here, not just at init: this runs again after every eval pass, and
        # without the flags it would quietly re-enable grads on the video expert.
        self._apply_dit_only_train_mode(
            model, freeze_video_dit=self.freeze_video_dit, lora=bool(self.lora_rank)
        )

    @classmethod
    def _apply_dit_only_train_mode(cls, model, *, freeze_video_dit: bool = False,
                                   lora: bool = False):
        model.eval()
        model.requires_grad_(False)
        model.dit.train()
        model.dit.requires_grad_(True)
        if freeze_video_dit or lora:
            # Fine-tuning a released checkpoint on a small downstream dataset:
            # hold the pretrained video world model back and spend the
            # optimizer on the action expert (+ proprio encoder). The MoT keeps
            # both experts in one joint attention, so this removes the video
            # expert's weight gradients and optimizer state, not its forward
            # pass. With `lora` the adapters are then re-enabled, which is the
            # only difference between the two arms.
            video_dit = cls._video_expert(model)
            if video_dit is None:
                raise ValueError(
                    "freeze_video_dit/lora require a model with a video DiT expert."
                )
            video_dit.eval()
            video_dit.requires_grad_(False)
            if lora:
                # train() for the adapters' dropout; the base blocks carry no
                # dropout or running statistics, so the mode is otherwise inert.
                video_dit.train()
                for param in lora_utils.lora_parameters(video_dit):
                    param.requires_grad_(True)
        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None:
            proprio_encoder.train()
            proprio_encoder.requires_grad_(True)

    @staticmethod
    def _to_batched_eval_sample(sample):
        video = sample["video"]
        prompt = sample["prompt"]
        action = sample.get("action", None)
        proprio = sample.get("proprio", None)
        context = sample.get("context", None)
        context_mask = sample.get("context_mask", None)

        if not isinstance(video, torch.Tensor):
            raise TypeError(
                f"Expected tensor video for evaluation, got {type(video)}. "
                "Evaluation now expects `video` with shape [3,T,H,W] or [B,3,T,H,W]."
            )
        if video.ndim == 4:
            video = video.unsqueeze(0)
        if video.ndim != 5:
            raise ValueError(f"Expected video shape [3,T,H,W] or [B,3,T,H,W], got {tuple(video.shape)}")
        num_video_frames = video.shape[2]
        if num_video_frames <= 1:
            raise ValueError(f"`sample['video']` must have at least 2 frames for action evaluation, got {num_video_frames}")

        if isinstance(prompt, str):
            prompt = [prompt]
        elif isinstance(prompt, tuple):
            prompt = list(prompt)
        elif not isinstance(prompt, list):
            raise TypeError(f"Expected prompt type str/list[str], got {type(prompt)}")
        if len(prompt) != video.shape[0]:
            raise ValueError(f"Prompt batch mismatch: len(prompt)={len(prompt)} vs video batch={video.shape[0]}")
        
        action_horizon = None
        action = None
        if "action" in sample:
            action = sample["action"]
            if not isinstance(action, torch.Tensor):
                raise TypeError(
                    f"`sample['action']` must be a torch.Tensor, got {type(action)}"
                )
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3:
                raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
            if action.shape[1] % (num_video_frames - 1) != 0:
                raise ValueError(f"`sample['action']` temporal dimension must be divisible by video frames-1={num_video_frames - 1}, got {action.shape[1]}")
            action_horizon = int(action.shape[1])

        proprio = None
        if "proprio" in sample:
            proprio = sample["proprio"]
            if not isinstance(proprio, torch.Tensor):
                raise TypeError(f"`sample['proprio']` must be a torch.Tensor, got {type(proprio)}")
            if proprio.ndim == 2:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")

        if context is not None or context_mask is not None:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must both exist in eval sample.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )

        return {
            "video": video,
            "prompt": prompt,
            "action": action,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "action_horizon": action_horizon,
        }

    @torch.no_grad()
    def evaluate(self):
        if self.val_dataset is None:
            return None

        model = self.accelerator.unwrap_model(self.model)
        was_dit_training = model.dit.training
        model.eval()

        # 1. validation loss over a FIXED subset with FIXED noise.
        #
        # The indices are evenly spaced over the split and identical at every
        # evaluation, and `fork_rng` + a per-sample seed replays the same
        # flow-matching noise and timesteps each time, so a change in the curve
        # is a change in the model. Sharding by rank keeps the cost flat as
        # GPUs are added; the sum and the count are gathered separately so an
        # uneven split still averages correctly.
        n_val = len(self.val_dataset)
        n_loss = max(1, min(self.eval_num_loss_samples, n_val))
        span = max(n_loss - 1, 1)
        eval_indices = [int(round(i * (n_val - 1) / span)) for i in range(n_loss)]
        rank = self.accelerator.process_index
        my_indices = eval_indices[rank :: self.accelerator.num_processes] or eval_indices[:1]

        fork_devices = [self.accelerator.device] if torch.cuda.is_available() else []
        val_loss_sum = 0.0
        with torch.random.fork_rng(devices=fork_devices, enabled=True):
            for idx in my_indices:
                torch.manual_seed(self.eval_noise_seed + idx)
                if fork_devices:
                    torch.cuda.manual_seed_all(self.eval_noise_seed + idx)
                loss_sample = self._to_batched_eval_sample(self.val_dataset[idx])
                with self.accelerator.autocast():
                    loss_value, _ = model.training_loss(loss_sample)
                val_loss_sum += float(loss_value.float().item())
        val_loss = val_loss_sum / len(my_indices)

        # The video and action metrics below stay on a single sample -- each one
        # costs a full denoising rollout -- but it is now a FIXED sample rather
        # than a fresh random draw, so those curves are comparable across steps
        # for the same reason the loss is.
        eval_index = my_indices[0]
        sample = self._to_batched_eval_sample(self.val_dataset[eval_index])
        
        prompt = sample["prompt"][0]
        video0 = sample["video"][0] # Tensor [3, T, H, W] in (-1, 1)
        action = sample["action"][0] if "action" in sample and sample["action"] is not None else None
        proprio = sample["proprio"][0, 0] if "proprio" in sample and sample["proprio"] is not None else None # from [1, T, d] to [d]
        input_image = video0[:, 0].unsqueeze(0)
        _, num_frames, _, _ = video0.shape

        # 2. inference and video saving
        infer_kwargs = {
            "input_image": input_image,
            "num_frames": num_frames,
            "action": action,
            "action_horizon": sample['action_horizon'],
            "proprio": proprio,
            "text_cfg_scale": 1.0,
            "action_cfg_scale": 1.0,
            "num_inference_steps": self.eval_num_inference_steps,
            "seed": 42,
            "tiled": False,
        }
        if sample["context"] is not None:
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = sample["context"][0]
            infer_kwargs["context_mask"] = sample["context_mask"][0]
        else:
            infer_kwargs["prompt"] = prompt

        pred = model.infer(
            **infer_kwargs,
        )
        
        pred_video = pred["video"]
        pred_action = pred.get("action", None)

        # 3. inference metrics against GT video
        pred_video_tensor = pil_frames_to_video_tensor(pred_video)
        gt_video_tensor = ((video0.detach().float().cpu().clamp(-1.0, 1.0) + 1.0) * 0.5).contiguous()

        assert pred_video_tensor.shape == gt_video_tensor.shape, (
            "Eval infer prediction/GT shape mismatch: "
            f"pred={tuple(pred_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_rollout_vs_gt = video_psnr(pred=pred_video_tensor, target=gt_video_tensor)
        ssim_rollout_vs_gt = video_ssim(pred=pred_video_tensor, target=gt_video_tensor)

        action_l1 = None
        action_l2 = None
        if action is not None and pred_action is not None:
            if sample["proprio"] is None:
                raise ValueError("Eval sample must contain `proprio` for action denormalization.")
            proprio = sample["proprio"].detach().to(device="cpu", dtype=torch.float32)
            
            processor = self.val_dataset.lerobot_dataset.processor

            denorm_actions = {}
            action_meta = processor.shape_meta["action"]
            state_meta = processor.shape_meta["state"]
            for action_name, raw_action in (("pred", pred_action), ("gt", action)):
                if not isinstance(raw_action, torch.Tensor):
                    raise TypeError(f"{action_name} action must be a torch.Tensor, got {type(raw_action)}")
                if raw_action.ndim == 2:
                    action_btd = raw_action.unsqueeze(0)
                elif raw_action.ndim == 3 and raw_action.shape[0] == 1:
                    action_btd = raw_action
                else:
                    raise ValueError(
                        f"{action_name} action must have shape [T, D] or [1, T, D], got {tuple(raw_action.shape)}"
                    )
                action_btd = action_btd.detach().to(device="cpu", dtype=torch.float32)

                batch = {
                    "action": action_btd,
                    "state": proprio,
                }
                batch = processor.action_state_merger.backward(batch)
                batch = processor.normalizer.backward(batch)
                merged_batch = {
                    "action": {meta["key"]: batch["action"][meta["key"]].squeeze(0) for meta in action_meta},
                    "state": {meta["key"]: batch["state"][meta["key"]].squeeze(0) for meta in state_meta},
                }
                merged_batch = processor.action_state_merger.forward(merged_batch)
                denorm_action = merged_batch["action"].unsqueeze(0)
                if denorm_action.ndim != 3 or denorm_action.shape[0] != 1:
                    raise ValueError(
                        f"Denormalized {action_name} action must have shape [1, T, D], got {tuple(denorm_action.shape)}"
                    )
                denorm_actions[action_name] = denorm_action

            pred_action_denorm = denorm_actions["pred"]
            gt_action_denorm = denorm_actions["gt"]

            if pred_action_denorm.shape != gt_action_denorm.shape:
                raise ValueError(
                    "Predicted action/GT action shape mismatch after denormalization: "
                    f"pred={tuple(pred_action_denorm.shape)} vs gt={tuple(gt_action_denorm.shape)}"
                )
            action_diff = pred_action_denorm - gt_action_denorm
            action_l1 = action_diff.abs().mean().item()
            action_l2 = action_diff.pow(2).mean().item()

        # 4. VAE reconstruction metrics against GT video
        gt_video_batch = video0.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype)
        vae_latents = model._encode_video_latents(gt_video_batch, tiled=False)
        vae_recon_video = model._decode_latents(vae_latents, tiled=False)
        vae_video_tensor = pil_frames_to_video_tensor(vae_recon_video)

        assert vae_video_tensor.shape == gt_video_tensor.shape, (
            "Eval VAE reconstruction/GT shape mismatch: "
            f"vae={tuple(vae_video_tensor.shape)} vs gt={tuple(gt_video_tensor.shape)}"
        )

        psnr_decode_vs_gt = video_psnr(pred=vae_video_tensor, target=gt_video_tensor)
        ssim_decode_vs_gt = video_ssim(pred=vae_video_tensor, target=gt_video_tensor)

        psnr_rollout_vs_decode = video_psnr(pred=pred_video_tensor, target=vae_video_tensor)
        ssim_rollout_vs_decode = video_ssim(pred=pred_video_tensor, target=vae_video_tensor)

        stitched_video_tensor = torch.cat(
            [pred_video_tensor, vae_video_tensor, gt_video_tensor],
            dim=2,
        ).contiguous()
        stitched_frames = []
        for t in range(stitched_video_tensor.shape[1]):
            frame = (stitched_video_tensor[:, t].permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).astype(np.uint8)
            stitched_frames.append(Image.fromarray(frame))

        video_path = os.path.join(
            self.eval_dir,
            f"step_{self.global_step:06d}_rank_{self.accelerator.process_index:03d}.mp4",
        )
        save_mp4(stitched_frames, video_path, fps=8)

        local_metrics = torch.tensor(
            [
                float(val_loss_sum),
                float(len(my_indices)),
                float(psnr_rollout_vs_gt),
                float(ssim_rollout_vs_gt),
                float(psnr_rollout_vs_decode),
                float(ssim_rollout_vs_decode),
                float(psnr_decode_vs_gt),
                float(ssim_decode_vs_gt),
                float(action_l2) if action_l2 is not None else -1.0,
                float(action_l1) if action_l1 is not None else -1.0,
            ],
            device=self.accelerator.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        gathered_metrics = self.accelerator.gather_for_metrics(local_metrics)
        # Column 0 is this rank's summed loss and column 1 its sample count, so
        # the validation loss is a true mean over every sample scored anywhere
        # rather than a mean of per-rank means.
        val_loss_mean = (
            gathered_metrics[:, 0].sum() / gathered_metrics[:, 1].sum().clamp(min=1.0)
        ).item()
        mean_metrics = gathered_metrics[:, 2:8].mean(dim=0)
        action_l2_mean = gathered_metrics[:, 8].mean().item() if action_l2 is not None else None
        action_l1_mean = gathered_metrics[:, 9].mean().item() if action_l1 is not None else None

        if was_dit_training:
            self._set_dit_only_train_mode()

        result = {
            "val_loss": float(val_loss_mean),
            "val_loss_n": int(gathered_metrics[:, 1].sum().item()),
            "psnr_rg": float(mean_metrics[0].item()),
            "ssim_rg": float(mean_metrics[1].item()),
            "psnr_rd": float(mean_metrics[2].item()),
            "ssim_rd": float(mean_metrics[3].item()),
            "psnr_dg": float(mean_metrics[4].item()),
            "ssim_dg": float(mean_metrics[5].item()),
            "video_path": video_path,
        }
        if action_l2_mean is not None:
            result["action_l2"] = float(action_l2_mean)
        if action_l1_mean is not None:
            result["action_l1"] = float(action_l1_mean)
        return result

    def _save_weights_checkpoint(self, step_tag: str):
        model = self.accelerator.unwrap_model(self.model)
        ckpt_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        # A LoRA run folds its adapters back into the base weights on the way
        # out, so every arm writes the same key set and the policy server needs
        # no LoRA support of its own.
        mot_state_dict = lora_utils.merged_state_dict(model.mot) if self.lora_rank else None
        model.save_checkpoint(
            ckpt_path, optimizer=None, step=self.global_step, mot_state_dict=mot_state_dict
        )
        return ckpt_path

    def _save_trainer_state(self, state_path: str):
        state_file = os.path.join(state_path, "trainer_state.json")
        payload = {
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
            "batch_in_epoch": int(self.batch_in_epoch),
        }
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"

        self.accelerator.wait_for_everyone()
        ckpt_path = None
        if self.accelerator.is_main_process:
            ckpt_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        state_path = os.path.join(self.state_dir, step_tag)
        ensure_dir(state_path)
        self.accelerator.save_state(output_dir=state_path)
        if self.accelerator.is_main_process:
            self._save_trainer_state(state_path)
        self.accelerator.wait_for_everyone()

        # FASTWAM_KEEP_STATES=N keeps only the N newest state dirs (resume
        # needs one; a full DS ZeRO-2 state is tens of GB per save, and
        # nothing else ever reads an older one -- the per-step weights .pt
        # files are what curve evaluation uses). 0/unset keeps everything,
        # the upstream behaviour.
        keep = int(os.environ.get("FASTWAM_KEEP_STATES", "0") or 0)
        if keep > 0 and self.accelerator.is_main_process:
            import shutil

            states = sorted(
                d for d in os.listdir(self.state_dir)
                if d.startswith("step_") and os.path.isdir(os.path.join(self.state_dir, d))
            )
            for stale in states[:-keep]:
                shutil.rmtree(os.path.join(self.state_dir, stale), ignore_errors=True)
                logger.info("[ckpt] pruned old state %s (FASTWAM_KEEP_STATES=%d)", stale, keep)
        self.accelerator.wait_for_everyone()

        return {"weights_path": ckpt_path, "state_path": state_path}

    def load_training_state(self, state_dir: str):
        self.accelerator.load_state(input_dir=state_dir)
        state_file = Path(state_dir) / "trainer_state.json"
        if state_file.exists():
            with open(state_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
            self.global_step = int(payload["global_step"])

            if "epoch" in payload and "batch_in_epoch" in payload:
                self.epoch = int(payload["epoch"])
                self.batch_in_epoch = int(payload["batch_in_epoch"])
                self.train_sampler.set_epoch_offset(self.epoch)
                self.train_sampler.set_resume_batch_offset(self.batch_in_epoch)
                logger.info(
                    "Restored dataloader progress: epoch=%d batch_in_epoch=%d sample_offset=%d",
                    self.epoch,
                    self.batch_in_epoch,
                    self.batch_in_epoch * self.batch_size * self.accelerator.num_processes,
                )
            else:
                self.epoch = 0
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                logger.warning(
                    "State file does not contain `epoch`/`batch_in_epoch`; "
                    "optimizer/scheduler were restored, but dataloader progress resume is skipped."
                )
            self.accelerator.wait_for_everyone()
            return

        match = re.search(r"step[_-](\d+)$", str(state_dir).rstrip("/"))
        if match:
            self.global_step = int(match.group(1))
        else:
            self.global_step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.train_sampler.clear_resume_batch_offset()
        self.accelerator.wait_for_everyone()
        logger.info("Loaded accelerate training state from %s at step=%d", state_dir, self.global_step)
        logger.warning(
            "State file `%s` is missing; dataloader progress resume is skipped.",
            state_file,
        )

    @staticmethod
    def _first_nonfinite(named_tensors):
        """Name of the first non-finite tensor in the sequence, or None."""
        for name, tensor in named_tensors:
            if tensor is None or not torch.is_floating_point(tensor):
                continue
            if not torch.isfinite(tensor).all():
                return name
        return None

    def _debug_nonfinite(self, stage, sample=None, loss=None):
        """Report the first non-finite tensor at one point in the step.

        `stage` is where we are looking: "input" before the forward, "loss"
        after it, "grad" after the backward, "param" after the optimizer step.
        The first stage to fire is the diagnosis -- a bad batch, an overflowing
        backward and a poisoned update are three different bugs that produce the
        same nan loss one step later.
        """
        if not self.debug_nonfinite:
            return
        found = None
        if stage == "input" and sample is not None:
            items = getattr(sample, "items", None)
            if items is None:
                return
            found = self._first_nonfinite(
                (k, v) for k, v in items() if torch.is_tensor(v)
            )
        elif stage == "loss" and loss is not None:
            found = "loss" if not torch.isfinite(loss.detach()).all() else None
        elif stage == "grad":
            found = self._first_nonfinite(
                (n, p.grad) for n, p in self.model.named_parameters() if p.grad is not None
            )
        elif stage == "param":
            found = self._first_nonfinite(self.model.named_parameters())
        if found is None:
            return
        key = (stage, found)
        if key in self._debug_reported:
            return
        self._debug_reported.add(key)
        logger.error(
            "[nonfinite] first non-finite %s at step=%d batch_in_epoch=%d rank=%d: %s",
            stage,
            self.global_step,
            self.batch_in_epoch,
            self.accelerator.process_index,
            found,
        )

    def train(self):
        self._set_dit_only_train_mode()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        if self.max_steps is None:
            raise ValueError("`max_steps` must be set before entering the while-step training loop.")

        logger.info("Starting training with max_steps=%d.", self.max_steps)
        data_iter = iter(self.train_loader)
        self.run_start_step = self.global_step
        self.run_start_time = time.perf_counter()

        while self.global_step < self.max_steps:
            try:
                sample = next(data_iter)
                self.batch_in_epoch += 1
            except StopIteration:
                self.epoch += 1
                self.batch_in_epoch = 0
                self.train_sampler.clear_resume_batch_offset()
                data_iter = iter(self.train_loader)
                continue

            with self.accelerator.accumulate(self.model):
                train_model = self.model if hasattr(self.model, "training_loss") else self.accelerator.unwrap_model(self.model)

                self._debug_nonfinite("input", sample=sample)
                with self.accelerator.autocast():
                    loss, loss_dict = train_model.training_loss(sample)
                self._debug_nonfinite("loss", loss=loss)
                self.accelerator.backward(loss)
                self._debug_nonfinite("grad")

                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self._debug_nonfinite("param")
                    if not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.global_step += 1
                    global_loss = float(
                        self.accelerator.gather(loss.detach().float().reshape(1)).mean().item()
                    )
                    # Stop on a run that has gone non-finite instead of burning
                    # the walltime on it. Job 2129235 trained 60 steps at
                    # loss=nan and only the log said so; a 40k-step run would
                    # have spent a day writing nan checkpoints. One nan can be a
                    # single bad batch, so this waits for a run of them.
                    if math.isfinite(global_loss):
                        self._nonfinite_streak = 0
                    else:
                        self._nonfinite_streak = getattr(self, "_nonfinite_streak", 0) + 1
                        if 0 < self.max_nonfinite_steps <= self._nonfinite_streak:
                            raise RuntimeError(
                                f"training loss has been non-finite for "
                                f"{self._nonfinite_streak} consecutive optimizer steps "
                                f"(step={self.global_step}); aborting. Set "
                                f"max_nonfinite_steps=0 to disable this check."
                            )
                    global_loss_metrics = {}
                    for key, value in loss_dict.items():
                        metric_tensor = torch.tensor(float(value), device=loss.device, dtype=torch.float32).reshape(1)
                        global_loss_metrics[key] = float(
                            self.accelerator.gather(metric_tensor).mean().item()
                        )
                    grad_norm_tensor = torch.tensor(grad_norm, device=loss.device, dtype=torch.float32)
                    global_grad_norm = float(self.accelerator.gather(grad_norm_tensor).mean().item())

                    current_lr = float(self.optimizer.param_groups[0]["lr"])

                    if self.log_every > 0 and self.global_step % self.log_every == 0 and self.accelerator.is_main_process:
                        eta_str, steps_per_sec = self._estimate_eta()
                        description = "[train] epoch=%d step=%d/%d loss=%.4f " % (
                            self.epoch,
                            self.global_step,
                            self.max_steps,
                            global_loss,
                        )
                        if global_loss_metrics:
                            detail_str = " ".join([f"{k}={v:.4f}" for k, v in sorted(global_loss_metrics.items())])
                            description += detail_str + " "
                        description += "lr=%.2e speed=%.2f step/s, %.2f samples/s eta=%s" % (
                            current_lr,
                            steps_per_sec,
                            steps_per_sec * self.batch_size * self.accelerator.num_processes,
                            eta_str,
                        )
                        logger.info(description)

                        wandb_payload = {
                            "train/loss": global_loss,
                            "train/grad_norm": global_grad_norm,
                            "train/lr": current_lr,
                            "performance/steps_per_sec": steps_per_sec,
                            "performance/samples_per_sec": steps_per_sec * self.batch_size * self.accelerator.num_processes,
                        }
                        for key, value in global_loss_metrics.items():
                            wandb_payload[f"train/{key}"] = value
                        self._wandb_log(wandb_payload)

                    if (
                        self.eval_every > 0
                        and self.val_dataset is not None
                        and self.global_step % self.eval_every == 0
                    ):
                        metrics = self.evaluate()
                        self.accelerator.wait_for_everyone()
                        if metrics is not None and self.accelerator.is_main_process:
                            description = "[eval] step=%d val_loss=%.4f infer_psnr=%.4f infer_ssim=%.4f" % (
                                self.global_step,
                                metrics["val_loss"],
                                metrics["psnr_rd"],
                                metrics["ssim_rd"],
                            )
                            if "action_l2" in metrics:
                                description += " action_l2=%.4f" % metrics["action_l2"]
                            if "action_l1" in metrics:
                                description += " action_l1=%.4f" % metrics["action_l1"]
                            logger.info(description)
                            eval_payload = {
                                "eval/val_loss": float(metrics["val_loss"]),
                                "eval/psnr_rg": float(metrics["psnr_rg"]),
                                "eval/ssim_rg": float(metrics["ssim_rg"]),
                                "eval/psnr_rd": float(metrics["psnr_rd"]),
                                "eval/ssim_rd": float(metrics["ssim_rd"]),
                                "eval/psnr_dg": float(metrics["psnr_dg"]),
                                "eval/ssim_dg": float(metrics["ssim_dg"]),
                                # How many held-out samples the loss averaged.
                                # Logged so a curve can never again be read as
                                # if it were a full-split number when it is not.
                                "eval/val_loss_n": int(metrics["val_loss_n"]),
                            }
                            if "action_l2" in metrics:
                                eval_payload["eval/action_l2"] = float(metrics["action_l2"])
                            if "action_l1" in metrics:
                                eval_payload["eval/action_l1"] = float(metrics["action_l1"])
                            self._wandb_log(eval_payload)

                    if self.save_every > 0 and self.global_step % self.save_every == 0:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[ckpt] step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )

                    if self.global_step >= self.max_steps:
                        ckpt_info = self.save_checkpoint()
                        if self.accelerator.is_main_process:
                            logger.info(
                                "[done] max_steps reached step=%d weights=%s state=%s",
                                self.global_step,
                                ckpt_info["weights_path"],
                                ckpt_info["state_path"],
                            )
                        return

        ckpt_info = self.save_checkpoint()
        if self.accelerator.is_main_process:
            logger.info(
                "[done] training finished step=%d weights=%s state=%s",
                self.global_step,
                ckpt_info["weights_path"],
                ckpt_info["state_path"],
            )
        
