"""Validation callback for DreamZero.

Self-scheduled (fires every ``eval_steps`` on ``on_step_end``; we do NOT use HF's
eval loop because the streaming val dataset reports an ~infinite ``__len__`` that
would hang ``evaluation_loop``). For a FIXED val subset it runs the full denoise
inference (``model.lazy_joint_video_action``) and logs:

  - ``val/hand_l1``   : L1 on the hand_joints dims (first hand_dim), PHYSICAL units
  - ``val/action_l1`` : L1 on the body_token dims (after hand), PHYSICAL units
  - ``val/pred_video``: the model's predicted (VAE-decoded) video (to wandb)

Both L1s are unnormalized to physical units (invert the [-1,1] min_max norm).

These are the *denoised* metrics (16-step inference), not the teacher-forcing
flow loss that ``compute_loss`` reports. The subset is pulled once and reused at
every eval so the metric is comparable across steps.
"""
import torch
from transformers import TrainerCallback


class DenoiseValCallback(TrainerCallback):
    def __init__(
        self,
        val_dataloader,
        eval_steps: int = 500,
        hand_dim: int = 14,
        num_val_batches: int = 1,
        num_videos: int = 1,
        video_fps: int = 8,
        action_min=None,
        action_max=None,
    ):
        self.val_dataloader = val_dataloader
        self.eval_steps = max(1, int(eval_steps))
        self.hand_dim = hand_dim
        self.num_val_batches = num_val_batches
        self.num_videos = num_videos
        self.video_fps = video_fps
        # Per-dim action min/max (concat order, [A]) for reporting L1 in PHYSICAL
        # units. min_max normalization maps physical -> [-1,1]; we invert it with
        # x_phys = (x_norm + 1)/2 * (max - min) + min before the L1.
        self.action_min = action_min
        self.action_max = action_max
        # A FIXED val subset: pulled once (kept on CPU) and reused at every eval so
        # the metric is comparable across steps (same samples each time), instead of
        # streaming fresh random samples on every eval.
        self._fixed_batches = None
        self._metric_defined = False  # wandb define_metric done once

    @torch.no_grad()
    def on_step_end(self, args, state, control, model=None, **kwargs):
        # Eval at step 1 (initial baseline before the model has moved), then every
        # eval_steps after that.
        if state.global_step <= 0:
            return
        if state.global_step != 1 and state.global_step % self.eval_steps != 0:
            return
        if self.val_dataloader is None or model is None:
            return
        core = getattr(model, "module", model)  # unwrap deepspeed engine
        was_training = core.training
        core.eval()
        device = next(core.parameters()).device

        # This action head only exposes the *causal* rollout
        # (lazy_joint_video_action), which carries KV-cache state across calls via
        # action_head.current_start_frame / .language. We reset it before each batch
        # so every eval is a fresh, comparable denoise and training state is untouched.
        head = getattr(core, "action_head", None)

        def _reset_causal_state():
            if head is None:
                return
            # create-or-reset (these are read by the causal rollout)
            head.current_start_frame = 0
            head.language = None
            # trt_engine is only created by the inference-only post_initialize()
            # (model.to(cuda)+torch.compile), which training never calls. The eager
            # denoise path still checks `self.trt_engine is not None`, so the attr must
            # exist. None => plain eager denoise (no TensorRT).
            if not hasattr(head, "trt_engine"):
                head.trt_engine = None

        # Build the fixed subset once (first eval), cache on CPU, reuse forever.
        if self._fixed_batches is None:
            self._fixed_batches = []
            for i, batch in enumerate(self.val_dataloader):
                if i >= self.num_val_batches:
                    break
                self._fixed_batches.append(
                    {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in batch.items()}
                )
            if getattr(state, "is_world_process_zero", True):
                print(f"[DenoiseValCallback] cached fixed val subset: "
                      f"{len(self._fixed_batches)} batch(es), reused every eval", flush=True)

        hand_l1s, action_l1s, videos = [], [], []
        for i, cpu_batch in enumerate(self._fixed_batches):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cpu_batch.items()}
            _reset_causal_state()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = core.lazy_joint_video_action(batch)
                _, action_inputs = core.prepare_input(batch)

            action_pred = out["action_pred"].float()
            gt = action_inputs.action.float()
            # collapse any leading chunk dim so both are [B, H, A]
            if action_pred.dim() == 4:
                action_pred = action_pred.reshape(action_pred.shape[0], -1, action_pred.shape[-1])
            if gt.dim() == 4:
                gt = gt.reshape(gt.shape[0], -1, gt.shape[-1])
            h = min(action_pred.shape[1], gt.shape[1])
            a = min(action_pred.shape[-1], gt.shape[-1])
            ap, ga = action_pred[:, :h, :a], gt[:, :h, :a]
            hd = min(self.hand_dim, a)
            # Report L1 in PHYSICAL units: invert the [-1,1] min_max normalization.
            if self.action_min is not None:
                mn = self.action_min[:a].to(device=ap.device, dtype=ap.dtype)
                mx = self.action_max[:a].to(device=ap.device, dtype=ap.dtype)
                ap = (ap + 1.0) / 2.0 * (mx - mn) + mn
                ga = (ga + 1.0) / 2.0 * (mx - mn) + mn
            # hand_l1 = hand_joints (first hand_dim); action_l1 = body_token only
            # (dims after hand), NOT hand+token combined.
            hand_l1s.append((ap[..., :hd] - ga[..., :hd]).abs().mean().item())
            action_l1s.append((ap[..., hd:] - ga[..., hd:]).abs().mean().item())

            if len(videos) < self.num_videos:
                videos.append(self._decode_pred_video(core, out["video_pred"], state))

        # Aggregate across ALL ranks so the metric is the mean over
        # world_size * num_val_batches samples (e.g. 8 gpus x 20 = 160), not just
        # rank0's. all_reduce is a collective: every rank must reach it, and they do
        # (all ran the same num_val_batches denoise steps above -> balanced).
        hand, action, n = self._reduce(hand_l1s, action_l1s, device)

        if getattr(state, "is_world_process_zero", True):
            self._log(state, hand, action, n, videos)

        _reset_causal_state()  # leave a clean slate for training
        if was_training:
            core.train()

    @staticmethod
    def _reduce(hand_l1s, action_l1s, device):
        import torch.distributed as dist
        hand_sum, action_sum, cnt = float(sum(hand_l1s)), float(sum(action_l1s)), len(hand_l1s)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            t = torch.tensor([hand_sum, action_sum, cnt], dtype=torch.float64, device=device)
            dist.all_reduce(t)  # sum over ranks
            hand_sum, action_sum, cnt = t[0].item(), t[1].item(), int(t[2].item())
        if cnt == 0:
            return None, None, 0
        return hand_sum / cnt, action_sum / cnt, cnt

    def _log(self, state, hand, action, n, videos):
        # Always print to stdout so the metric is visible in training logs even
        # when wandb is disabled (report_to=none); wandb is just an extra sink.
        if hand is not None:
            print(f"[DenoiseValCallback] step {state.global_step}: "
                  f"val/hand_l1={hand:.5f} val/action_l1={action:.5f} "
                  f"(n={n} samples over all ranks, {len(videos)} video(s))", flush=True)
        import wandb
        if wandb.run is None:
            return
        # Give val/* their OWN x-axis = the real training step. wandb's internal
        # `_step` counts every log call (loss/grad/lr/system metrics) and runs ~2x
        # ahead of global_step; logging val against it (or an explicit step=) mis-
        # places the points. define_metric decouples val from _step entirely.
        if not self._metric_defined:
            wandb.define_metric("val/step")
            wandb.define_metric("val/*", step_metric="val/step")
            self._metric_defined = True
        log = {"val/step": int(state.global_step)}
        if hand is not None:
            log["val/hand_l1"] = hand
            log["val/action_l1"] = action
        if videos:
            log["val/pred_video"] = wandb.Video(
                self._to_wandb_video(videos[0]), fps=self.video_fps, format="mp4"
            )
        wandb.log(log)  # no explicit step; val/* plotted against val/step (=global_step)

    @staticmethod
    def _decode_pred_video(core, latent, state):
        """out["video_pred"] is a denoised VAE *latent* (16-ch), NOT pixels. Decode it
        to RGB frames via the VAE. Prints shapes once so the layout is verifiable."""
        vae = core.action_head.vae
        p = next(vae.parameters())
        if getattr(state, "is_world_process_zero", True):
            print(f"[DenoiseValCallback] video_pred latent shape={tuple(latent.shape)}", flush=True)
        px = vae.decode(latent.to(device=p.device, dtype=p.dtype))  # [B,3,T,H,W] in [-1,1]
        if getattr(state, "is_world_process_zero", True):
            print(f"[DenoiseValCallback] decoded pixels shape={tuple(px.shape)}", flush=True)
        return px[0].detach().float().cpu()

    @staticmethod
    def _to_wandb_video(v):
        """tensor [C,T,H,W] or [T,C,H,W], range [-1,1] or [0,1] -> uint8 [T,C,H,W].

        wandb.Video wants CHANNEL-FIRST (time, channels, height, width); passing
        channel-last silently produces a garbled clip. Raises (never silently drops)
        on an unexpected shape so a broken pred_video surfaces instead of vanishing.
        """
        import numpy as np
        v = v.numpy()
        assert v.ndim == 4, f"pred_video must be 4-D ([C,T,H,W] or [T,C,H,W]), got {v.shape}"
        if v.shape[1] == 3:        # already [T, C, H, W]
            pass
        elif v.shape[0] == 3:      # [C, T, H, W] -> [T, C, H, W]
            v = np.transpose(v, (1, 0, 2, 3))
        else:
            raise ValueError(f"pred_video has no channel axis of size 3: shape {v.shape}")
        if v.min() < 0:
            v = (v + 1.0) / 2.0
        v = np.clip(v, 0.0, 1.0)
        return (v * 255).astype(np.uint8)  # [T, C, H, W] for wandb.Video
