# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""LingBot-VA on MHBench: one shared multitask policy, decentralized.

Not upstream. What differs from `va_robotwin30_train_cfg`:

  action_dim      30 (dual-arm EEF/joint/gripper) -> 35, MHBench's
                  [31 joint targets, pelvis height, base velocity xyz]. The two
                  projections bound to that width are reinitialised; see
                  `baselines/scripts/prepare_lingbot_init_ckpt.py`.
  obs_cam_keys    three RoboTwin cameras -> the one ego view each agent has,
                  at 224x320 rather than a 256 square (see LINGBOT_LATENT_SIZE).
  action_per_frame 12 -> 20: MHBench records at 50 fps and the latents are
                  sampled at 10, so a latent frame spans 4 sampled frames of
                  5 raw actions each.
  norm_stat       read from the prepared dataset (or LINGBOT_NORM_STAT), not
                  baked in -- it is RoboTwin's arm ranges upstream.

Everything here is settable from the environment because upstream's entry
point takes no config overrides (`wan_va.train` parses only --config-name and
--save-root).
"""
import json
import os

from easydict import EasyDict

from .shared_config import va_shared_cfg

_POLICY_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def _env_int(name, default):
    value = os.environ.get(name, "")
    return int(value) if value else default


def _env_float(name, default):
    value = os.environ.get(name, "")
    return float(value) if value else default


def _load_norm_stat(dataset_path, action_dim):
    """The action quantiles the head normalises against.

    Written by process_data.py into the prepared dataset and copied next to
    every checkpoint; LINGBOT_NORM_STAT points at either. There is no sensible
    default -- upstream's baked-in table is RoboTwin's arm ranges -- so the
    placeholder below is tagged and both the trainer and the adapter refuse it.
    """
    path = os.environ.get("LINGBOT_NORM_STAT") or (
        os.path.join(dataset_path, "meta", "lingbot_norm_stat.json")
        if dataset_path else "")
    if path and os.path.exists(path):
        stat = json.load(open(path))
        for key in ("q01", "q99"):
            if len(stat[key]) != action_dim:
                raise ValueError(
                    f"{path}: {key} has {len(stat[key])} entries, action_dim is {action_dim}")
        return {"q01": stat["q01"], "q99": stat["q99"]}, path
    return {"q01": [-1.0] * action_dim, "q99": [1.0] * action_dim}, ""


va_mhbench_cfg = EasyDict(__name__="Config: VA mhbench")
va_mhbench_cfg.update(va_shared_cfg)

va_mhbench_cfg.infer_mode = "server"
va_mhbench_cfg.wan22_pretrained_model_name_or_path = (
    os.environ.get("LINGBOT_VA_BASE_MODEL_PATH")
    or os.path.join(_POLICY_ROOT, ".merged_ckpt"))
va_mhbench_cfg.enable_offload = True

va_mhbench_cfg.attn_window = _env_int("LINGBOT_ATTN_WINDOW", 72)
va_mhbench_cfg.frame_chunk_size = _env_int("LINGBOT_FRAME_CHUNK_SIZE", 2)
va_mhbench_cfg.env_type = "none"

# The canvas the latents were encoded at, and the one serving resizes to.
# 224x320: the ego view is 240x320 and both sides must be a multiple of 32 so
# the VAE's stride-16 grid stays even under the model's 2x2 patching.
_size = os.environ.get("LINGBOT_LATENT_SIZE", "224x320")
_h, _w = (int(v) for v in (_size.split("x") if "x" in _size else (_size, _size)))
va_mhbench_cfg.height = _h
va_mhbench_cfg.width = _w
va_mhbench_cfg.action_dim = _env_int("LINGBOT_ACTION_DIM", 35)
va_mhbench_cfg.action_per_frame = _env_int("LINGBOT_ACTION_PER_FRAME", 20)

va_mhbench_cfg.obs_cam_keys = os.environ.get(
    "LINGBOT_OBS_CAM_KEYS", "observation.images.ego").split(",")

va_mhbench_cfg.guidance_scale = _env_float("LINGBOT_GUIDANCE_SCALE", 5)
va_mhbench_cfg.action_guidance_scale = _env_float("LINGBOT_ACTION_GUIDANCE_SCALE", 1)

va_mhbench_cfg.num_inference_steps = _env_int("LINGBOT_NUM_INFERENCE_STEPS", 25)
va_mhbench_cfg.video_exec_step = -1
va_mhbench_cfg.action_num_inference_steps = _env_int(
    "LINGBOT_ACTION_NUM_INFERENCE_STEPS", 50)

va_mhbench_cfg.snr_shift = 5.0
va_mhbench_cfg.action_snr_shift = 1.0

va_mhbench_cfg.used_action_channel_ids = list(range(va_mhbench_cfg.action_dim))
va_mhbench_cfg.inverse_used_action_channel_ids = list(range(va_mhbench_cfg.action_dim))

va_mhbench_cfg.action_norm_method = "quantiles"

va_mhbench_train_cfg = EasyDict(__name__="Config: VA mhbench train")
va_mhbench_train_cfg.update(va_mhbench_cfg)

va_mhbench_train_cfg.dataset_path = os.environ.get("LINGBOT_VA_DATASET_PATH", "")
va_mhbench_train_cfg.empty_emb_path = os.path.join(
    va_mhbench_train_cfg.dataset_path, "empty_emb.pt")
va_mhbench_cfg.norm_stat, va_mhbench_cfg.norm_stat_source = _load_norm_stat(
    va_mhbench_train_cfg.dataset_path, va_mhbench_cfg.action_dim)
va_mhbench_train_cfg.norm_stat = va_mhbench_cfg.norm_stat
va_mhbench_train_cfg.norm_stat_source = va_mhbench_cfg.norm_stat_source

va_mhbench_train_cfg.enable_wandb = os.environ.get("LINGBOT_WANDB", "1") != "0"
va_mhbench_train_cfg.load_worker = _env_int("LINGBOT_NUM_WORKERS", 8)
# One LeRobot tree, so no pool: forking one from a process that already holds a
# CUDA context is what hung the first probe for two hours.
va_mhbench_train_cfg.dataset_init_workers = _env_int("LINGBOT_DATASET_INIT_WORKERS", 1)
va_mhbench_train_cfg.save_interval = _env_int("LINGBOT_SAVE_INTERVAL", 2500)
va_mhbench_train_cfg.gc_interval = 50
va_mhbench_train_cfg.cfg_prob = _env_float("LINGBOT_CFG_PROB", 0.1)

va_mhbench_train_cfg.learning_rate = _env_float("LINGBOT_LR", 1e-5)
va_mhbench_train_cfg.beta1 = 0.9
va_mhbench_train_cfg.beta2 = 0.95
va_mhbench_train_cfg.weight_decay = 0.1
va_mhbench_train_cfg.warmup_steps = _env_int("LINGBOT_WARMUP_STEPS", 10)

# Global batch 32, the benchmark's shared budget. The step count is NOT the
# benchmark's 40,000 and that is deliberate: one sample here is one whole
# episode (52 latent frames, ~9,900 tokens), where GR00T, pi0.5 and FastWAM
# take one action chunk, so 32 x 40,000 is 1.28M episode-samples -- measured at
# 144 s/step on one 3090, i.e. 67 days, and about 5.5x upstream's own
# post-training compute. 10,000 steps x 32 = 320,000 samples is exactly
# upstream's post-training budget (64 x 5,000). Same kind of exception, and the
# same arithmetic, as DreamZero's batch 8 x 8,000.
_world = _env_int("WORLD_SIZE", 1)
va_mhbench_train_cfg.global_batch_size = _env_int("LINGBOT_GLOBAL_BATCH", 32)
va_mhbench_train_cfg.batch_size = _env_int("LINGBOT_BATCH_SIZE", 1)
_derived_accum = max(
    1,
    va_mhbench_train_cfg.global_batch_size // (va_mhbench_train_cfg.batch_size * _world))
va_mhbench_train_cfg.gradient_accumulation_steps = _env_int(
    "LINGBOT_GRAD_ACCUM", _derived_accum)
va_mhbench_train_cfg.num_steps = _env_int("LINGBOT_MAX_STEPS", 10000)

# LoRA arm (not upstream). rank 0 = the paper's full fine-tune.
va_mhbench_train_cfg.lora_rank = _env_int("LINGBOT_LORA_RANK", 32)
va_mhbench_train_cfg.lora_alpha = _env_float("LINGBOT_LORA_ALPHA", 16)
va_mhbench_train_cfg.lora_dropout = _env_float("LINGBOT_LORA_DROPOUT", 0.1)
va_mhbench_train_cfg.train_action_condition = os.environ.get(
    "LINGBOT_TRAIN_ACTION_COND", "0") == "1"
va_mhbench_train_cfg.resume_from = os.environ.get("LINGBOT_RESUME_FROM", "")
