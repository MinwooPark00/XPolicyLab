import os
# Set working directory to project root
os.chdir('/home/jliu/we_learn')
os.environ['PWD'] = '/home/jliu/we_learn'

import dotenv
dotenv.load_dotenv('/home/jliu/we_learn/.env')

import torch
import numpy as np
from pathlib import Path
from safetensors.torch import load_file
from we.utils import parse_args_to_tyro_config, seed_everything, move_to_device, batchify
from we.config.config import LaunchConfig
from we.config.data import LerobotDataConfig
from we.config.model import ACT_ModelConfig
from we.learn.models.act import ACTConfig, ACTPolicy

# ============ Configuration ============
ckpt_step = 50000
run_dir = Path(".runs/act/act.vlt.cosin.lr1.0e-04.b256.gpus4.2601111414")

# ============ Load Config ============
launch_config: LaunchConfig = parse_args_to_tyro_config(run_dir / "argv.txt")  # type: ignore
conf = (run_dir / "run_config.json").open("r").read()
config = launch_config.model_validate_json(conf)

data_cfg: LerobotDataConfig = config.data  # type: ignore
model_cfg: ACT_ModelConfig = config.model  # type: ignore

seed_everything(config.seed or 42)

# ============ Device Setup ============
DEVICE = "cuda:0"
print(f"Using device: {DEVICE}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")

# ============ Load Model ============
def load_model(model_cfg: ACT_ModelConfig, run_dir: Path, ckpt_step: int | str = "latest"):
    ckpt_path = run_dir / "checkpoints" / f"ckpt_{ckpt_step}" / "model.safetensors"
    if not ckpt_path.exists():
        # Try .pth format
        ckpt_path = run_dir / "checkpoints" / f"ckpt_{ckpt_step}.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    act_config = ACTConfig(
        n_obs_steps=model_cfg.n_obs_steps,
        chunk_size=model_cfg.chunk_size,
        n_action_steps=model_cfg.n_action_steps,
        action_dim=model_cfg.action_dim,
        state_dim=model_cfg.state_dim,
        dim_model=model_cfg.dim_model,
        n_heads=model_cfg.n_heads,
        dim_feedforward=model_cfg.dim_feedforward,
        feedforward_activation=model_cfg.feedforward_activation,
        n_encoder_layers=model_cfg.n_encoder_layers,
        n_decoder_layers=model_cfg.n_decoder_layers,
        pre_norm=model_cfg.pre_norm,
        dropout=model_cfg.dropout,
        use_vae=model_cfg.use_vae,
        latent_dim=model_cfg.latent_dim,
        n_vae_encoder_layers=model_cfg.n_vae_encoder_layers,
        kl_weight=model_cfg.kl_weight,
        temporal_ensemble_coeff=model_cfg.temporal_ensemble_coeff,
    )
    
    model = ACTPolicy(config=act_config, dataset_stats=None)
    
    print(f"Loading checkpoint from {ckpt_path}")
    if ckpt_path.suffix == ".safetensors":
        state_dict = load_file(ckpt_path)
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model

model = load_model(model_cfg, run_dir, ckpt_step)
model = model.to(DEVICE)
model.eval()
print("Model loaded successfully!")

num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters (in millions): {num_params * 1e-6:.3f}")

# ============ Load Dataset ============
transform_kwargs = {
    "no_aug": True,  # No augmentation for evaluation
}

train_dataset = data_cfg(split="train", transform_kwargs=transform_kwargs)
print(f"Train dataset size: {len(train_dataset)}")

val_dataset = data_cfg(split="val", transform_kwargs=transform_kwargs)
print(f"Validation dataset size: {len(val_dataset)}")

# ============ Load Normalization Stats ============
maxmin = data_cfg.transform.action_state

# ============ Evaluation on One Episode ============
np.set_printoptions(precision=4, suppress=True)

dataset = train_dataset

start_frame_idx = train_dataset.raw_dataset.base_dataset.episode_data_index["from"][515].item()
# tensor(16261)
end_frame_idx = train_dataset.raw_dataset.base_dataset.episode_data_index["to"][515].item()
# tensor(16301)

print(f"\n{'='*50}")
print(f"Frame range: {start_frame_idx} - {end_frame_idx}")
print(f"Episode length: {end_frame_idx - start_frame_idx} frames")
print(f"{'='*50}\n")

# Collect all action errors for RMSE calculation (same as evaluate)
all_action_l1_errs = []  # Will be (N, Da) where N = num_frames * Tp

l2_xyz_errors = []
l2_rpy_errors = []
l2_gripper_errors = []

for i in range(start_frame_idx, end_frame_idx):
    batch = dataset[i]
    batch = move_to_device(batchify(batch), DEVICE)
    
    gt_actions = batch["action"]  # (B, Tp, Da)
    B, Tp, Da = gt_actions.shape
    
    # ACT dataset uses "observation.images" and "observation.state" keys
    images = batch["observation.images"]  # (B, N*To, C, H, W)
    if len(images.shape) == 5:
        # (B, N*To, C, H, W) -> keep batch dim for ACT
        pass
    
    states = batch["observation.state"]  # (B, state_dim) or (B, obs_horizon, state_dim)
    if len(states.shape) == 3:
        states = states.squeeze(1)  # (B, 1, S) -> (B, S)
    
    # Run inference with ACT model
    # ACT expects batch dict with "observation.images" and "observation.state"
    with torch.inference_mode():
        # Prepare batch dict for ACT model (already in correct format)
        act_batch = {
            "observation.images": images,  # (B, N*To, C, H, W)
            "observation.state": states,  # (B, S)
        }
        
        # Use predict_action for full chunk prediction
        pred_actions = model.predict_action(act_batch)
    
    # pred_actions is tensor (B, Tp, Da) or (Tp, Da)
    if len(pred_actions.shape) == 3:
        pred_actions = pred_actions[0]  # (Tp, Da)
    
    # Convert to numpy
    pred_actions = pred_actions.cpu().numpy()
    
    # Denormalize actions
    denorm_gt_actions = maxmin.denormalize(gt_actions[0]).cpu().numpy()  # (Tp, Da)
    denorm_pred_actions = maxmin.denormalize(torch.from_numpy(pred_actions)).numpy()  # (Tp, Da)
    
    # Only compare the first action step for fair comparison with act_client
    # (act_client only has single-step gt_action from raw dataset)
    action_l1_errs = np.abs(denorm_pred_actions[0] - denorm_gt_actions[0])  # (Da,)
    all_action_l1_errs.append(action_l1_errs[np.newaxis, :])
    
    # Also compute per-frame average for display
    avg_action_errors = action_l1_errs  # (Da,)
    
    # Split into xyz, rpy, gripper
    labels_denormed = [
        "denorm_err_l1_xyz",
        "denorm_err_l1_rpy",
        "denorm_err_l1_gripper",
    ]
    avg_l1_action_err = np.split(avg_action_errors, [3, 6], axis=-1)
    metric = {**dict(zip(labels_denormed, map(np.linalg.norm, avg_l1_action_err)))}
    
    print(metric)
    l2_xyz_errors.append(metric["denorm_err_l1_xyz"])
    l2_rpy_errors.append(metric["denorm_err_l1_rpy"])
    l2_gripper_errors.append(metric["denorm_err_l1_gripper"])

# ============ Summary Statistics ============
print(f"\n{'='*50}")
print("Episode Summary")
print(f"{'='*50}")

l2_xyz_errors = np.array(l2_xyz_errors)
l2_rpy_errors = np.array(l2_rpy_errors)
l2_gripper_errors = np.array(l2_gripper_errors)

from we.utils import rmse

print(f"XYZ Error:     Mean={np.mean(l2_xyz_errors):.4f} ± {np.std(l2_xyz_errors):.4f}, RMSE={rmse(l2_xyz_errors):.4f}")
print(f"RPY Error:     Mean={np.mean(l2_rpy_errors):.4f} ± {np.std(l2_rpy_errors):.4f}, RMSE={rmse(l2_rpy_errors):.4f}")
print(f"Gripper Error: Mean={np.mean(l2_gripper_errors):.4f} ± {np.std(l2_gripper_errors):.4f}, RMSE={rmse(l2_gripper_errors):.4f}")

print(f"\nTotal frames evaluated: {len(l2_xyz_errors)}")

# ============ Quality Assessment ============
print(f"\n{'='*50}")
print("Quality Assessment")
print(f"{'='*50}")

xyz_mean = np.mean(l2_xyz_errors)
rpy_mean = np.mean(l2_rpy_errors)

# Position error assessment (in meters)
if xyz_mean < 0.01:
    xyz_quality = "✅ Good (< 1cm)"
elif xyz_mean < 0.03:
    xyz_quality = "⚠️  Acceptable (1-3cm)"
else:
    xyz_quality = "❌ Poor (> 3cm)"

# Rotation error assessment (in radians)
if rpy_mean < 0.1:
    rpy_quality = "✅ Good (< 5.7°)"
elif rpy_mean < 0.3:
    rpy_quality = "⚠️  Acceptable (5.7-17°)"
else:
    rpy_quality = "❌ Poor (> 17°)"

print(f"Position (XYZ): {xyz_quality} - Mean error: {xyz_mean*100:.2f} cm")
print(f"Rotation (RPY): {rpy_quality} - Mean error: {np.degrees(rpy_mean):.2f}°")
print(f"\n💡 Tip: Compare with other checkpoints and validation set for relative assessment.")
