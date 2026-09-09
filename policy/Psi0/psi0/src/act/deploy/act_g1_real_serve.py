from pathlib import Path
from typing import Union, Dict, Any, List
import torch
import torch.nn as nn
import os
import sys
import json
import time
import numpy as np
import os.path as osp
import tyro
import uvicorn
from PIL import Image
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from dataclasses import dataclass
from torchvision.transforms import v2
from psi.deploy.helpers import *
from psi.config.config import LaunchConfig, ServerConfig
from psi.config.transform import ACTModelTransform, ActionStateTransform, pad_to_len
from psi.config.model_act import ACTModelConfig
from psi.utils import parse_args_to_tyro_config
from psi.utils import seed_everything
from psi.utils.overwatch import initialize_overwatch
from act.models.act import ACTConfig, ACTPolicy

overwatch = initialize_overwatch(__name__)


def load_model(model_cfg: ACTModelConfig, run_dir: Path, ckpt_step: int | str = "latest"):
    """Load ACT model from checkpoint."""
    ckpt_path = run_dir / "checkpoints" / f"ckpt_{ckpt_step}.pth"
    if not ckpt_path.exists():
        # Try safetensors
        ckpt_path = run_dir / "checkpoints" / f"ckpt_{ckpt_step}" / "model.safetensors"
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

    overwatch.info(f"Loading checkpoint from {ckpt_path}")
    from safetensors.torch import load_file

    state_dict = load_file(ckpt_path)
    model.load_state_dict(state_dict)
    return model


class Server:
    def __init__(
        self,
        policy: str,
        run_dir: Path,
        ckpt_step: int | str = "latest",
        device: str = "cuda:0",
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your CUDA installation.")

        self.device = torch.device(device)
        overwatch.info(f"Using device: {self.device}")
        overwatch.info(f"Serving {policy}")

        assert osp.exists(run_dir), f"run_dir {run_dir} does not exist!"
        assert osp.exists(run_dir / "checkpoints" / f"ckpt_{ckpt_step}"), f"ckpt {ckpt_step} does not exist!"
        assert osp.exists(run_dir / "run_config.json"), f"run config does not exist!"

        # Build dynamic config and load from previously saved json
        config: LaunchConfig = parse_args_to_tyro_config(run_dir / "argv.txt")  # type: ignore
        conf = (run_dir / "run_config.json").open("r").read()
        launch_config = config.model_validate_json(conf)

        seed_everything(launch_config.seed or 42)

        self.model_cfg = launch_config.model
        assert isinstance(self.model_cfg, ACTModelConfig)

        self.model = load_model(self.model_cfg, run_dir, ckpt_step)
        self.model = self.model.to(self.device)
        self.model.eval()
        overwatch.info("Loaded model checkpoint successfully.")

        self.maxmin:ActionStateTransform = launch_config.data.transform.field # type:ignore
        self.model_transform:ACTModelTransform = launch_config.data.transform.model # type:ignore

        # Print number of total/trainable model parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        overwatch.info(f"Parameters (in millions): {num_params * 1e-6:.3f} Total", ctx_level=1)


        self.previous_rpy = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.previous_height = np.array([0.75], dtype=np.float32)

        overwatch.info("Loaded Dataset Statistics from run directory.")
        self.action_state_norm = launch_config.data.transform.field
        assert isinstance(self.action_state_norm, ActionStateTransform)
        self.action_normalization_type = self.action_state_norm.action_norm_type
        overwatch.info(f"Action Normalization Type: {self.action_normalization_type}")

        self.launch_config = launch_config
        self.num_image_chunk = 1  # Number of image frames in history
        self.count = 0

    def preprocess_image(self, image_dict: Dict[str, Any]) -> Dict[str, Any]:
        imgs = {}
        # FIXME 
        # image_key_to_cam_idx = {'rgb_head_stereo_left': 0}
        # for img_key in self.launch_config.data.transform.repack.image_keys:
        #     cam_idx = image_key_to_cam_idx[img_key] #self.launch_config.data.transform.repack.image_key_to_cam_idx[img_key]
        #     imgs[f"cam{cam_idx}"] = self._process_img(image_dict[f"{img_key}"])#[None, ...]

        for k in image_dict.keys():
            imgs[k] = self._process_img(image_dict[k])
            # if self.count % 1 == 0:
            #     # Convert tensor back to PIL Image for saving
            #     from torchvision.transforms import v2
            #     import os
            #     # Create directory if it doesn't exist
            #     os.makedirs("./act/tmp_visualize_images", exist_ok=True)
            #     # Denormalize and convert to PIL Image
            #     img_tensor = imgs[k][0]  # First image in batch
            #     # Simple denormalization for visualization (approximate)
            #     img_for_save = torch.clamp((img_tensor + 1.0) * 127.5, 0, 255).byte()
            #     pil_img = v2.ToPILImage()(img_for_save)
            #     pil_img.save(f"./act/tmp_visualize_images/img_{time.time()}_count{self.count}.png")

        # DEBUG write image to disk
        # from PIL import Image
        # import numpy as np
        # from misc import restore_resnet_normalized_image
        # Image.fromarray((restore_resnet_normalized_image(image_dict["cam0"][0,0])*255.).permute(1,2,0).cpu().numpy().astype(np.uint8)).save("req-0.png")

        return imgs

    def _process_img(self, img):

        # Handle PIL Image input
        if hasattr(img, 'save'):  # PIL Image object
            # Convert PIL Image to numpy array and add batch dimension
            import numpy as np
            img_array = np.array(img)  # (H, W, C)
            img = img_array[None, ...]  # (1, H, W, C)
        
        assert len(img.shape) == 4, f"expected shape: (1, h, w, c), got {img.shape}"
        
        transforms = [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            self.model_transform.resize(),
            self.model_transform.center_crop(),
            self.model_transform.normalize(),
        ]

        transform = v2.Compose(transforms)
        np_imgs = [img[i] for i in range(img.shape[0])]
        return torch.stack(transform(np_imgs))
        # Apply transform twice to match training bug (double normalization)
        # first_pass = torch.stack(transform(np_imgs))
        # second_pass = torch.stack([transform(first_pass[i]) for i in range(first_pass.shape[0])])
        # return second_pass

    # def preprocess_obs(self, state_dict: Dict[str, Any]) -> torch.Tensor:
    #     state_32d = self.compose_state(state_dict)
    #     obs = torch.tensor(state_32d, dtype=torch.float32)
    #     return obs[None, ...]  # (1, 38)

    def postprocess_action(self, action: np.ndarray) -> np.ndarray:
        return self.maxmin.denormalize(action)

    def predict_action(self, payload: Dict[str, Any]) -> str:
        try:
            request = RequestMessage.deserialize(payload)
            image_dict, instruction, history_dict, state_dict, gt_action, dataset_name = \
                    request.image, request.instruction, request.history, request.state, request.gt_action, request.dataset_name

            imgs = {}
            for cam_idx, img_key in enumerate(self.launch_config.data.transform.repack.image_keys):
                imgs[f"cam{cam_idx}"] = Image.fromarray(np.clip(image_dict[img_key], 0, 255).astype(np.uint8))

            image_input = self.preprocess_image(imgs)
            
            hand_joints = state_dict["hand_joints"].copy() # shape (14,)
            arm_joints = state_dict["arm_joints"].copy() # shape (14,)
            torso_rpy = self.previous_rpy # shape (3,)
            torso_height = self.previous_height # shape (1,)
            
            # Ensure all inputs are numpy arrays
            if hasattr(hand_joints, 'cpu'):
                hand_joints = hand_joints.cpu().numpy()
            if hasattr(arm_joints, 'cpu'):
                arm_joints = arm_joints.cpu().numpy()
            if hasattr(torso_rpy, 'cpu'):
                torso_rpy = torso_rpy.cpu().numpy()
            if hasattr(torso_height, 'cpu'):
                torso_height = torso_height.cpu().numpy()
                
            obs = np.concatenate([hand_joints, arm_joints, torso_rpy, torso_height], axis=-1) # (32,)


            # state normalization
            obs_input = self.maxmin.normalize_state_func(obs) # shape (32,)
            obs_input = torch.from_numpy(obs_input).unsqueeze(0).to(self.device) # (1, 32)

            # Debug logging
            for k, v in image_input.items():
                overwatch.info(
                    f"[DEBUG] {k}: shape={v.shape}, min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}"
                )
            overwatch.info(f"[DEBUG] obs_input: shape={obs_input.shape}, values={obs_input[0, :5].tolist()}...")

            with torch.inference_mode():
                # Prepare batch for ACT model
                imgs_tensor = next(iter(image_input.values())).to(self.device)  # (T, C, H, W)

                batch = {
                    "observation.images": imgs_tensor.unsqueeze(0),  # (1, T, C, H, W)
                    "observation.state": obs_input,  # (1, 32)
                }

                # Predict action chunk
                pred_actions = self.model.predict_action(batch)
                if len(pred_actions.shape) == 3:
                    pred_action = pred_actions[0]  # (T, D)
                else:
                    pred_action = pred_actions  # already (T, D)

            # Denormalize
            pred_action = pred_action[:self.model_cfg.n_action_steps, :]
            pred_actions_denorm = self.maxmin.denormalize(pred_action)
            
            self.previous_rpy = pred_action[self.model_cfg.n_action_steps-1, 28:31].cpu().numpy() # shape (3,)
            self.previous_height = pred_action[self.model_cfg.n_action_steps-1, 31:32].cpu().numpy() # shape (1,)

            # Convert to numpy
            if isinstance(pred_actions_denorm, torch.Tensor):
                pred_actions_denorm = pred_actions_denorm.cpu().numpy()

            overwatch.info(f"Predicted Action: {pred_actions_denorm[0]}")
            self.count += 1

            response = ResponseMessage(pred_actions_denorm, 0.0)
            return JSONResponse(content=response.serialize())

        except Exception as e:
            import traceback

            overwatch.warning(traceback.format_exc())
            return JSONResponse(content={"error": str(e)}, status_code=500)
    def reset(self, ):
        self.previous_rpy = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.previous_height = np.array([0.75], dtype=np.float32)

        self.count = 0
        print("=== rpy, height and self.count reset: ", self.previous_rpy, self.previous_height, self.count)
        
    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        self.app.get("/reset")(self.reset)
        self.app.get("/health")(lambda: JSONResponse(content={"status": "ok"}))
        overwatch.info(f"Server listening on {host}:{port}")
        try:
            uvicorn.run(self.app, host=host, port=port)
        except Exception as e:
            overwatch.warning(f"Server crashed: {e}")
        finally:
            overwatch.info("Server stopped.")
            exit(1)


def serve(cfg: ServerConfig) -> None:
    overwatch.info("Server :: Initializing Policy")
    assert cfg.policy is not None, "which policy to serve?"
    server = Server(cfg.policy, Path(cfg.run_dir), cfg.ckpt_step, cfg.device)

    overwatch.info("Server :: Spinning Up")
    server.run(cfg.host, cfg.port)

def main():
    overwatch.info("Start Serving from uv")
    overwatch.info(f"Args: {sys.argv}")
    config = tyro.cli(ServerConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=sys.argv[1:])
    serve(config)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()  # take environment variables from .env file
    config = tyro.cli(ServerConfig, config=(tyro.conf.ConsolidateSubcommandArgs,))
    serve(config)
