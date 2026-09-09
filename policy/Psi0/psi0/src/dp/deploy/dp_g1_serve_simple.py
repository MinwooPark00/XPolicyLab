from pathlib import Path
from typing import Union, Dict, Any, List
import torch
import torch.nn as nn
import os
import sys
import json
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
from psi.config.transform import SimpleRepackTransform, DiffusionPolicyModelTransform, ActionStateTransform, pad_to_len
from psi.config.model_dp import DiffusionPolicyModelConfig
from psi.utils import parse_args_to_tyro_config
from psi.utils import seed_everything
from psi.utils.overwatch import initialize_overwatch
from dp.models.diffusion_policy import DiffusionPolicyModel

overwatch = initialize_overwatch(__name__)


def load_model(model_cfg: DiffusionPolicyModelConfig, run_dir: Path, ckpt_step: int | str = "latest"):
    ckpt_path = run_dir / "checkpoints" / f"ckpt_{ckpt_step}.pth"
    if not ckpt_path.exists():
        # Try safetensors
        ckpt_path = run_dir / "checkpoints" / f"ckpt_{ckpt_step}" / "model.safetensors"
        if not ckpt_path.exists():
             raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    model = DiffusionPolicyModel(
        vision_feature_dim=512,
        lowdim_obs_dim=model_cfg.obs_dim,
        action_dim=model_cfg.action_dim,
        obs_horizon=model_cfg.obs_horizon,
        pred_horizon=model_cfg.action_chunk_size,
        num_diffusion_iters=model_cfg.num_diffusion_iters
    )
    
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
        assert isinstance(self.model_cfg, DiffusionPolicyModelConfig)

        self.model = load_model(self.model_cfg, run_dir, ckpt_step)
        self.model = self.model.to(self.device)
        self.model.eval()
        overwatch.info("Loaded model checkpoint successfully.")

        self.maxmin:ActionStateTransform = launch_config.data.transform.field # type:ignore
        self.repack_transform:SimpleRepackTransform = launch_config.data.transform.repack # type:ignore
        self.model_transform:DiffusionPolicyModelTransform = launch_config.data.transform.model # type:ignore

        # Print number of total/trainable model parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        overwatch.info(f"Parameters (in millions): {num_params*1e-6:.3f} Total", ctx_level=1)

        self.previous_rpy = np.array([0.0, 0.0, 0.0], dtype=np.float32) # FIXME 
        self.previous_height = np.array([0.75], dtype=np.float32)

        overwatch.info("Loaded Dataset Statistics from run directory.")
        self.action_state_norm = launch_config.data.transform.field
        assert isinstance(self.action_state_norm, ActionStateTransform)
        self.action_normalization_type = self.action_state_norm.action_norm_type
        overwatch.info(f"Action Normalization Type: {self.action_normalization_type}")

        self.launch_config = launch_config
        self.num_image_chunk = 1  # Number of image frames in history
        self.idx = 0


    def predict_action(self, payload: Dict[str, Any]) -> str:
        try:
            request = RequestMessage.deserialize(payload)
            image_dict, instruction, history_dict, state_dict, gt_action, dataset_name = \
                request.image, request.instruction, request.history, request.state, request.gt_action, request.dataset_name
            
            overwatch.info(f"Instruction: {instruction}")

            transforms = [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                self.model_transform.resize(),
                self.model_transform.center_crop(),
                self.model_transform.normalize(),
            ]
            t = v2.Compose(transforms)

            states = torch.from_numpy(state_dict["states"].copy())

            if self.maxmin.normalize_state: # type:ignore
                states = torch.from_numpy(
                    self.maxmin.normalize_state_func(
                        pad_to_len(states.numpy(), self.maxmin.pad_state_dim, dim=1)[0]
                    )
                ).to(self.device)

            with torch.inference_mode():

                observations = torch.stack([t(Image.fromarray(img)) for img in image_dict.values()])  # (To, C, H, W)
                observations = observations.to(self.device)  # (To, C, H, W) - no batch dim for sample_actions

                pred_actions = self.model.sample_actions(
                    nimages=observations,
                    nagent_poses=states
                )
                if len(pred_actions.shape) == 3:
                    pred_action = pred_actions[0]  # (T, D)
                else:
                    pred_action = pred_actions  # already (T, D)

            # Denormalize
            pred_action = pred_action[:self.model_cfg.action_chunk_size, :]
            pred_actions_denorm = self.maxmin.denormalize(pred_action)

            # Convert to numpy
            if isinstance(pred_actions_denorm, torch.Tensor):
                pred_actions_denorm = pred_actions_denorm.cpu().numpy()

            overwatch.info(f"Predicted Action: {pred_actions_denorm[0]}")

            response = ResponseMessage(pred_actions_denorm, 0.0)
            self.idx += 1
            return JSONResponse(content=response.serialize())

        except Exception as e:
            import traceback

            overwatch.warning(traceback.format_exc())
            return JSONResponse(content={"error": str(e)}, status_code=500)

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
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
