import os
import re
import sys
import json
import torch
import uvicorn
import numpy as np
import tyro



from PIL import Image


from pathlib import Path
from fastapi import FastAPI
from typing import Union, Dict, Any, List
from base64 import b64decode, b64encode
from numpy.lib.format import descr_to_dtype, dtype_to_descr
from fastapi.responses import JSONResponse

from InternVLA.model.framework.M1 import InternVLA_M1
from InternVLA.model.framework.share_tools import read_mode_config
from pydantic import BaseModel


class ServerConfig(BaseModel):
    checkpoint_path: str
    host: str = "0.0.0.0"
    port: int = 21074
    device: str = "cuda:0"

def numpy_serialize(o):
    if isinstance(o, (np.ndarray, np.generic)):
        data = o.data if o.flags["C_CONTIGUOUS"] else o.tobytes()
        return {
            "__numpy__": b64encode(data).decode(),
            "dtype": dtype_to_descr(o.dtype),
            "shape": o.shape,
        }

    msg = f"Object of type {o.__class__.__name__} is not JSON serializable"
    raise TypeError(msg)

def numpy_deserialize(dct):
    if "__numpy__" in dct:
        np_obj = np.frombuffer(b64decode(dct["__numpy__"]), descr_to_dtype(dct["dtype"]))
        return np_obj.reshape(shape) if (shape := dct["shape"]) else np_obj[0]
    return dct


def convert_numpy_in_dict(data, func):
    """
    Recursively processes a JSON-like dictionary, converting any NumPy arrays
    or lists of NumPy arrays into a serializable format using the provided function.

    Args:
        data: The JSON-like dictionary or object to process.
        func: A function to apply to each NumPy array to make it serializable.

    Returns:
        The processed dictionary or object with all NumPy arrays converted.
    """
    if isinstance(data, dict):
        if "__numpy__" in data:
            return func(data)
        return {key: convert_numpy_in_dict(value, func) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_in_dict(item, func) for item in data]
    elif isinstance(data, (np.ndarray, np.generic)):
        return func(data)
    else:
        return data


class Message(object):
    def __init__(self):
        pass
    
    def serialize(self):
        raise NotImplementedError
    
    @classmethod
    def deserialize(cls, response: Dict[str, Any]):
        raise NotImplementedError

class ResponseMessage(Message):
    def __init__(self, action: np.ndarray, err: float, traj_image: np.ndarray = np.zeros((1,1,3), dtype=np.uint8)):
        self.action = action
        self.err = err
        self.traj_image = traj_image
    
    def serialize(self):
        msg = {
            "action": self.action,
            "err": self.err,
            "traj_image": self.traj_image
        }
        return convert_numpy_in_dict(msg, numpy_serialize)
    
    @classmethod
    def deserialize(cls, response: Dict[str, Any]):
        response = convert_numpy_in_dict(response, numpy_deserialize)
        return cls(action=response["action"], err=response["err"], traj_image=response["traj_image"])




class Server:

    def __init__(self, checkpoint_path: str, device: str = "cuda:0"):
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # Load the model
        self.vla : InternVLA_M1 = InternVLA_M1.from_pretrained(checkpoint_path)
        self.unnorm_key = list(self.vla.norm_stats.keys())[0]
        print("unnorm_key:", self.unnorm_key)

        self.vla = self.vla.to(device).eval()

        # Get action normalization stats
        self.action_norm_stats = self.get_action_stats(self.unnorm_key, policy_ckpt_path=checkpoint_path)
        print("action_norm_stats:", self.action_norm_stats)

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Unnormalize actions using the action normalization statistics.
        
        Args:
            normalized_actions: Normalized actions in range [-1, 1]
            action_norm_stats: Dictionary containing 'q01', 'q99', and optionally 'mask'
            
        Returns:
            Unnormalized actions
        """
        action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        mask = action_high != action_low


        normalized_actions = np.clip(normalized_actions, -1, 1)
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        
        return actions
    
    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        """
        Get action normalization statistics from the model checkpoint.
        
        Args:
            unnorm_key: Key to access the normalization statistics
            policy_ckpt_path: Path to the policy checkpoint
            
        Returns:
            Dictionary containing action normalization statistics
        """
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)  # read config and norm_stats
        unnorm_key = InternVLA_M1._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]
   
        
    
    def predict_action(self, payload: Dict[str, Any]) -> str:

        try:
            # request = RequestMessage.deserialize(payload)
            # image_dict, instruction, history_dict, state_dict, gt_action, dataset_name = \
            #         request.image, request.instruction, request.history, request.state, request.gt_action, request.dataset_name
            
            # condition = request.condition
            payload = convert_numpy_in_dict(payload, numpy_deserialize)
            print(payload)
            image = payload["image"]
            image = np.clip(image, 0, 255, dtype=np.uint8)
            image = Image.fromarray(image)
            images = [[image]]
            # instruction = payload["instruction"]
            instruction = ""
            
            if instruction == "":
                raise ValueError("instruction is empty, please manually provide a valid instruction for this task")
            instructions = [instruction]

            print(f"Instruction: {instruction}")

            
            with torch.inference_mode():
                normalized_actions = self.vla.predict_action(
                    batch_images=images, 
                    instructions=instructions,
                    unnorm_key=self.unnorm_key,
                    do_sample=False, 
                    cfg_scale=1.5,
                    use_ddim=True,
                    num_ddim_steps=20, #FIXME
                )['normalized_actions'][0]

                print("normalized_actions:", normalized_actions)
                print("normalized_actions shape:", normalized_actions.shape)
                unnormalized_actions = self.unnormalize_actions(
                    normalized_actions=normalized_actions, 
                    action_norm_stats=self.action_norm_stats
                )
            


            print(f"Predicted Action Unnormalized: {unnormalized_actions}")
            # print(f"Normalized Error: {normed_err}")
            response = ResponseMessage(unnormalized_actions, err=0.0)
            return JSONResponse(content=response.serialize())

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            # overwatch.critical(f"{e}")
            return "error"
        
    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        self.app.get("/health")(lambda: JSONResponse(content={"status": "ok"}))
        print(f"Server listens on {host}:{port}")
        try:
            uvicorn.run(self.app, host=host, port=port)
        except Exception as e:
            print(f"Server crashed, {e}")
        finally:
            print("Server stopped.")
            exit(1)

def serve(cfg: ServerConfig) -> None:
    print("Server :: Initializing Policy")
    server = Server(cfg.checkpoint_path, cfg.device)
    
    print("Server :: Spinning Up")
    server.run(cfg.host, cfg.port)

def main():
    print("Start Serving from uv")
    print(f"Args: {sys.argv}")
    config = tyro.cli(ServerConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=sys.argv[1:])
    serve(config)

if __name__ == "__main__":
    config = tyro.cli(ServerConfig, config=(tyro.conf.ConsolidateSubcommandArgs,))
    serve(config)