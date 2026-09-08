import os
import numpy as np
import requests
import argparse
from PIL import Image
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass
from numpy.lib.format import descr_to_dtype, dtype_to_descr
from base64 import b64decode, b64encode

# ============ Serialization Utils ============

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


# ============ Message Classes ============

class RequestMessage:
    def __init__(self, image: Dict[str, Any], instruction: str, history: Dict[str, Any], 
                 state: Dict[str, Any], gt_action, dataset_name: str):
        self.image = image
        self.instruction = instruction
        self.history = history
        self.state = state
        self.gt_action = gt_action
        self.dataset_name = dataset_name

    def serialize(self):
        from datetime import datetime
        msg = {
            "image": self.image,
            "instruction": self.instruction,
            "history": self.history,
            "state": self.state,
            "condition": {},  # Not used by DP but required by server
            "gt_action": self.gt_action,
            "dataset_name": self.dataset_name,
            "timestamp": str(datetime.now()).replace(" ", "_").replace(":", "-")
        }
        return convert_numpy_in_dict(msg, numpy_serialize)


class ResponseMessage:
    def __init__(self, action: np.ndarray, err: float):
        self.action = action
        self.err = err
    
    @classmethod
    def deserialize(cls, response: Dict[str, Any]):
        response = convert_numpy_in_dict(response, numpy_deserialize)
        return cls(action=response["action"], err=response["err"])


# ============ HTTP Client ============

class DPActionClient:
    def __init__(self, server_ip: str = "127.0.0.1", server_port: int = 22089):
        self.server_ip = server_ip
        self.server_port = server_port
    
    def query_action(self, image_dict: Dict[str, np.ndarray], instruction: str, 
                     state_dict: Dict[str, np.ndarray], 
                     gt_action: Optional[np.ndarray] = None,
                     history: Optional[Dict[str, Any]] = None,
                     dataset: str = "grasp") -> tuple[np.ndarray, float]:
        if history is None:
            history = {k: [] for k in image_dict.keys()}
        if gt_action is None:
            gt_action = []
        
        request = RequestMessage(image_dict, instruction, history, state_dict, gt_action, dataset)
        response = requests.post(
            f"http://{self.server_ip}:{self.server_port}/act",
            json=request.serialize()
        )
        response = ResponseMessage.deserialize(response.json())
        return response.action, response.err
    
    def health_check(self) -> bool:
        try:
            response = requests.get(f"http://{self.server_ip}:{self.server_port}/health")
            return response.json().get("status") == "ok"
        except Exception as e:
            print(f"Health check failed: {e}")
            return False


# ============ Main Evaluation ============

def main():
    parser = argparse.ArgumentParser(description="DP Client for testing Diffusion Policy server")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="Server IP address")
    parser.add_argument("--port", type=int, default=22085, help="Server port")
    parser.add_argument("--dataset-path", type=str, 
                        default="/hfm/data/lerobot/vlt-fr3-frontstereo+wrist+side.rest.randall-37312-256-traj2d-obj63",
                        help="Path to LeRobot dataset")
    parser.add_argument("--episode-idx", type=int, default=0, help="Episode index to evaluate")
    args = parser.parse_args()
    
    np.set_printoptions(precision=4, suppress=True)
    
    # 1. Create client
    client = DPActionClient(args.ip, args.port)
    
    # Health check
    print(f"Connecting to server at {args.ip}:{args.port}...")
    if not client.health_check():
        print("❌ Server is not responding. Please start dp_serve.py first.")
        return
    print("✅ Server is healthy!")
    
    # 2. Load dataset
    print(f"\nLoading dataset from {args.dataset_path}...")
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    
    dataset_name = Path(args.dataset_path).name
    raw_dataset = LeRobotDataset(dataset_name, args.dataset_path)
    
    # Helper function - LeRobot images are in [0, 1] range (not normalized to [-1, 1])
    def tensor_to_numpy(x):
        """Convert tensor [0, 1] to uint8 image"""
        return (x.float().clamp(0, 1) * 255.0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    
    # 3. Evaluate one episode
    eps_idx = args.episode_idx
    from_idx = raw_dataset.episode_data_index["from"][eps_idx].item()
    to_idx = raw_dataset.episode_data_index["to"][eps_idx].item()
    
    print(f"\n{'='*50}")
    print(f"Evaluating Episode {eps_idx}")
    print(f"Frame range: {from_idx} - {to_idx}")
    print(f"Episode length: {to_idx - from_idx} frames")
    print(f"{'='*50}\n")
    
    all_action_l1_errs = []
    l2_xyz_errors = []
    l2_rpy_errors = []
    l2_gripper_errors = []
    
    for idx in range(from_idx, to_idx - 1):
        frame = raw_dataset[idx]
        
        # Prepare image - LeRobot images are already in [0, 1] range
        image = tensor_to_numpy(frame["observation.images.front_stereo_left"])
        
        # Prepare state
        obs = frame["observation.state"].numpy()
        state_dict = {
            "proprio_joint_positions": obs[:8],
            "proprio_eef_pose": obs[-7:]
        }
        
        # Prepare instruction and GT action
        instruction = frame["task"]
        gt_action = frame["action"].numpy()
        
        # Send request
        observations = {"front_stereo_left": image}
        
        try:
            pred_actions, err = client.query_action(
                observations, 
                instruction, 
                state_dict, 
                gt_action=gt_action
            )
        except Exception as e:
            print(f"❌ Request failed at frame {idx}: {e}")
            continue
        
        # Calculate errors (pred_actions is (Tp, Da), gt_action is (Da,))
        # Only compare the first predicted action with gt_action
        action_l1_errs = np.abs(pred_actions[0] - gt_action)  # Compare first action only
        avg_action_errors = action_l1_errs  # (Da,)
        all_action_l1_errs.append(action_l1_errs[np.newaxis, :])
        
        # Split into xyz, rpy, gripper
        avg_l1_action_err = np.split(avg_action_errors, [3, 6], axis=-1)
        xyz_err = np.linalg.norm(avg_l1_action_err[0])
        rpy_err = np.linalg.norm(avg_l1_action_err[1])
        gripper_err = np.linalg.norm(avg_l1_action_err[2])
        
        metric = {
            "denorm_err_l1_xyz": xyz_err,
            "denorm_err_l1_rpy": rpy_err,
            "denorm_err_l1_gripper": gripper_err,
        }
        
        print(f"Frame {idx - from_idx:3d} | {metric}")
        l2_xyz_errors.append(xyz_err)
        l2_rpy_errors.append(rpy_err)
        l2_gripper_errors.append(gripper_err)
    
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
    
    if xyz_mean < 0.01:
        xyz_quality = "✅ Good (< 1cm)"
    elif xyz_mean < 0.03:
        xyz_quality = "⚠️  Acceptable (1-3cm)"
    else:
        xyz_quality = "❌ Poor (> 3cm)"
    
    if rpy_mean < 0.1:
        rpy_quality = "✅ Good (< 5.7°)"
    elif rpy_mean < 0.3:
        rpy_quality = "⚠️  Acceptable (5.7-17°)"
    else:
        rpy_quality = "❌ Poor (> 17°)"
    
    print(f"Position (XYZ): {xyz_quality} - Mean error: {xyz_mean*100:.2f} cm")
    print(f"Rotation (RPY): {rpy_quality} - Mean error: {np.degrees(rpy_mean):.2f}°")


if __name__ == "__main__":
    main()
