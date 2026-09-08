from dotenv import load_dotenv
assert load_dotenv(), "Failed to load .env file. Make sure it exists and is properly formatted."

import dataclasses
import enum
import logging
import pathlib
import time
import os
import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
import polars as pl
import rich
import tqdm
import tyro

logger = logging.getLogger(__name__)

class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    HFM = "hfm"

@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str

@dataclasses.dataclass
class Args:
    """Command line arguments."""

    # Host and port to connect to the server.
    host: str = "0.0.0.0"
    # Port to connect to the server. If None, the server will use the default port.
    port: int | None = 8000
    # API key to use for the server.
    api_key: str | None = None
    # # Number of steps to run the policy for.
    # num_steps: int = 20
    # Path to save the timings to a parquet file. (e.g., timing.parquet)
    timing_file: pathlib.Path | None = None
    # Environment to run the policy in.
    env: EnvMode = EnvMode.HFM
    task: str = "Pick_toys_into_box_and_lift_and_turn_and_put_on_the_chair_new_target_yaw"

    # policy: Checkpoint = dataclasses.field(default_factory=Checkpoint)

class TimingRecorder:
    """Records timing measurements for different keys."""

    def __init__(self) -> None:
        self._timings: dict[str, list[float]] = {}

    def record(self, key: str, time_ms: float) -> None:
        """Record a timing measurement for the given key."""
        if key not in self._timings:
            self._timings[key] = []
        self._timings[key].append(time_ms)

    def get_stats(self, key: str) -> dict[str, float]:
        """Get statistics for the given key."""
        times = self._timings[key]
        return {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "p25": float(np.quantile(times, 0.25)),
            "p50": float(np.quantile(times, 0.50)),
            "p75": float(np.quantile(times, 0.75)),
            "p90": float(np.quantile(times, 0.90)),
            "p95": float(np.quantile(times, 0.95)),
            "p99": float(np.quantile(times, 0.99)),
        }

    def print_all_stats(self) -> None:
        """Print statistics for all keys in a concise format."""

        table = rich.table.Table(
            title="[bold blue]Timing Statistics[/bold blue]",
            show_header=True,
            header_style="bold white",
            border_style="blue",
            title_justify="center",
        )

        # Add metric column with custom styling
        table.add_column("Metric", style="cyan", justify="left", no_wrap=True)

        # Add statistical columns with consistent styling
        stat_columns = [
            ("Mean", "yellow", "mean"),
            ("Std", "yellow", "std"),
            ("P25", "magenta", "p25"),
            ("P50", "magenta", "p50"),
            ("P75", "magenta", "p75"),
            ("P90", "magenta", "p90"),
            ("P95", "magenta", "p95"),
            ("P99", "magenta", "p99"),
        ]

        for name, style, _ in stat_columns:
            table.add_column(name, justify="right", style=style, no_wrap=True)

        # Add rows for each metric with formatted values
        for key in sorted(self._timings.keys()):
            stats = self.get_stats(key)
            values = [f"{stats[key]:.1f}" for _, _, key in stat_columns]
            table.add_row(key, *values)

        # Print with custom console settings
        console = rich.console.Console(width=None, highlight=True)
        console.print(table)

    def write_parquet(self, path: pathlib.Path) -> None:
        """Save the timings to a parquet file."""
        print(f"Writing timings to {path}")
        frame = pl.DataFrame(self._timings)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(path)

import torch
from PIL import Image
def pt_to_pil(x, normalizee=False):
    s, b = (0.5, 0.5) if normalizee else (1.0, 0.0)
    return Image.fromarray(
        (((x.float() * s + b).clamp(0, 1))*255.0).permute(1,2,0).cpu().numpy().astype(np.uint8)
    )

def obs_fn(sample, prompt): 
    return {
        "observation/image": pt_to_pil(sample["observation.images.egocentric"]),
        # "observation/arm_joints": sample["observation.arm_joints"].numpy(),
        # "observation/hand_joints": sample["observation.hand_joints"].numpy(),
        # "observation/leg_joints": sample["observation.leg_joints"].numpy(),
        # "observation/torso_rpy": sample["observation.prev_rpy"].numpy(),
        # "observation/base_height": sample["observation.prev_height"].numpy(),
        "states": sample["states"].numpy()[:28],
        "prompt": f"{prompt}" #"g1/Remove_the_cap_turn_on_the_faucet_and_fill_the_bottle_with_water",
    }

def main(args: Args) -> None:
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
    
    
    root_dir = f"{os.environ['DATA_HOME']}/simple/{args.task}"
    action_horizon = 30
    action_sequence_keys = ("action",)
    meta = lerobot_dataset.LeRobotDatasetMetadata(root_dir)
    delta_timestamps={
        key: [t / meta.fps for t in range(action_horizon)] for key in action_sequence_keys
    }

    dataset = LeRobotDataset(root_dir, delta_timestamps=delta_timestamps)
    episode_idx = 0
    from_idx:int = dataset.episode_data_index["from"][episode_idx].item()
    to_idx:int = dataset.episode_data_index["to"][episode_idx].item()
    

    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
    )
    print(f"Server metadata: {policy.get_server_metadata()}")

    timing_recorder = TimingRecorder()

    l1_losses = []
    for i in tqdm.tqdm(range(from_idx, to_idx + 1, 10)):
        sample = dataset[i]

        # print(type(sample["action"])) # torch.Tensor, action
        inference_start = time.time()
        # print(f"============={i}=============")
        obs = obs_fn(sample, prompt=sample["task"])
        # print(obs["observation/image"].save(f"debug_obs_{i}.png"))
        obs["observation/image"] = np.array(obs["observation/image"], dtype=np.uint8)
        result = policy.infer(obs)
        pred_actions = result["actions"]
        # print(f"actions.shape {pred_actions.shape}")
        n_action_dim = pred_actions.shape[-1]
        gt_action = sample["action"][:, :n_action_dim].numpy()  # FIXME
        # print(f"gt_action.shape:{gt_action.shape}")
        l1_loss = np.abs(gt_action - pred_actions)
        timing_recorder.record("client_infer_ms", 1000 * (time.time() - inference_start))
        # print(l1_loss)
        # print("="*20)
        l1_losses.append(l1_loss)
    
    # print("==== Final L1 Loss ====")
    l1_losses = np.array(l1_losses)
    # print(f"L1 Loss shape: {l1_losses.shape}")
    np.save("hfm_pick_n_squat_l1_loss.npy", l1_losses)
    # print(f"Mean L1 Loss: {np.mean(l1_losses)}")
    # print(f"Std L1 Loss: {np.std(l1_losses)}"
    plot_error(l1_losses)

def plot_error(l1_losses=None):
    import numpy as np
    import matplotlib.pyplot as plt

    _plot_dir = pathlib.Path(__file__).parent

    if l1_losses is None:
        l1_losses = np.load("hfm_pick_n_squat_l1_loss copy.npy")
    print(f"L1 Loss shape: {l1_losses.shape}")
    print(f"Mean L1 Loss: {np.mean(l1_losses)}")

    if True:
        errors_roll = np.rad2deg(l1_losses[:, 0, 28])
        errors_pitch = np.rad2deg(l1_losses[:, 0, 29])
        errors_yaw = np.rad2deg(l1_losses[:, 0, 30])
        # Plot
        plt.figure(figsize=(8, 4))
        # plt.plot(np.rad2deg(errors_roll), marker='o')
        plt.plot(errors_roll, label="Roll (0,0)")
        plt.plot(errors_pitch, label="Pitch (0,1)")
        plt.plot(errors_yaw, label="Yaw (0,2)")

        plt.xlabel("Index")
        plt.ylabel("L1 Loss")
        plt.title("L1 Loss for Roll, Pitch, Yaw (l1_losses[:, 0, *]) (Degree) ")
        plt.grid(True)
        plt.legend()

        # plt.show()
        _save_path = _plot_dir / "l1_loss_plot_rpy.png"
        plt.savefig(_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved figure: {_save_path}")

    if True:
        errors_height = l1_losses[:, 0, 31]
        # Plot
        plt.figure(figsize=(8, 4))
        plt.plot(errors_height, marker='o')

        plt.xlabel("Index")
        plt.ylabel("L1 Loss")
        plt.title("L1 Loss for height (Meter)")
        plt.grid(True)

        # plt.show()
        _save_path = _plot_dir / "l1_loss_plot_height.png"
        plt.savefig(_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved figure: {_save_path}")

    if True:
        errors_vx = l1_losses[:, 0, 32]
        # Plot
        plt.figure(figsize=(8, 4))
        plt.plot(errors_vx, marker='o')

        plt.xlabel("Index")
        plt.ylabel("L1 Loss")
        plt.title("L1 Loss for Vx (m/s)")
        plt.grid(True)

        # plt.show()
        _save_path = _plot_dir / "l1_loss_plot_vx.png"
        plt.savefig(_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved figure: {_save_path}")

    if True:
        errors_vy = l1_losses[:, 0, 33]
        # Plot
        plt.figure(figsize=(8, 4))
        plt.plot(errors_vy, marker='o')

        plt.xlabel("Index")
        plt.ylabel("L1 Loss")
        plt.title("L1 Loss for Vy (m/s)")
        plt.grid(True)

        # plt.show()
        _save_path = _plot_dir / "l1_loss_plot_vy.png"
        plt.savefig(_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved figure: {_save_path}")

    if True:
        errors_vyaw = l1_losses[:, 0, 34]
        # Plot
        plt.figure(figsize=(8, 4))
        plt.plot(errors_vyaw, marker='o')

        plt.xlabel("Index")
        plt.ylabel("L1 Loss")
        plt.title("L1 Loss for Vyaw (m/s)")
        plt.grid(True)

        # plt.show()
        _save_path = _plot_dir / "l1_loss_plot_vyaw.png"
        plt.savefig(_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved figure: {_save_path}")

    if True:
        errors_target_yaw = np.rad2deg(l1_losses[:, 0, 35])
        # Plot
        plt.figure(figsize=(8, 4))
        plt.plot(errors_target_yaw, marker='o')

        plt.xlabel("Index")
        plt.ylabel("L1 Loss")
        plt.title("L1 Loss for Target Yaw (Degree)")
        plt.grid(True)

        # plt.show()
        _save_path = _plot_dir / "l1_loss_plot_target_yaw.png"
        plt.savefig(_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved figure: {_save_path}")

    if True:
        errors_arm = np.rad2deg(l1_losses[:, 0, 14:28])
        plt.figure(figsize=(8, 4))

        # Plot all 14 curves
        plt.figure(figsize=(10, 5))
        for i in range(errors_arm.shape[1]):
            plt.plot(errors_arm[:, i], label=f"Arm Joint {i}")

        plt.xlabel("Index")
        plt.ylabel("L1 Loss")
        plt.title("Arm Joint L1 Loss (joints 0–13) (Degree)")
        plt.grid(True)
        plt.legend(ncol=2, fontsize=8)

        # plt.show()
        _save_path = _plot_dir / "l1_loss_plot_arm.png"
        plt.savefig(_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved figure: {_save_path}")

    if True:
        errors_hand = np.rad2deg(l1_losses[:, 0, 0:14])
        plt.figure(figsize=(8, 4))

        # Plot all 14 curves
        plt.figure(figsize=(10, 5))
        for i in range(errors_hand.shape[1]):
            plt.plot(errors_hand[:, i], label=f"Hand Joint {i}")

        plt.xlabel("Index")
        plt.ylabel("L1 Loss")
        plt.title("Hand Joint L1 Loss (joints 0–13) (Degree)")
        plt.grid(True)
        plt.legend(ncol=2, fontsize=8)

        # plt.show()
        _save_path = _plot_dir / "l1_loss_plot_hand.png"
        plt.savefig(_save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved figure: {_save_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
    # plot_error()
    