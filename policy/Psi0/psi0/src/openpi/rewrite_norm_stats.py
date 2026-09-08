import tyro
import json
from dataclasses import dataclass

@dataclass
class Args:
    task_path: str = "lerobot/YourTaskName"  # Path to the task directory
    norm_stats_path: str = "meta/stats.json"  # Path to existing norm stats
    output_path: str = "norm_stats.json"  # Output path for rewritten norm

def rewrite_stats(args: Args) -> None:
    """Rewrite normalization statistics for a given task."""
    
    # read existing norm stats
    groot_stats = json.load(open(f"{args.task_path}/{args.norm_stats_path}"))
    print(groot_stats.keys())
    print(groot_stats["states"].keys())
    print(groot_stats["action"].keys())
    # exit(0)
    openpi_stats = {
        "norm_stats": {
            "state": groot_stats["states"],
            "actions": groot_stats["action"]
        }
    }
    # write to output path
    with open(f"{args.task_path}/{args.output_path}", "w") as f:
        json.dump(openpi_stats, f, indent=2)

    print(f"{args.task_path}/{args.output_path}")

if __name__ == "__main__":
    # use tyro to parse command line arguments
    rewrite_stats(tyro.cli(Args))