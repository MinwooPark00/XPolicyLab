from huggingface_hub import snapshot_download
from pathlib import Path

local_dir = Path(__file__).parent / "pi"

snapshot_download(
    repo_id="physical-intelligence/fast",
    repo_type="model",
    local_dir=str(local_dir),
)

print(f"Downloaded to {local_dir.resolve()}")
