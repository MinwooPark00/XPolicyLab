import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env BEFORE importing huggingface_hub
PROJECT_ROOT = Path(__file__).resolve().parents[2]
assert (PROJECT_ROOT / ".env").exists(), f".env file not found in {PROJECT_ROOT}"
load_dotenv(PROJECT_ROOT / ".env")

from huggingface_hub import snapshot_download, hf_hub_download, list_repo_files
import tyro


def download_from_huggingface(repo_id: str, remote_dir: str, repo_type: str= "model", local_dir: str=".data", move_to_local_root: bool=True) -> str:
    """Download files or folders from Hugging Face repository.
    
    Args:
        repo_id: Repository ID on Hugging Face
        repo_dir: File or folder path within the repository to download
        repo_type: Type of the repository (e.g., "model", "dataset")
        local_dir: Local directory to download files to
        move_to_local_root: If True and downloading a folder, move files to root local_dir 
    """
    
    try:
        # List all files in the repository to check if repo_dir is a file or folder
        repo_files = list_repo_files(repo_id=repo_id, repo_type=repo_type, token=os.getenv("HF_TOKEN"))
        
        # Check if repo_dir is an exact file match
        is_file = remote_dir in repo_files
        
        if is_file:
            # Download single file
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=remote_dir,
                local_dir=local_dir,
                repo_type=repo_type,
                token=os.getenv("HF_TOKEN")
            )
            print("Downloaded file to:", local_path)
        else:
            # Download all files under remote_dir directly into local_dir (flatten structure)
            local_path = snapshot_download(
                repo_id=repo_id,
                allow_patterns=f"{remote_dir}/**",
                local_dir=local_dir,
                repo_type=repo_type,
                token=os.getenv("HF_TOKEN"),
                ignore_patterns=None,
                # flatten: remove remote_dir prefix from local files
                # This will place files directly under local_dir
                # If snapshot_download does not support flatten, move files after download
            )
            if move_to_local_root:
                # Move files from local_dir/remote_dir/* to local_dir/*
                import shutil, glob
                src_dir = os.path.join(local_dir, remote_dir)
                if os.path.exists(src_dir):
                    for f in glob.glob(os.path.join(src_dir, "**"), recursive=True):
                        if os.path.isfile(f):
                            rel_path = os.path.relpath(f, src_dir)
                            dest_path = os.path.join(local_dir, rel_path)
                            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                            shutil.move(f, dest_path)
                    shutil.rmtree(src_dir)
                print("Downloaded folder to:", local_dir)
            
    except Exception as e:
        print(f"Error downloading {remote_dir}: {e}")
        raise
    
    return local_path

if __name__ == "__main__":
    tyro.cli(download_from_huggingface)
