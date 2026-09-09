import os, glob, sys

from dotenv import load_dotenv

# Load the repo-root .env so PSI_HOME / HF_HOME match the training & serving scripts.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
assert load_dotenv(os.path.join(_REPO_ROOT, ".env")), "Failed to load .env at " + _REPO_ROOT

PSI_HOME = os.environ["PSI_HOME"]
HF_HOME = os.environ["HF_HOME"]

repo_id = "Wan-AI/Wan2.1-I2V-14B-480P"

# Pre-downloaded checkpoint dir (hf download --local-dir) and the HF hub cache root.
local_dir = os.path.join(PSI_HOME, "cache", "checkpoints", repo_id.split("/")[-1])
hub_cache = os.path.join(HF_HOME, "hub")  # hf_hub_download cache = $HF_HOME/hub

repo_folder = "models--" + repo_id.replace("/", "--")
cache_repo = os.path.join(hub_cache, repo_folder)
dl_dir = os.path.join(local_dir, ".cache/huggingface/download")

meta_files = sorted(glob.glob(os.path.join(dl_dir, "**", "*.metadata"), recursive=True))
if not meta_files:
    sys.exit("No .metadata files found under " + dl_dir)

commit = None
created, skipped_missing = [], []
for meta in meta_files:
    rel = os.path.relpath(meta, dl_dir)[: -len(".metadata")]  # e.g. models_t5_...pth
    real_file = os.path.join(local_dir, rel)
    if not os.path.exists(real_file):
        skipped_missing.append(rel)
        continue
    with open(meta) as f:
        lines = f.read().splitlines()
    c, etag = lines[0].strip(), lines[1].strip()
    commit = c

    # blob: named by etag, symlink to the real downloaded file
    blob_dir = os.path.join(cache_repo, "blobs")
    os.makedirs(blob_dir, exist_ok=True)
    blob_path = os.path.join(blob_dir, etag)
    if not os.path.lexists(blob_path):
        os.symlink(real_file, blob_path)

    # snapshot: snapshots/<commit>/<rel> -> ../../blobs/<etag> (relative symlink)
    snap_path = os.path.join(cache_repo, "snapshots", c, rel)
    os.makedirs(os.path.dirname(snap_path), exist_ok=True)
    if not os.path.lexists(snap_path):
        os.symlink(os.path.relpath(blob_path, os.path.dirname(snap_path)), snap_path)
    created.append(rel)

# refs/main -> commit
if commit:
    refs_dir = os.path.join(cache_repo, "refs")
    os.makedirs(refs_dir, exist_ok=True)
    with open(os.path.join(refs_dir, "main"), "w") as f:
        f.write(commit)

print("repo cache:", cache_repo)
print("commit (refs/main):", commit)
print("linked %d files" % len(created))
if skipped_missing:
    print("skipped (no local file):", len(skipped_missing), skipped_missing[:5])
