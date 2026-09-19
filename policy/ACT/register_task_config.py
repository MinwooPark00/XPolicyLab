"""Register a scripts/data_convertion.py --format act output in TASK_CONFIGS.json.

ACT's own detr/process_data.py writes TASK_CONFIGS.json from a different raw
data layout (data/<bench>/<ckpt>/<env_cfg_type>/data/episode_*.hdf5, packed
via pack_robot_state against the parent env_cfg/ tree). MHBench datasets go
through scripts/data_convertion.py instead, which writes the same per-episode
<dataset_dir>/episode_N.hdf5 shape ACT's dataloader (utils.py) expects, but
never touches TASK_CONFIGS.json -- this script closes that gap.

Usage (run from this directory, so TASK_CONFIGS.json lands where
imitate_episodes.py / constants.py look for it):
    python register_task_config.py \
        --bench_name mhbench --ckpt_name cocarry_robot_a \
        --env_cfg_type unitree_g1x2_decentralized --action_type joint \
        --dataset_dir <mhbench>/baselines/data/cocarry_act_robot_a

The registry key is "{bench_name}-{ckpt_name}-{env_cfg_type}-{action_type}"
(imitate_episodes.py's --ckpt_setting). --env_cfg_type only encodes the
dimensional profile (state/action dims), which decentralized robot_a and
robot_b share -- so for a decentralized dataset pair, give each robot a
distinct --ckpt_name (e.g. "cocarry_robot_a" / "cocarry_robot_b"), not the
same one, or the second registration silently overwrites the first.

--variant names an MHBench experiment setting trained over the same
(ckpt_name, env_cfg_type) as the default one -- `jointobs` (both robots' obs
in, own action out) and `jointact` (own obs in, both robots' action out). Its
dataset is a different directory, so it gets its own key: the variant is
appended, exactly as train.sh appends it to --ckpt_setting (ACT_VARIANT).
`dtde` and `ctce` are the unsuffixed decentralized and centralized defaults.

--val_dataset_dir registers the held-out split (`data_convertion.py --split
val`), which utils.load_data then validates on instead of carving 20% out of
the training episodes.
"""

import argparse
import json
import os

import h5py


VARIANTS = ("dtde", "jointobs", "jointact", "ctce")
"""train.sh's ACT_VARIANT values. Only the two middle ones suffix the key."""


def episode_files_in(dataset_dir: str) -> list[str]:
    """`episode_0.hdf5 ..` in index order, refusing a gap: EpisodicDataset
    indexes by range(num_episodes)."""
    files = sorted(
        (f for f in os.listdir(dataset_dir) if f.startswith("episode_") and f.endswith(".hdf5")),
        key=lambda f: int(f[len("episode_"):-len(".hdf5")]),
    )
    if not files:
        raise SystemExit(f"no episode_*.hdf5 files in {dataset_dir}")
    if files != [f"episode_{i}.hdf5" for i in range(len(files))]:
        raise SystemExit(f"{dataset_dir} episode files aren't a contiguous 0..N-1 sequence: got {files}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench_name", required=True)
    parser.add_argument("--ckpt_name", required=True, help="Run name; combined with the others into ckpt_setting.")
    parser.add_argument("--env_cfg_type", required=True, help="Must match a key in utils/robot/_robot_info.json.")
    parser.add_argument("--action_type", required=True)
    parser.add_argument("--dataset_dir", required=True, help="A scripts/data_convertion.py --format act output dir.")
    parser.add_argument("--config_path", default="./TASK_CONFIGS.json")
    parser.add_argument("--variant", default="dtde", choices=VARIANTS,
                        help="MHBench experiment setting; jointobs/jointact get their own key suffix.")
    parser.add_argument("--val_dataset_dir", default=None,
                        help="The held-out split's directory; validated on instead of a random 20%% carve.")
    args = parser.parse_args()

    episode_files = sorted(
        (f for f in os.listdir(args.dataset_dir) if f.startswith("episode_") and f.endswith(".hdf5")),
        key=lambda f: int(f[len("episode_"):-len(".hdf5")]),
    )
    if not episode_files:
        raise SystemExit(f"no episode_*.hdf5 files in {args.dataset_dir}")
    expected = [f"episode_{i}.hdf5" for i in range(len(episode_files))]
    if episode_files != expected:
        raise SystemExit(
            f"{args.dataset_dir} episode files aren't a contiguous 0..N-1 sequence "
            f"(EpisodicDataset indexes by range(num_episodes)): got {episode_files}"
        )

    episode_lens = []
    camera_names = None
    for f in episode_files:
        with h5py.File(os.path.join(args.dataset_dir, f), "r") as demo:
            episode_lens.append(demo["/action"].shape[0])
            cams = sorted(demo["/observations/images"].keys())
            if camera_names is None:
                camera_names = cams
            elif cams != camera_names:
                raise SystemExit(f"{f} has cameras {cams}, earlier episodes had {camera_names} -- inconsistent dataset")

    ckpt_setting = f"{args.bench_name}-{args.ckpt_name}-{args.env_cfg_type}-{args.action_type}"
    if args.variant in ("jointobs", "jointact"):
        ckpt_setting += f"-{args.variant}"

    try:
        with open(args.config_path, "r") as fh:
            task_configs = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        task_configs = {}

    task_configs[ckpt_setting] = {
        "dataset_dir": os.path.abspath(args.dataset_dir),
        "num_episodes": len(episode_files),
        "episode_len": max(episode_lens),
        "camera_names": camera_names,
    }
    if args.val_dataset_dir:
        val_files = episode_files_in(args.val_dataset_dir)
        with h5py.File(os.path.join(args.val_dataset_dir, val_files[0]), "r") as demo:
            val_cams = sorted(demo["/observations/images"].keys())
        if val_cams != camera_names:
            raise SystemExit(f"{args.val_dataset_dir} has cameras {val_cams}, the training set {camera_names}")
        task_configs[ckpt_setting]["val_dataset_dir"] = os.path.abspath(args.val_dataset_dir)
        task_configs[ckpt_setting]["num_val_episodes"] = len(val_files)

    with open(args.config_path, "w") as fh:
        json.dump(task_configs, fh, indent=4)

    print(f"[register_task_config] {args.config_path}[{ckpt_setting!r}] = {task_configs[ckpt_setting]}")


if __name__ == "__main__":
    main()
