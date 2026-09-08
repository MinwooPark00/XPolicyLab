from human_plan.dataset_preprocessing.otv_isaaclab.utils import (
  parse_single_seq_image,
  parse_single_seq_hand,
  get_all_seqs,
)

from human_plan.dataset_preprocessing.otv_isaaclab.utils import load_episode_data
dataset_root = "data/OTV_FIXED_SET"


all_seqs = get_all_seqs(dataset_root, seq_skip=1)

from collections import defaultdict
starting_dict = defaultdict(dict)

# task_lists = 
# task_name = task_args.task[9:-3]
# seq_name = f"{task_name}/episode_{task_args.episode_label}.hdf5"

# load_name = task_name
# if load_name == "Press-Gamepad-Red-Blue":
#     load_name = "Press-Gamepad-Blue-Red"
from tqdm import tqdm

padding_idx = 0


import numpy as np
left_hand_dof_index = np.array([26, 36, 27, 37, 28, 38, 29, 39, 30, 40, 46, 48])
right_hand_dof_index = np.array([31, 41, 32, 42, 33, 43, 34, 44, 35, 45, 47, 49])

for task_name, seq_name in tqdm(
  all_seqs,
):
  seq_data =load_episode_data(
    dataset_root,
    task_name,
    seq_name,
    clip_starting=0
  )

  starting_dict[task_name][seq_name] = {}
  for padding_idx in range(21):
    left_dof = seq_data["action"][padding_idx, left_hand_dof_index]
    right_dof = seq_data["action"][padding_idx, right_hand_dof_index]
    
    left_ee_pose_traj_gt = seq_data["observations"]["left_ee_pose"][padding_idx]
    right_ee_pose_traj_gt = seq_data["observations"]["right_ee_pose"][padding_idx]

    starting_dict[task_name][seq_name][padding_idx] = {
      "left_dof" : left_dof,
      "right_dof": right_dof,
      "left_ee": left_ee_pose_traj_gt,
      "right_ee": right_ee_pose_traj_gt
    }

import pickle

with open("init_poses_fixed_set.pkl", "wb") as f:
  pickle.dump(starting_dict, f)