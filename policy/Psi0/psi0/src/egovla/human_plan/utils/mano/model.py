import torch
import numpy as np
import smplx
import os
import warnings
from smplx.lbs import blend_shapes, vertices2joints

_mano_root = os.environ.get("MANO_MODEL_DIR", "mano_v1_2/models")
_mano_left_path = os.path.join(_mano_root, "MANO_LEFT.pkl")
_mano_right_path = os.path.join(_mano_root, "MANO_RIGHT.pkl")

mano_left = None
mano_right = None
if os.path.exists(_mano_left_path) and os.path.exists(_mano_right_path):
  mano_left = smplx.create(
    _mano_left_path,
    "mano",
    use_pca=True,
    is_rhand=False,
    num_pca_comps=15,
  )
  mano_left.to("cpu")

  mano_right = smplx.create(
    _mano_right_path,
    "mano",
    use_pca=True,
    is_rhand=True,
    num_pca_comps=15,
  )
  mano_right.to("cpu")
else:
  warnings.warn(
    f"MANO model files not found under '{_mano_root}'. "
    "Set MANO_MODEL_DIR to a directory containing MANO_LEFT.pkl and MANO_RIGHT.pkl."
  )
