"""quat (xyzw) <-> 6D rotation, dependency-free (no pytorch3d).

MHBench stores every recorded/commanded orientation as an xyzw quaternion.
This module lets training data and `model.py`'s eval output move between that
native quat representation and the 6D rotation representation 
diffusion_policy/LatentToM's action space originally used (Zhou et al., CVPR 2019
-- 6D is continuous over SO(3), quat's antipodal q == -q double-cover is not, 
which can matter for a diffusion model regressing/denoising through the representation).

Convention (self-contained, not required to match pytorch3d's row/column
choice since nothing here loads a pytorch3d-trained checkpoint -- only needs
to be its own exact inverse, which the round-trip tests below check): the 6D
vector is the first two ROWS of the rotation matrix, Gram-Schmidt
orthonormalized to recover the third.
"""

import numpy as np


def quat_xyzw_to_matrix(quat_xyzw: np.ndarray) -> np.ndarray:
    """(..., 4) xyzw -> (..., 3, 3) rotation matrix."""
    quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
    x, y, z, w = np.moveaxis(quat_xyzw, -1, 0)
    n = x * x + y * y + z * z + w * w
    s = np.where(n > 0, 2.0 / n, 0.0)
    xs, ys, zs = x * s, y * s, z * s
    wx, wy, wz = w * xs, w * ys, w * zs
    xx, xy, xz = x * xs, x * ys, x * zs
    yy, yz, zz = y * ys, y * zs, z * zs
    rows = [
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ]
    matrix = np.stack([np.stack(row, axis=-1) for row in rows], axis=-2)
    return matrix.astype(np.float32)


def matrix_to_quat_xyzw(matrix: np.ndarray) -> np.ndarray:
    """(..., 3, 3) rotation matrix -> (..., 4) xyzw, via Shepperd's method."""
    matrix = np.asarray(matrix, dtype=np.float64)
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]
    trace = m00 + m11 + m22

    def _safe_sqrt(x):
        return np.sqrt(np.clip(x, 1e-12, None))

    case0 = trace > 0
    case1 = (~case0) & (m00 > m11) & (m00 > m22)
    case2 = (~case0) & (~case1) & (m11 > m22)
    case3 = (~case0) & (~case1) & (~case2)

    w = np.empty_like(trace)
    x = np.empty_like(trace)
    y = np.empty_like(trace)
    z = np.empty_like(trace)

    s = _safe_sqrt(trace + 1.0) * 2.0
    w = np.where(case0, 0.25 * s, w)
    x = np.where(case0, (m21 - m12) / s, x)
    y = np.where(case0, (m02 - m20) / s, y)
    z = np.where(case0, (m10 - m01) / s, z)

    s = _safe_sqrt(1.0 + m00 - m11 - m22) * 2.0
    w = np.where(case1, (m21 - m12) / s, w)
    x = np.where(case1, 0.25 * s, x)
    y = np.where(case1, (m01 + m10) / s, y)
    z = np.where(case1, (m02 + m20) / s, z)

    s = _safe_sqrt(1.0 + m11 - m00 - m22) * 2.0
    w = np.where(case2, (m02 - m20) / s, w)
    x = np.where(case2, (m01 + m10) / s, x)
    y = np.where(case2, 0.25 * s, y)
    z = np.where(case2, (m12 + m21) / s, z)

    s = _safe_sqrt(1.0 + m22 - m00 - m11) * 2.0
    w = np.where(case3, (m10 - m01) / s, w)
    x = np.where(case3, (m02 + m20) / s, x)
    y = np.where(case3, (m12 + m21) / s, y)
    z = np.where(case3, 0.25 * s, z)

    quat = np.stack([x, y, z, w], axis=-1)
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat.astype(np.float32)


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 6): the first two rows, concatenated."""
    return np.concatenate([matrix[..., 0, :], matrix[..., 1, :]], axis=-1).astype(np.float32)


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """(..., 6) -> (..., 3, 3), Gram-Schmidt orthonormalization."""
    rot6d = np.asarray(rot6d, dtype=np.float64)
    a1, a2 = rot6d[..., :3], rot6d[..., 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-8)
    a2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 / (np.linalg.norm(a2, axis=-1, keepdims=True) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2).astype(np.float32)


def quat_xyzw_to_rot6d(quat_xyzw: np.ndarray) -> np.ndarray:
    """(..., 4) xyzw -> (..., 6). Exact inverse of rot6d_to_quat_xyzw (up to
    the q == -q double cover, which represents the same physical rotation)."""
    return matrix_to_rot6d(quat_xyzw_to_matrix(quat_xyzw))


def rot6d_to_quat_xyzw(rot6d: np.ndarray) -> np.ndarray:
    """(..., 6) -> (..., 4) xyzw."""
    return matrix_to_quat_xyzw(rot6d_to_matrix(rot6d))


# Compressed 22D per-robot action column layout (scripts/data_convertion.py's
# default --action-key actions --compress-hands, hands already compressed
# 14D -> 4D): [0:3] left pos, [3:7] left quat(xyzw), [7:10] right pos,
# [10:14] right quat(xyzw), [14:18] hands, [18:21] base vel, [21:22] height.
# Shared by convert_to_replay_buffer.py (training-side, applied once at
# conversion time) and model.py's _unpack_robot_action (eval-side, inverse
# direction) -- a checkpoint trained under one rotation_rep is only decodable
# under the same one.
_LEFT_POS, _LEFT_QUAT = slice(0, 3), slice(3, 7)
_RIGHT_POS, _RIGHT_QUAT = slice(7, 10), slice(10, 14)
_REST = slice(14, 22)  # hands(4) + base_vel(3) + height(1) -- untouched by rotation_rep


def action_dim_for(rotation_rep: str) -> int:
    if rotation_rep == "quat":
        return 22
    if rotation_rep == "rot6d":
        return 26  # each wrist's quat(4) -> rot6d(6): +2 per wrist, x2 wrists
    raise ValueError(f"rotation_rep must be 'quat' or 'rot6d', got {rotation_rep!r}")


def repack_action(action_22d: np.ndarray, rotation_rep: str) -> np.ndarray:
    """(..., 22) compressed action -> (..., action_dim_for(rotation_rep))."""
    if rotation_rep == "quat":
        return action_22d
    left_6d = quat_xyzw_to_rot6d(action_22d[..., _LEFT_QUAT])
    right_6d = quat_xyzw_to_rot6d(action_22d[..., _RIGHT_QUAT])
    return np.concatenate(
        [action_22d[..., _LEFT_POS], left_6d, action_22d[..., _RIGHT_POS], right_6d, action_22d[..., _REST]],
        axis=-1,
    )
