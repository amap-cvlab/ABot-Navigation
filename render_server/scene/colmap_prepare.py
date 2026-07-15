import numpy as np
from math import *
def get_RT(R33, T3):
    RT = np.zeros((4, 4))
    RT[:3, :3] = R33
    RT[:3, 3] = T3
    RT[3, 3] = 1.0
    return RT


def get_pmat(roll, pitch, yaw, ox, oy, oz, fx, fy, cx, cy) -> np.ndarray:
    """
    get_pmat_c2w
    get transform matrix for each frame using intrinsic and extrinsic information

    Returns:
        np.ndarray: transform matrix
    """
    deg2rad = np.pi / 180.0
    w = roll * deg2rad
    f = pitch * deg2rad
    k = yaw * deg2rad

    rot_mat = np.zeros((3, 3), dtype=np.float32)
    rot_mat[0, 0] = cos(f) * cos(k)
    rot_mat[0, 1] = cos(k) * sin(f) * sin(w) - cos(w) * sin(k)
    rot_mat[0, 2] = sin(k) * sin(w) + cos(k) * sin(f) * cos(w)
    rot_mat[1, 0] = cos(f) * sin(k)
    rot_mat[1, 1] = cos(k) * cos(w) + sin(f) * sin(k) * sin(w)
    rot_mat[1, 2] = sin(f) * cos(w) * sin(k) - cos(k) * sin(w)
    rot_mat[2, 0] = -sin(f)
    rot_mat[2, 1] = cos(f) * sin(w)
    rot_mat[2, 2] = cos(f) * cos(w)

    intrinsic = np.zeros((3, 3), dtype=np.float32)
    intrinsic[0, 0] = fx
    intrinsic[1, 1] = fy
    intrinsic[0, 2] = cx
    intrinsic[1, 2] = cy
    intrinsic[2, 2] = 1.0

    rot_mat_inv = np.linalg.inv(rot_mat)
    pmat_rot = np.dot(intrinsic, rot_mat_inv)

    pmat = np.zeros((3, 4), dtype=np.float32)
    pmat[:, :3] = pmat_rot

    C = np.array([ox, oy, oz], dtype=np.float32).reshape(3, -1)
    T = -np.dot(pmat_rot, C)
    pmat[:, -1] = T.reshape(-1)
    return pmat