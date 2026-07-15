import torch 
import sys
from datetime import datetime
import numpy as np
import random

def inverse_sigmoid(x):
    return torch.log(x / (1 - x))

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L

def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 0, None], q1[..., 1, None], q1[..., 2, None], q1[..., 3, None]
    w2, x2, y2, z2 = q2[..., 0, None], q2[..., 1, None], q2[..., 2, None], q2[..., 3, None]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.hstack([w, x, y, z])

def safe_state(silent):
    old_f = sys.stdout

    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))

def flip_align_view(normal, viewdir):
    # normal: (N, 3), viewdir: (N, 3)
    dotprod = torch.sum(
        normal * -viewdir, dim=-1, keepdims=True)  # (N, 1)
    non_flip = dotprod >= 0  # (N, 1)
    normal_flipped = normal * torch.where(non_flip, 1, -1)  # (N, 3)
    return normal_flipped, non_flip
def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    from: https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#quaternion_to_matrix
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def load_animation_npz(filename, device='cuda:0'):
    """
    加载动画NPZ文件
    
    Args:
        filename: NPZ文件路径
        device: 目标设备
    Returns:
        dict: 包含所有加载数据的字典
    """
    # 加载NPZ文件
    data = np.load(filename, allow_pickle=True)
    # 获取gs_attr数据
    gs_attr = data['gs_attr'].item()  # 注意这里需要.item()因为是对象数组
    gs_data = {
        'offset_xyz': torch.tensor(gs_attr['offset_xyz'], device=device),
        'opacity': torch.tensor(gs_attr['opacity'], device=device),
        'rotation': torch.tensor(gs_attr['rotation'], device=device),
        'scaling': torch.tensor(gs_attr['scaling'], device=device),
        'shs': torch.tensor(gs_attr['shs'], device=device),
        'use_rgb': gs_attr['use_rgb']
    }
    
    # 获取query_pt数据
    query_pt = torch.tensor(data['query_pt'], device=device)

    # gs_data = {
    #     'offset_xyz': torch.tensor(gs_attr['offset_xyz'], device=device),
    #     'opacity': torch.tensor(gs_attr['opacity'], device=device),
    #     'rotation': torch.tensor(gs_attr['rotation'], device=device),
    #     'scaling': torch.tensor(gs_attr['scaling'], device=device),
    #     'shs': torch.tensor(gs_attr['shs'], device=device),
    #     'use_rgb': gs_attr['use_rgb']
    # }
    
    # # 获取query_pt数据
    # query_pt = torch.tensor(data['query_pt'], device=device)
    
    # 获取smplx_data数据
    smplx_data = data['smplx_data'].item()  # 同样需要.item()
    smplx_data = {k: torch.tensor(v, device=device) for k, v in smplx_data.items()}
    # 返回所有数据
    return {
        'gs_attr': gs_data,
        'query_pt': query_pt,
        'smplx_data': smplx_data
    }

