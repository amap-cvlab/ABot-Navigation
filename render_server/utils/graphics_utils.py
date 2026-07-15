import math
import numpy as np 
import torch 
from typing import NamedTuple
from global_config import cuda_init_lock

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY, cx=0, cy=0, use_cxcy=False):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    if use_cxcy:
        P[0, 2] = (right + left) / (right - left) + cx
        P[1, 2] = (top + bottom) / (top - bottom) + cy
    else:
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def ndc_2_cam(ndc_xyz, intrinsic, W, H):
    inv_scale = torch.tensor([[W - 1, H - 1]], device=ndc_xyz.device)
    cam_z = ndc_xyz[..., 2:3]
    cam_xy = ndc_xyz[..., :2] * inv_scale * cam_z
    cam_xyz = torch.cat([cam_xy, cam_z], dim=-1)

    # 先计算、然后转换为常量，以避免懒惰错误
    intrinsic_matrix = intrinsic[0, ...].t()

    # 在锁的保护下执行CUDA操作
    with cuda_init_lock:
        inv_intrinsic = torch.inverse(intrinsic_matrix)

    cam_xyz = cam_xyz @ inv_intrinsic

    return cam_xyz


def depth2point_cam(sampled_depth, ref_intrinsic):
    B, N, C, H, W = sampled_depth.shape
    valid_z = sampled_depth
    valid_x = torch.arange(W, dtype=torch.float32, device=sampled_depth.device) / (W - 1)
    valid_y = torch.arange(H, dtype=torch.float32, device=sampled_depth.device) / (H - 1)
    valid_y, valid_x = torch.meshgrid(valid_y, valid_x)
    # B,N,H,W
    valid_x = valid_x[None, None, None, ...].expand(B, N, C, -1, -1)
    valid_y = valid_y[None, None, None, ...].expand(B, N, C, -1, -1)
    ndc_xyz = torch.stack([valid_x, valid_y, valid_z], dim=-1).view(B, N, C, H, W, 3)  # 1, 1, 5, 512, 640, 3
    cam_xyz = ndc_2_cam(ndc_xyz, ref_intrinsic, W, H)  # 1, 1, 5, 512, 640, 3
    return ndc_xyz, cam_xyz

def depth2point_world(depth_image, intrinsic_matrix, extrinsic_matrix):
    # depth_image: (H, W), intrinsic_matrix: (3, 3), extrinsic_matrix: (4, 4)
    _, xyz_cam = depth2point_cam(depth_image[None, None, None, ...], intrinsic_matrix[None, ...])
    xyz_cam = xyz_cam.reshape(-1, 3)
    # xyz_world = torch.cat([xyz_cam, torch.ones_like(xyz_cam[...,0:1])], axis=-1) @ torch.inverse(extrinsic_matrix).transpose(0,1)
    # xyz_world = xyz_world[...,:3]

    return xyz_cam


def depth_pcd2normal(xyz, offset=None, gt_image=None):
    hd, wd, _ = xyz.shape
    if offset is not None:
        ix, iy = torch.meshgrid(
            torch.arange(wd), torch.arange(hd), indexing='xy')
        xy = (torch.stack((ix, iy), dim=-1)[1:-1, 1:-1]).to(xyz.device)
        p_offset = torch.tensor([[0, 1], [0, -1], [1, 0], [-1, 0]]).float().to(xyz.device)
        new_offset = p_offset[None, None] + offset.reshape(hd, wd, 4, 2)[1:-1, 1:-1]
        xys = xy[:, :, None] + new_offset
        xys[..., 0] = 2 * xys[..., 0] / (wd - 1) - 1.0
        xys[..., 1] = 2 * xys[..., 1] / (hd - 1) - 1.0
        sampled_xyzs = torch.nn.functional.grid_sample(xyz.permute(2, 0, 1)[None], xys.reshape(1, -1, 1, 2))
        sampled_xyzs = sampled_xyzs.permute(0, 2, 3, 1).reshape(hd - 2, wd - 2, 4, 3)
        bottom_point = sampled_xyzs[:, :, 0]
        top_point = sampled_xyzs[:, :, 1]
        right_point = sampled_xyzs[:, :, 2]
        left_point = sampled_xyzs[:, :, 3]
    else:
        bottom_point = xyz[..., 2:hd, 1:wd - 1, :]
        top_point = xyz[..., 0:hd - 2, 1:wd - 1, :]
        right_point = xyz[..., 1:hd - 1, 2:wd, :]
        left_point = xyz[..., 1:hd - 1, 0:wd - 2, :]
    left_to_right = right_point - left_point
    bottom_to_top = top_point - bottom_point
    xyz_normal = torch.cross(left_to_right, bottom_to_top, dim=-1)
    xyz_normal = torch.nn.functional.normalize(xyz_normal, p=2, dim=-1)
    xyz_normal = torch.nn.functional.pad(xyz_normal.permute(2, 0, 1), (1, 1, 1, 1), mode='constant').permute(1, 2, 0)
    return xyz_normal


def normal_from_depth_image(depth, intrinsic_matrix, extrinsic_matrix, offset=None, gt_image=None):
    # depth: (H, W), intrinsic_matrix: (3, 3), extrinsic_matrix: (4, 4)
    # xyz_normal: (H, W, 3)
    xyz_world = depth2point_world(depth, intrinsic_matrix, extrinsic_matrix)  # (HxW, 3)
    xyz_world = xyz_world.reshape(*depth.shape, 3)
    xyz_normal = depth_pcd2normal(xyz_world, offset, gt_image)

    return xyz_normal

class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array
    labels: np.array = None
    min_cam_dis: np.array = None
    min_dis_cam_uid: np.array = None
    min_dis_cam_center: np.array = None