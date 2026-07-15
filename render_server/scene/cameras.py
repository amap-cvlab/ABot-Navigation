#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, fov2focal, focal2fov
import copy

from scene.colmap_prepare import get_pmat
from scipy.spatial.transform import Rotation as scipy_R
import os


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask, image_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, cx=0, cy=0, use_cxcy=False,
                 data_device="cuda", image_path=None, proj_info=None,
                 img_res_scale=[1.0, 1.0], image_gray=None, ncc_scale=0.0, nearest_id=None, nearest_names=None
                 , inpainting_mask=None):

        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R  # w2c R
        self.T = T  # w2c T
        self.image_name = image_name
        self.image_path = image_path
        self.image_name_ext = os.path.basename(image_path)
        self.proj_info = proj_info
        self.img_res_scale = img_res_scale  # ori img to scaled img
        assert len(self.img_res_scale) == 2, 'img_res_scale must be [scale_x, scale_y] for high accuracy'
        self.ncc_scale = ncc_scale  # ori img to scaled ncc img

        if nearest_id is None:
            self.nearest_id = []
        else:
            self.nearest_id = nearest_id

        if nearest_names is None:
            self.nearest_names = []
        else:
            self.nearest_names = nearest_names

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[WARNING] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        # note: w/h after scale
        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_gray = None
        if image_gray is not None:
            self.image_gray = image_gray.clamp(0.0, 1.0).to(self.data_device)

        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        self.resolution = (self.image_width, self.image_height)

        if img_res_scale[0] < 1 or img_res_scale[1] < 1:
            # img_res_scale is origin(data folder) img size to scale img size
            assert (int(self.proj_info['H'] * img_res_scale[1] - self.image_height) <= 1 and
                    int(self.proj_info['W'] * img_res_scale[0] - self.image_width) <= 1),\
                'ERROR: camera.py img_res_scale Error! ori_H:{}, img_H:{}, ori_W:{}, img_W:{}, scale:{}'.format(
                    self.proj_info['H'], self.image_height, self.proj_info['W'], self.image_width, img_res_scale)
            cx = cx * img_res_scale[0]
            cy = cy * img_res_scale[1]
            fx = self.proj_info['focal_length_x'] * img_res_scale[0]
            fy = self.proj_info['focal_length_y'] * img_res_scale[1]
            FoVx = focal2fov(fx, self.image_width)
            FoVy = focal2fov(fy, self.image_height)
        else:
            fx = fov2focal(FoVx, self.image_width)
            fy = fov2focal(FoVy, self.image_height)

        self.FoVx = FoVx
        self.FoVy = FoVy
        self.fx = fx
        self.fy = fy
        # self.Cx = 0.5 * self.image_width
        # self.Cy = 0.5 * self.image_height

        # 0 -> -1, width -> 1, width/2 -> 0, 2width -> 3
        # 0 -> -1, -width -> -3, -width/2 -> -2, -2width -> -5
        self.use_cxcy = use_cxcy
        self.cx = (2 * self.proj_info['cx'] / self.proj_info['W']) - 1  # default 0, use when args.use_cxcy True
        self.cy = (2 * self.proj_info['cy'] / self.proj_info['H']) - 1  # default 0, use when args.use_cxcy True
        self.cx_pixel = cx
        self.cy_pixel = cy
        # print(image_name, self.use_cxcy, cx, self.cx, cy, self.cy)

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.image_mask = None
        if image_mask is not None:
            self.image_mask = image_mask.clamp(0.0, 1.0).to(self.data_device)
        
        self.inpainting_mask = None
        if inpainting_mask is not None:
            self.inpainting_mask = inpainting_mask.clamp(0.0, 1.0).to(self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        # getWorld2View2(R, T, trans, scale).T
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1) # .cuda()

        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy,
                                                     cx=self.cx, cy=self.cy, use_cxcy=self.use_cxcy).transpose(0, 1)  # .cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def get_scaled_pmat(self, scale):
        ox_c2w, oy_c2w, oz_c2w = self.proj_info['T_c2w_3']
        return get_pmat(self.proj_info['roll_c2w'], self.proj_info['pitch_c2w'], self.proj_info['yaw_c2w'],
                 ox_c2w, oy_c2w, oz_c2w,
                 self.proj_info['focal_length_x'] * scale[0], self.proj_info['focal_length_y'] * scale[1]
                 , self.proj_info['cx'] * scale[0], self.proj_info['cy'] * scale[1])

    def get_offset_matrix(self, T_offset=None, roll_pitch_yaw_offset=None, device='cuda:0'):

        R_c2w_33 = self.proj_info['R_c2w_33']  # self.proj_info['R_w2c_33'].T == self.R
        T_c2w_3 = self.proj_info['T_c2w_3']
        if T_offset is not None:
            T_c2w_3 = T_c2w_3 + T_offset  # self.proj_info['T_w2c_3'] == self.T

        if roll_pitch_yaw_offset is not None:
            q_c2w = scipy_R.from_matrix(R_c2w_33)
            roll_pitch_yaw_c2w = q_c2w.as_euler('xyz', degrees=True)
            roll_pitch_yaw_c2w = roll_pitch_yaw_c2w + roll_pitch_yaw_offset
            q_c2w = scipy_R.from_euler(seq='xyz', angles=roll_pitch_yaw_c2w, degrees=True)
            R_c2w_33 = q_c2w.as_matrix()

        # inv to w2c R T
        quaternion = scipy_R.from_matrix(R_c2w_33)
        quaternion_w2c = quaternion.inv()
        R_w2c_33 = quaternion_w2c.as_matrix()
        # inv to w2c xyz
        T_w2c_3 = -np.dot(R_w2c_33, T_c2w_3.T).T

        # get gs render input
        # self.proj_info['Rt_w2c'] == getWorld2View2(R_w2c_33.T, T_w2c_3, self.trans, self.scale)
        # world_view_transform == self.proj_info['Rt_w2c'](row to col)
        world_view_transform = torch.tensor(getWorld2View2(R_w2c_33.T, T_w2c_3, self.trans, self.scale)).transpose(0, 1)

        # K = np.zeros((3, 3), dtype=np.float64)
        # K[0, 0], K[1, 1], K[0, 2], K[1, 2], K[2, 2] = self.proj_info['focal_length_x'], self.proj_info['focal_length_y'], self.proj_info['cx'], self.proj_info['cy'], 1.0
        projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, cx=self.cx, cy=self.cy, use_cxcy=self.use_cxcy).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]  # == T_c2w_3

        # TODO del debug code
        # import pdb
        # pdb.set_trace()
        # NDC_2_pixel = torch.tensor([[self.image_width / 2, 0, self.image_width / 2], [0, self.image_height / 2, self.image_height / 2], [0, 0, 1]]).cuda()
        # NDC_2_pixel = NDC_2_pixel.float()
        # cam_1_tranformation = full_proj_transform[:, [0, 1, 3]].T.float().cuda()
        # cam_1_pixel = NDC_2_pixel @ cam_1_tranformation
        # cam_1_pixel = cam_1_pixel.float() # == pmat

        return world_view_transform.to(device), full_proj_transform.to(device), camera_center.to(device)

    def get_matrix(self, R, T, device='cuda:0', inv=False):

        if inv:
            quaternion = scipy_R.from_matrix(R)
            quaternion = quaternion.inv()
            R = quaternion.as_matrix()
            T = -np.dot(R, T.T).T

        # cam R T to cameras.json R T
        # Rt = np.zeros((4, 4))
        # Rt[:3, :3] = R.transpose()
        # Rt[:3, 3] = T
        # Rt[3, 3] = 1.0
        # Rt_inv = np.linalg.inv(Rt)
        # T = Rt_inv[:3, 3]
        # R = Rt_inv[:3, :3]

        # get gs render input
        world_view_transform = torch.tensor(getWorld2View2(R, T, self.trans, self.scale)).transpose(0, 1)
        projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
                                                fovY=self.FoVy, cx=self.cx, cy=self.cy,
                                                use_cxcy=self.use_cxcy).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        return world_view_transform.to(device), full_proj_transform.to(device), camera_center.to(device)

    def RT_inv(self, R, T):
        quaternion = scipy_R.from_matrix(R)
        quaternion = quaternion.inv()
        R_inv = quaternion.as_matrix()
        T_inv = -np.dot(R_inv, T.T).T
        return R_inv, T_inv

    def reset_extrinsic(self, R, T):
        # w2c R and w2c T
        self.R = R
        self.T = T
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, self.trans, self.scale)).transpose(0, 1)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def T_scale(self, scale, center):
        R_c2w, T_c2w = self.RT_inv(self.R, self.T)
        T_c2w_scale = (T_c2w - center) * scale + center
        T_scaled = self.RT_inv(R_c2w, T_c2w_scale)
        return T_scaled

    def get_image(self, return_gray=False):
        if return_gray:
            return self.original_image.cuda(), self.image_gray.cuda()
        else:
            return self.original_image.cuda()

    def clean_mem(self):
        del self.original_image
        self.original_image = None
        del self.image_gray
        self.image_gray = None
        del self.image_mask
        self.image_mask = None

    def get_calib_matrix_nerf(self, scale=1.0):
        intrinsic_matrix = torch.tensor(
            [[self.fx / scale, 0, self.cx_pixel / scale], [0, self.fy / scale, self.cy_pixel / scale], [0, 0, 1]]).float()
        extrinsic_matrix = self.world_view_transform.transpose(0, 1).contiguous()  # cam2world
        return intrinsic_matrix, extrinsic_matrix

    def get_rays(self, scale=1.0, mask = None):
        W, H = int(self.image_width / scale), int(self.image_height / scale)
        ix, iy = torch.meshgrid(
            torch.arange(W), torch.arange(H), indexing='xy')
        rays_d = torch.stack(
            [(ix - self.cx_pixel / scale) / self.fx * scale,
             (iy - self.cy_pixel / scale) / self.fy * scale,
             torch.ones_like(ix)], -1).float().cuda()
        if mask is not None:
            rays_d = rays_d[mask]
        return rays_d

    def get_k(self, scale=1.0):
        K = torch.tensor([[self.fx / scale, 0, self.cx_pixel / scale],
                          [0, self.fy / scale, self.cy_pixel / scale],
                          [0, 0, 1]]).to(torch.float32).cuda()
        return K

    def get_inv_k(self, scale=1.0):
        K_T = torch.tensor([[scale / self.fx, 0, -self.cx_pixel / self.fx],
                            [0, scale / self.fy, -self.cy_pixel / self.fy],
                            [0, 0, 1]]).to(torch.float32).cuda()
        return K_T


