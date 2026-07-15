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
import copy
import os
import shutil
import sys
import copy
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import qvec2rotmat
import numpy as np
import torch
from scene.colmap_prepare import  get_pmat, get_RT
from scipy.spatial.transform import Rotation as scipy_R
import math
from utils.graphics_utils import getWorld2View2, focal2fov


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    FoVy: np.array
    FoVx: np.array
    camera_center: np.array
    world_view_transform: np.array
    image: np.array
    image_path: str
    image_name: str
    image_name_ext: str
    image_mask: np.array
    image_mask_path: str
    width: int
    height: int
    cx: int
    cy: int
    proj_info: dict
    use_cxcy: bool
    nearest_id: list
    nearest_names: list
    image_inpainting_mask_path: str
    image_inpainting_mask: np.array
    pre_scale_x: float = 1
    pre_scale_y: float = 1
    aug_pick_id: int = 0

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, images_mask_folder=None, img_mask_ext='_mask.png',
                      force_save_mem=False, use_cxcy=False, inpainting_mask_folder=None):
    print('readColmapCameras Start!')
    cam_infos = []
    all_cam_id = list(cam_intrinsics.keys())
    for idx, key in enumerate(cam_extrinsics):
        # the exact output you're looking for:
        if idx % 100 == 0:
            print("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        qw, qx, qy, qz = extr.qvec
        ox, oy, oz = extr.tvec

        quaternion_w2c = scipy_R.from_quat([qx, qy, qz, qw])
        quat_w2c = np.array([qw, qx, qy, qz])
        T_w2c_3 = np.array([ox, oy, oz])
        R_w2c_33 = quaternion_w2c.as_matrix()
        Rt_w2c = get_RT(R_w2c_33, T_w2c_3)

        quaternion_c2w = quaternion_w2c.inv()
        qx_c2w, qy_c2w, qz_c2w, qw_c2w = quaternion_c2w.as_quat()
        quat_c2w = np.array([qw_c2w, qx_c2w, qy_c2w, qz_c2w])
        roll_c2w, pitch_c2w, yaw_c2w = quaternion_c2w.as_euler("xyz", degrees=True)
        ox_c2w, oy_c2w, oz_c2w = -np.dot(quaternion_c2w.as_matrix(), T_w2c_3)
        T_c2w_3 = np.array([ox_c2w, oy_c2w, oz_c2w])
        R_c2w_33 = quaternion_c2w.as_matrix()
        Rt_c2w = get_RT(R_c2w_33, T_c2w_3)

        uid = extr.id  # Note: use img id as uid instead of intr.id

        # for GS
        # getWorld2View(R_w2c_33.T, T_w2c_3)  # == Rt_w2c
        # getWorld2View2(R_w2c_33.T, T_w2c_3)  # == Rt_w2c if not with scale and trans
        # getView2World(R_w2c_33.T, T_w2c_3)  # == Rt_c2w if not with scale and trans
        # view_world_transform = torch.tensor(getView2World(R_w2c_33.T, T_w2c_3)).transpose(0, 1)  # == Rt_c2w.T
        # view_world_transform[:3, :3].transpose(0, 1)  # == R_c2w_33

        # qw, qx, qy, qz == -1 * scipy_R.from_matrix(R_c2w_33).as_quat() (qx, qy, qz, qw)
        # mq = matrix_to_quaternion(view_world_transform[:3, :3].transpose(0,1))

        R = np.transpose(qvec2rotmat(extr.qvec))  # == R_w2c_33.T
        T = np.array(extr.tvec)  # == T_w2c_3
        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = focal_length_x
            cx, cy = intr.params[1], intr.params[2]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            cx, cy = intr.params[2], intr.params[3]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, extr.name)
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        pmat = np.array(
            get_pmat(roll_c2w, pitch_c2w, yaw_c2w, ox_c2w, oy_c2w, oz_c2w, focal_length_x, focal_length_y, cx, cy))

        proj_info = {'cam_id': extr.camera_id,
                     'all_cam_id': all_cam_id,
                     'quat_w2c': quat_w2c,
                     'T_w2c_3': T_w2c_3,
                     'R_w2c_33': R_w2c_33,
                     'Rt_w2c': Rt_w2c,
                     'quat_c2w': quat_c2w,
                     'T_c2w_3': T_c2w_3,
                     'R_c2w_33': R_c2w_33,
                     'Rt_c2w': Rt_c2w,
                     'H': height,
                     'W': width,
                     'pmat': pmat,
                     'roll_c2w': roll_c2w,
                     'pitch_c2w': pitch_c2w,
                     'yaw_c2w': yaw_c2w,
                     'focal_length_x': focal_length_x,
                     'focal_length_y': focal_length_y,
                     'cx': cx,
                     'cy': cy}

        if force_save_mem:
            # for OSError: [Errno 24] Too many open files or use np.array(image) when cpu mem not enough
            image = None
        else:
            image = Image.open(image_path)
        
        image_inpainting_mask_path=None
        image_inpainting_mask=None
        if inpainting_mask_folder is not None:
            image_inpainting_mask_path = os.path.join(inpainting_mask_folder, os.path.splitext(extr.name)[0]+"_mask.png")
            if force_save_mem:
                image_inpainting_mask = None
            else:
                image_inpainting_mask = Image.open(image_inpainting_mask_path)
                
        image_mask = None
        image_mask_path = None
        if images_mask_folder is not None:
            image_mask_path = os.path.join(images_mask_folder, image_name + img_mask_ext)

            if not os.path.exists(image_mask_path):
                for img_mask_ext in ['_mask.jpg', '.jpg', '.png']:
                    image_mask_path = os.path.join(images_mask_folder, image_name + img_mask_ext)
                    if os.path.exists(image_mask_path):
                        break

            if force_save_mem:
                image_mask = None
            else:
                image_mask = Image.open(image_mask_path)

        world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1)
        camera_center = world_view_transform.inverse()[3, :3]
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, FoVy=FovY, FoVx=FovX,
                              camera_center=camera_center, world_view_transform=world_view_transform,
                              image=image, image_path=image_path, image_name=image_name,
                              image_name_ext=os.path.basename(image_path)
                              , width=width, height=height,
                              image_mask=image_mask, image_mask_path=image_mask_path, cx=cx, cy=cy,
                              proj_info=proj_info, use_cxcy=use_cxcy, nearest_id=list(), nearest_names=list(),
                              image_inpainting_mask_path=image_inpainting_mask_path, image_inpainting_mask=image_inpainting_mask)
        # image.close()
        cam_infos.append(cam_info)
    print('readColmapCameras Done!')
    return cam_infos

