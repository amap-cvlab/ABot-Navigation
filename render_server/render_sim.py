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
import io
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
from torch import nn
from tqdm import tqdm
from gaussian_renderer.pgsr import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from scene import GaussianModel
import numpy as np
from plyfile import PlyData, PlyElement
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as scipy_R
from scene.colmap_loader import Image, read_cameras_text, read_images_text
from scene.colmap_loader import Camera as CameraDataset
from scene.dataset_readers import readColmapCameras
from scene.cameras import focal2fov, fov2focal, getWorld2View2, getProjectionMatrix, Camera
import collections
import cv2
import requests

from flask import Flask, jsonify, request, send_file, make_response
from flask_cors import CORS
import time
import base64
from concurrent.futures import ThreadPoolExecutor
from global_config import cuda_init_lock
from utils.xyz_euler_trans_gs_colmap_data import xyz_euler_trans_gs_colmap


thread_pool_executor = ThreadPoolExecutor(max_workers=2)
app = Flask(__name__)

allowed_origins = [
    "http://127.0.0.1:7001",
    "http://localhost:7001",
    "http://localhost"
]

# CORS(app, resource={r"/*": {"origins": allowed_origins}})
CORS(app)


MODEL_ARGS = None
ARGS_ITERATION = None
PIPELINE_EXTRACT = None
OP_EXTRACT = None
ARGS = None
ARGS_LOAD_PLY_ROOT = None
GSRENDER_INSTANCE = None

class CameraModel(Camera):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, 
                 uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, cx=0, cy=0, use_cxcy=False,
                 data_device="cuda", proj_info=None,
                 img_res_scale=[1.0, 1.0]):

        #super(CameraModel, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R  # w2c R
        self.T = T  # w2c T
        self.proj_info = proj_info
        self.img_res_scale = img_res_scale
        assert len(self.img_res_scale) == 2, 'img_res_scale must be [scale_x, scale_y] for high accuracy'

        with cuda_init_lock:
            try:
                self.data_device = torch.device(data_device)
            except Exception as e:
                print(e)
                print(f"[WARNING] Custom device {data_device} failed, fallback to default cuda device")
                self.data_device = torch.device("cuda")

            # note: w/h after scale

            self.image_width = proj_info['W']
            self.image_height = proj_info['H']
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

def RT_inv(R, T):
    quaternion = scipy_R.from_matrix(R)
    quaternion = quaternion.inv()
    R_inv = quaternion.as_matrix()
    T_inv = -np.dot(R, T.T).T
    return R_inv, T_inv


def T_scale(R_w2c, T_w2c, scale, center):
    R_c2w, T_c2w = RT_inv(R_w2c, T_w2c)
    T_c2w_scale = (T_c2w - center) * scale + center
    _, T_scaled = RT_inv(R_c2w, T_c2w_scale)
    return T_scaled

def read_images_str(lines):
    images = {}
    for line in lines:
        elems = line.split()
        image_id = int(elems[0])
        qvec = np.array(tuple(map(float, elems[1:5])))
        tvec = np.array(tuple(map(float, elems[5:8])))
        camera_id = int(elems[8])
        image_name = elems[9]
        xys = np.array([])
        point3D_ids = np.array([])
        images[image_id] = Image(
            id=image_id, qvec=qvec, tvec=tvec,
            camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images

def read_cameras_str(lines):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasText(const std::string& path)
        void Reconstruction::ReadCamerasText(const std::string& path)
    """
    cameras = {}
    for line in lines:
        if len(line) > 0 and line[0] != "#":
            elems = line.split()
            camera_id = int(elems[0])
            model = elems[1]
            width = int(elems[2])
            height = int(elems[3])
            params = np.array(tuple(map(float, elems[4:])))
            cameras[camera_id] = CameraDataset(id=camera_id, model=model,
                                        width=width, height=height,
                                        params=params)
    return cameras



class GSRender:
    def __init__(self,
                dataset, 
                iteration, 
                pipeline, 
                opt, 
                args,
                load_ply_root
        ):
        with torch.no_grad():
            # TODO use label input
            self.gaussians_list = []
            self.polygon_list = []
            for block_id in os.listdir(load_ply_root):
                for file in os.listdir(os.path.join(load_ply_root, block_id)):
                    if 'polygon_z' in file:
                        continue
                    if 'polygon_' in file and file.endswith('.txt'):
                        polygon_xyz = [[float(v) for v in l.strip().split(',')] for l in open(os.path.join(load_ply_root, block_id, file)).readlines() if
                                    ',' in l and not l.startswith('#')]
                        polygon_xyz = torch.tensor(polygon_xyz, device=args.device)[:,:2]
                        self.polygon_list.append(polygon_xyz)
                    if file.endswith('.ply'):
                        gaussians = GaussianModel(args.sh_degree, app_opt_cfg=args.app_opt_cfg, use_label=args.use_label)
                        gaussians.load_ply(os.path.join(load_ply_root, block_id, file))
                        self.gaussians_list.append(gaussians)


            bg_color = [0, 0, 0]
            self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

            self.pipeline = pipeline
            self.args = args
            
    def is_point_in_polygons(self, point, polygons: torch.Tensor) -> torch.Tensor:
        """
        判断点是否位于多个多边形中的任意一个
        :param point: 待判断点，形状为 (2,)
        :param polygons: 多边形集合，形状为 (num_polygons, num_vertices, 2)
        :return: 布尔张量，形状为 (num_polygons,), True表示点在该多边形内部或边界上
        """
        # 将点扩展为与多边形相同的batch维度
        point = torch.from_numpy(point).to(polygons.device).to(polygons.dtype)
        point = point.unsqueeze(0).unsqueeze(0)  # (1, 1, 2)
        x, y = point[..., 0], point[..., 1]
        # 获取多边形的边信息
        p1 = polygons  # 所有边的起点
        p2 = torch.roll(polygons, shifts=-1, dims=1)  # 所有边的终点

        # 提取边的坐标
        x1, y1 = p1[..., 0], p1[..., 1]
        x2, y2 = p2[..., 0], p2[..., 1]

        # 计算射线与边的相交条件
        with torch.no_grad():
            # 判断y是否在边的y范围内 ‌:ml-citation{ref="3,6" data="citationList"}
            y_in_range = (y > torch.minimum(y1, y2)) & (y <= torch.maximum(y1, y2))
            
            # 计算交点x坐标 ‌:ml-citation{ref="3,6" data="citationList"}
            x_inters = (y - y1) * (x2 - x1) / (y2 - y1 + 1e-8) + x1
            
            # 处理水平边（分母为0的情况）
            horizontal_edge = (y1 == y2)
            x_inters[horizontal_edge] = -torch.inf  # 使后续条件判断失效

            # 判断点是否在交点的左侧 ‌:ml-citation{ref="3,6" data="citationList"}
            is_left = (x <= x_inters)

            # 综合判断是否相交
            intersect = y_in_range & is_left

        # 统计每个多边形的相交次数
        intersect_count = intersect.sum(dim=1)  # (num_polygons,)

        # 奇数次相交则为内部 ‌:ml-citation{ref="4,6" data="citationList"}
        result = (intersect_count % 2) == 1
        return result[0]

    def render_set(self, cam_extrinsics, cam_intrinsics, camera_render_gs_path):
        pipeline = self.pipeline
        background = self.background
        #scene = self.scene
        args = self.args
        cam_extrinsics = read_images_str(cam_extrinsics)
        cam_intrinsics = read_cameras_str(cam_intrinsics)
        camera_infos = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder='./', force_save_mem = True)
        
        views = []
        for cam_info in camera_infos:
            views.append(
                CameraModel(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                    FoVx=cam_info.FovX, FoVy=cam_info.FovY, uid=cam_info.uid, data_device='cpu',
                    cx=cam_info.cx, cy=cam_info.cy, use_cxcy=cam_info.use_cxcy,
                    proj_info=cam_info.proj_info)
            )
        rendering_np_list = []
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            #contruct view
            T_c2w_3 = view.proj_info['T_c2w_3']
            block_id = 0
            for i in range(len(self.polygon_list)):
                if self.is_point_in_polygons(T_c2w_3[:2], self.polygon_list[i][None]):
                    block_id = i
            gaussians = self.gaussians_list[block_id]
            
            app_color_zero_img_embed = None
            app_color_beta_zero_img_embed = None
            app_opacity_zero_img_embed = None
            if args.app_opt_cfg is not None:
                dir_pp = (gaussians.get_xyz - (view.camera_center.cuda()).unsqueeze(0))
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                app_color_zero_img_embed, app_color_beta_zero_img_embed, app_opacity_zero_img_embed = gaussians.app_module(
                    features=gaussians.get_app_features,
                    embed_ids=None,  # view.uid
                    dirs=dir_pp_normalized,
                    sh_degree=3 if args.app_opt_cfg['app_opt_mode'] == 'dir-sh' else 0,
                )

            out = render(view, gaussians, pipeline, background, app_color=app_color_zero_img_embed,
                        app_color_beta=app_color_beta_zero_img_embed, app_opacity=app_opacity_zero_img_embed,
                        return_plane=False)
            
            rendering = out["render"].clamp(0.0, 1.0)
            _, H, W = rendering.shape
            
            rendering_np = (
                    rendering.permute(1, 2, 0).clamp(0, 1)[:, :, [2, 1, 0]] * 255).detach().cpu().numpy().astype(
                np.uint8)
            
            rendering_np_list.append(rendering_np)
            cv2.imwrite(f"{camera_render_gs_path}/{camera_infos[idx].image_name}.jpg", rendering_np)
        return rendering_np_list

    def render_set_one_image(self, cam_extrinsics, cam_intrinsics, camera_render_gs_path, return_depth_img=False, return_depth_npz=False, render_scale=1.0):
        pipeline = self.pipeline
        background = self.background
        # scene = self.scene
        args = self.args
        cam_extrinsics = read_images_str(cam_extrinsics)
        cam_intrinsics = read_cameras_str(cam_intrinsics)
        camera_infos = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                         images_folder='./', force_save_mem=True)

        cam_info = camera_infos[0]
        image_name = cam_info.image_name
        view = CameraModel(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                    FoVx=cam_info.FovX, FoVy=cam_info.FovY, uid=cam_info.uid, data_device='cuda',
                    cx=cam_info.cx, cy=cam_info.cy, use_cxcy=cam_info.use_cxcy,
                    proj_info=cam_info.proj_info)
        # contruct view
        T_c2w_3 = view.proj_info['T_c2w_3']
        block_id = 0
        for i in range(len(self.polygon_list)):
            if self.is_point_in_polygons(T_c2w_3[:2], self.polygon_list[i][None]):
                block_id = i
        gaussians = self.gaussians_list[block_id]

        app_color_zero_img_embed = None
        app_color_beta_zero_img_embed = None
        app_opacity_zero_img_embed = None
        if args.app_opt_cfg is not None:
            dir_pp = (gaussians.get_xyz - (view.camera_center.cuda()).unsqueeze(0))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            app_color_zero_img_embed, app_color_beta_zero_img_embed, app_opacity_zero_img_embed = gaussians.app_module(
                features=gaussians.get_app_features,
                embed_ids=None,  # view.uid
                dirs=dir_pp_normalized,
                sh_degree=3 if args.app_opt_cfg['app_opt_mode'] == 'dir-sh' else 0,
            )

        # 超采样：render_scale > 1.0 时按放大分辨率渲染；==1.0 时透传 0 表示关闭
        ssaa_scale = float(render_scale) if render_scale and render_scale > 1.0 else 0
        target_w = int(view.image_width)
        target_h = int(view.image_height)

        # 根据是否需要深度信息决定是否返回plane
        need_depth = return_depth_img or return_depth_npz
        out = render(view, gaussians, pipeline, background, app_color=app_color_zero_img_embed,
                     app_color_beta=app_color_beta_zero_img_embed, app_opacity=app_opacity_zero_img_embed,
                     return_plane=need_depth, img_res_scale=ssaa_scale)

        rendering = out["render"].clamp(0.0, 1.0)
        _, H, W = rendering.shape

        rendering_np = (
                rendering.permute(1, 2, 0).clamp(0, 1)[:, :, [2, 1, 0]] * 255).detach().cpu().numpy().astype(
            np.uint8)

        # 超采样后用 INTER_AREA 高质量降采样回原尺寸（等效 SSAA 抗锯齿）
        if ssaa_scale > 0 and (rendering_np.shape[1] != target_w or rendering_np.shape[0] != target_h):
            rendering_np = cv2.resize(rendering_np, (target_w, target_h),
                                      interpolation=cv2.INTER_AREA)
        # 编码RGB图像
        success, encode_image = cv2.imencode('.jpg', rendering_np, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        success, encode_image = cv2.imencode('.jpg', rendering_np)
        if not success:
            raise ValueError('RGB image imencode error.')
        image_bytes = encode_image.tobytes()

        # 构建返回结果
        result = {
            'image_bytes': image_bytes
        }

        # 如果需要深度相关数据
        if need_depth:
            depth_image_arr = out["plane_depth"].squeeze().detach().cpu().numpy()

            # 深度图也按超采样渲染了，需缩回原尺寸；用 INTER_NEAREST 避免插值破坏边界
            if ssaa_scale > 0 and (depth_image_arr.shape[1] != target_w or depth_image_arr.shape[0] != target_h):
                depth_image_arr = cv2.resize(depth_image_arr, (target_w, target_h),
                                             interpolation=cv2.INTER_NEAREST)
            
            # 复制原始深度数组用于npz（保留原始值）
            depth_arr_for_npz = depth_image_arr.copy()

            if return_depth_img:
                # 处理 inf 值，用最大深度值来替代 inf
                depth_for_img = depth_image_arr.copy()
                finite_mask = np.isfinite(depth_for_img)
                if np.any(finite_mask):
                    max_depth = np.nanmax(depth_for_img[finite_mask])
                    depth_for_img[np.isinf(depth_for_img)] = max_depth
                else:
                    depth_for_img[np.isinf(depth_for_img)] = 0

                # 归一化深度图像数据到 0-255 之间
                depth_min = np.min(depth_for_img)
                depth_max = np.max(depth_for_img)
                if depth_max - depth_min > 0:
                    depth_image_normalized = (depth_for_img - depth_min) / (depth_max - depth_min)
                else:
                    depth_image_normalized = np.zeros_like(depth_for_img)
                depth_image_scaled = (depth_image_normalized * 255).astype(np.uint8)

                # 将图像转换为伪彩色
                depth_image_color = cv2.applyColorMap(depth_image_scaled, cv2.COLORMAP_JET)
                
                # 编码深度图像
                success, encode_depth_img = cv2.imencode('.jpg', depth_image_color, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                if not success:
                    raise ValueError('Depth image imencode error.')
                result['depth_img_bytes'] = encode_depth_img.tobytes()

            if return_depth_npz:
                # 将深度数组编码为npz格式的字节流
                npz_buffer = io.BytesIO()
                np.savez_compressed(npz_buffer, depth=depth_arr_for_npz)
                npz_buffer.seek(0)
                result['depth_npz_bytes'] = npz_buffer.getvalue()

        return result


    def render_set_one_depth_image(self, cam_extrinsics, cam_intrinsics):
        pipeline = self.pipeline
        background = self.background
        # scene = self.scene
        args = self.args
        cam_extrinsics = read_images_str(cam_extrinsics)
        cam_intrinsics = read_cameras_str(cam_intrinsics)
        camera_infos = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                         images_folder='./', force_save_mem=True)

        cam_info = camera_infos[0]
        image_name = cam_info.image_name
        view = CameraModel(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                    FoVx=cam_info.FovX, FoVy=cam_info.FovY, uid=cam_info.uid, data_device='cpu',
                    cx=cam_info.cx, cy=cam_info.cy, use_cxcy=cam_info.use_cxcy,
                    proj_info=cam_info.proj_info)
        # contruct view
        T_c2w_3 = view.proj_info['T_c2w_3']
        block_id = 0
        for i in range(len(self.polygon_list)):
            if self.is_point_in_polygons(T_c2w_3[:2], self.polygon_list[i][None]):
                block_id = i
        gaussians = self.gaussians_list[block_id]

        app_color_zero_img_embed = None
        app_color_beta_zero_img_embed = None
        app_opacity_zero_img_embed = None
        if args.app_opt_cfg is not None:
            dir_pp = (gaussians.get_xyz - (view.camera_center.cuda()).unsqueeze(0))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            app_color_zero_img_embed, app_color_beta_zero_img_embed, app_opacity_zero_img_embed = gaussians.app_module(
                features=gaussians.get_app_features,
                embed_ids=None,  # view.uid
                dirs=dir_pp_normalized,
                sh_degree=3 if args.app_opt_cfg['app_opt_mode'] == 'dir-sh' else 0,
            )

        out = render(view, gaussians, pipeline, background, app_color=app_color_zero_img_embed,
                     app_color_beta=app_color_beta_zero_img_embed, app_opacity=app_opacity_zero_img_embed,
                     return_plane=True)

        rendering = out["render"].clamp(0.0, 1.0)
        _, H, W = rendering.shape

        depth_image_arr = out["plane_depth"].squeeze().detach().cpu().numpy()

        rendering_np = (
                rendering.permute(1, 2, 0).clamp(0, 1)[:, :, [2, 1, 0]] * 255).detach().cpu().numpy().astype(
            np.uint8)


        # 处理 inf 值，在这里我们用最大深度值来替代 inf
        max_depth = np.nanmax(depth_image_arr[np.isfinite(depth_image_arr)])  # 找到最大深度
        depth_image_arr[np.isinf(depth_image_arr)] = max_depth  # 将 inf 替换为最大深度

        # 归一化深度图像数据到 0-255 之间
        depth_image_normalized = (depth_image_arr - np.min(depth_image_arr)) / (
                np.max(depth_image_arr) - np.min(depth_image_arr))
        depth_image_scaled = (depth_image_normalized * 255).astype(np.uint8)

        # 将图像转换为伪彩色（可选）
        depth_image_color = cv2.applyColorMap(depth_image_scaled, cv2.COLORMAP_JET)
        success, encode_image = cv2.imencode('.jpg', depth_image_color)
        if not success:
            raise ValueError('Image imencode error.')
        image_bytes = encode_image.tobytes()
        return image_bytes

@dataclass
class GaussianData:
    xyz: np.ndarray
    rot: np.ndarray
    scale: np.ndarray
    opacity: np.ndarray
    sh: np.ndarray

    def flat(self) -> np.ndarray:
        ret = np.concatenate([self.xyz, self.rot, self.scale, self.opacity, self.sh], axis=-1)
        return np.ascontiguousarray(ret)

    def __len__(self):
        return len(self.xyz)

    @property
    def sh_dim(self):
        return self.sh.shape[-1]


def load_gs_ply(path, max_sh_degree=3):
    plydata = PlyData.read(path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])), axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names) == 3 * (max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
    features_extra = np.transpose(features_extra, [0, 2, 1])

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # pass activate function
    xyz = xyz.astype(np.float32)
    # rots = rots / np.linalg.norm(rots, axis=-1, keepdims=True)
    rots = rots.astype(np.float32)
    # scales = np.exp(scales)
    scales = scales.astype(np.float32)
    # opacities = 1/(1 + np.exp(- opacities))  # sigmoid
    opacities = opacities.astype(np.float32)
    shs = np.concatenate([features_dc.reshape(-1, 3),
                          features_extra.reshape(len(features_dc), -1)], axis=-1).astype(np.float32)
    shs = shs.astype(np.float32)
    return GaussianData(xyz, rots, scales, opacities, shs)


def save_gs_ply(path, xyz, rotation, scale, opacities, shs):
    # os.makedirs(os.path.dirname(path), exist_ok=True)
    f_dc, f_rest = shs[:, :3], shs[:, 3:]
    normals = np.zeros_like(xyz)

    construct_list_of_attributes = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'f_dc_0', 'f_dc_1', 'f_dc_2', 'f_rest_0',
                                    'f_rest_1', 'f_rest_2', 'f_rest_3', 'f_rest_4', 'f_rest_5', 'f_rest_6', 'f_rest_7',
                                    'f_rest_8', 'f_rest_9', 'f_rest_10', 'f_rest_11', 'f_rest_12', 'f_rest_13',
                                    'f_rest_14', 'f_rest_15', 'f_rest_16', 'f_rest_17', 'f_rest_18', 'f_rest_19',
                                    'f_rest_20', 'f_rest_21', 'f_rest_22', 'f_rest_23', 'f_rest_24', 'f_rest_25',
                                    'f_rest_26', 'f_rest_27', 'f_rest_28', 'f_rest_29', 'f_rest_30', 'f_rest_31',
                                    'f_rest_32', 'f_rest_33', 'f_rest_34', 'f_rest_35', 'f_rest_36', 'f_rest_37',
                                    'f_rest_38', 'f_rest_39', 'f_rest_40', 'f_rest_41', 'f_rest_42', 'f_rest_43',
                                    'f_rest_44', 'opacity', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2',
                                    'rot_3']

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)

    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)


def save_gs_mini_ply(path, xyz, rotation, scale, opacities, shs):
    # os.makedirs(os.path.dirname(path), exist_ok=True)
    f_dc, f_rest = shs[:, :3], shs[:, 3:]
    normals = np.zeros_like(xyz)

    construct_list_of_attributes = ['x', 'y', 'z', 'nx', 'ny', 'nz',
                                    'f_dc_0', 'f_dc_1', 'f_dc_2',
                                    'opacity',
                                    'scale_0', 'scale_1', 'scale_2',
                                    'rot_0', 'rot_1', 'rot_2', 'rot_3']
    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)

    attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)


def add_logo_to_img(img, logo, logo_offst=(0, 0), bg_color_bgr=(0, 0, 0)):
    h, w, c = logo.shape
    img_h, img_w, _ = img.shape
    logo_x, logo_y = logo_offst
    for row in range(h):
        for col in range(w):
            b, g, r = logo[row][col]
            row_l = row + logo_y
            col_l = col + logo_x
            if row_l < img_h and col_l < img_w:
                if b != bg_color_bgr[0] or g != bg_color_bgr[0] or r != bg_color_bgr[0]:
                    img[row_l][col_l] = (b, g, r)

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)

@app.route('/ping', methods=['GET'])
def ping():
    return "pong"


def make_get_request(url, params):
    try:
        response = requests.get(url, params=params)
        # 检查响应状态码
        if response.status_code == 200:
            # 处理返回的内容
            if response.headers.get('Content-Type') == 'application/json':
                return response.json()  # 返回 JSON 格式的响应
            else:
                # 如果返回做其他类型的数据（如整数），则处理为字典
                return {'result': response.text}  # 将返回内容包装成标准返回格式
        else:
            # 错误处理
            return {'error': f"Error: {response.status_code} - {response.text}"}, response.status_code

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return {'error': str(e)}, 500  # 返回500状态码及错误信息



#和Java状态交互
@app.route('/gs_status', methods=['GET'])
def gs_status():
    taskId = request.args.get('taskId')

    # 接口URL
    url = 'http://127.0.0.1:8001/gs/status'
    params = {'taskId': str(taskId)}  # GET 请求的参数

    result = make_get_request(url, params)
    return jsonify(result)


@app.route('/gs_end_status', methods=['GET'])
def gs_end_status():
    taskId = request.args.get('taskId')

    # 接口URL
    url = 'http://127.0.0.1:8001/gs/endStatus'
    params = {'taskId': str(taskId)}  # GET 请求的参数

    result = make_get_request(url, params)
    return jsonify(result)

@app.route('/gs_end', methods=['GET'])
def gs_end():
    taskId = request.args.get('taskId')
    ip = request.args.get('ip')

    # 接口URL
    url = 'http://127.0.0.1:8001/gs/gsEnd'
    params = {'taskId': str(taskId), 'ip': str(ip)}  # GET 请求的参数

    result = make_get_request(url, params)
    return jsonify(result)

@app.route('/gs_status_now', methods=['GET'])
def gs_status_now():
    taskId = request.args.get('taskId')
    ip = request.args.get('ip')

    # 接口URL
    url = 'http://127.0.0.1:8001/gs/gsStatus'
    params = {'taskId': str(taskId), 'ip': str(ip)}  # GET 请求的参数

    result = make_get_request(url, params)
    return jsonify(result)

@app.route('/init_task', methods=['GET'])
def init_task():
    # 接口URL
    url = 'http://127.0.0.1:8001/gs/initTask'
    params = {}  # GET 请求的参数

    result = make_get_request(url, params)
    return jsonify(result)

@app.route('/render_gs_multi', methods=['POST'])
def render_gs_multi():
    stime = time.time()
    data = request.get_json()
    cam_extrinsics_list = data.get('cam_extrinsics_pair_list')
    cam_intrinsics_list = data.get('cam_intrinsics_pair_list')
    # int(f'cam_extrinsics = {cam_extrinsics}, cam_intrinsics = {cam_intrinsics}')
    camera_render_gs_path = data.get('camera_render_gs_path')
    return_depth_img = data.get('return_depth_img', False)
    return_depth_npz = data.get('return_depth_npz', False)

    result_features = []
    for i in range(len(cam_extrinsics_list)):
        cam_extrinsics = cam_extrinsics_list[i]
        cam_intrinsics = cam_intrinsics_list[i]
        future_res = thread_pool_executor.submit(GSRENDER_INSTANCE.render_set_one_image,
            [cam_extrinsics],
            [cam_intrinsics],
            camera_render_gs_path,
            return_depth_img,
            return_depth_npz
        )
        result_features.append(future_res)

    # 结果获取
    render_results = []
    for future_res in result_features:
        render_results.append(future_res.result())

    etime = time.time()
    cost_time = (etime - stime) * 1000
    print(f'cost time is : {cost_time} ms')
    
    # 返回包含所有图像数据的JSON响应
    if return_depth_img or return_depth_npz:
        response_list = []
        for render_result in render_results:
            item = {
                'image': base64.b64encode(render_result['image_bytes']).decode('utf-8')
            }
            if return_depth_img and 'depth_img_bytes' in render_result:
                item['depth_img'] = base64.b64encode(render_result['depth_img_bytes']).decode('utf-8')
            if return_depth_npz and 'depth_npz_bytes' in render_result:
                item['depth_npz'] = base64.b64encode(render_result['depth_npz_bytes']).decode('utf-8')
            response_list.append(item)
        return jsonify({'results': response_list})
    else:
        return 'ok'


@app.route('/render_gs_xyz_euler', methods=['POST'])
def render_gs_xyz_euler():
    stime = time.time()
    data = request.get_json()
    x = data.get('x')
    y = data.get('y')
    z = data.get('z')
    roll = data.get('roll')
    pitch = data.get('pitch')
    yaw = data.get('yaw')
    degree_flag = data.get('degree_flag')

    cam_extrinsics = xyz_euler_trans_gs_colmap(x, y, z, roll, pitch, yaw, degree_flag)

    cam_intrinsics = data.get('cam_intrinsics')
    # int(f'cam_extrinsics = {cam_extrinsics}, cam_intrinsics = {cam_intrinsics}')
    camera_render_gs_path = data.get('camera_render_gs_path')
    save_image = data.get('save_image', False)
    return_depth_img = data.get('return_depth_img', False)
    return_depth_npz = data.get('return_depth_npz', False)
    # gsrender = GSRender(MODEL_ARGS, ARGS_ITERATION, pipeline=PIPELINE_EXTRACT, opt=OP_EXTRACT, args=ARGS,
    #                     load_ply_root=ARGS_LOAD_PLY_ROOT)
    render_result = GSRENDER_INSTANCE.render_set_one_image(
        [cam_extrinsics],
        [cam_intrinsics],
        camera_render_gs_path,
        return_depth_img=return_depth_img,
        return_depth_npz=return_depth_npz
    )
    
    etime = time.time()
    cost_time = (etime - stime) * 1000
    print(f'cost time is : {cost_time} ms')
    
    # 根据请求返回不同的数据
    if return_depth_img or return_depth_npz:
        # 返回包含所有数据的JSON响应
        response_data = {
            'image': base64.b64encode(render_result['image_bytes']).decode('utf-8')
        }
        if return_depth_img and 'depth_img_bytes' in render_result:
            response_data['depth_img'] = base64.b64encode(render_result['depth_img_bytes']).decode('utf-8')
        if return_depth_npz and 'depth_npz_bytes' in render_result:
            response_data['depth_npz'] = base64.b64encode(render_result['depth_npz_bytes']).decode('utf-8')
        return jsonify(response_data)
    else:
        # 只返回RGB图像
        image_io = io.BytesIO(render_result['image_bytes'])
        return send_file(image_io, mimetype='image/jpeg')


@app.route('/render_gs', methods=['POST'])
def render_gs():
    stime = time.time()
    data = request.get_json()
    cam_extrinsics = data.get('cam_extrinsics')
    cam_intrinsics = data.get('cam_intrinsics')
    # int(f'cam_extrinsics = {cam_extrinsics}, cam_intrinsics = {cam_intrinsics}')
    camera_render_gs_path = data.get('camera_render_gs_path')
    return_depth_img = data.get('return_depth_img', False)
    return_depth_npz = data.get('return_depth_npz', False)
    # gsrender = GSRender(MODEL_ARGS, ARGS_ITERATION, pipeline=PIPELINE_EXTRACT, opt=OP_EXTRACT, args=ARGS,
    #                     load_ply_root=ARGS_LOAD_PLY_ROOT)
    render_result = GSRENDER_INSTANCE.render_set_one_image(
        [cam_extrinsics],
        [cam_intrinsics],
        camera_render_gs_path,
        return_depth_img=return_depth_img,
        return_depth_npz=return_depth_npz
    )
    
    etime = time.time()
    cost_time = (etime - stime) * 1000
    print(f'cost time is : {cost_time} ms')
    
    # 根据请求返回不同的数据
    if return_depth_img or return_depth_npz:
        # 返回包含所有数据的JSON响应
        response_data = {
            'image': base64.b64encode(render_result['image_bytes']).decode('utf-8')
        }
        if return_depth_img and 'depth_img_bytes' in render_result:
            response_data['depth_img'] = base64.b64encode(render_result['depth_img_bytes']).decode('utf-8')
        if return_depth_npz and 'depth_npz_bytes' in render_result:
            response_data['depth_npz'] = base64.b64encode(render_result['depth_npz_bytes']).decode('utf-8')
        return jsonify(response_data)
    else:
        # 只返回RGB图像
        image_io = io.BytesIO(render_result['image_bytes'])
        return send_file(image_io, mimetype='image/jpeg')

@app.route('/render_gs_depth', methods=['POST'])
def render_gs_depth():
    stime = time.time()
    data = request.get_json()
    cam_extrinsics = data.get('cam_extrinsics')
    cam_intrinsics = data.get('cam_intrinsics')
    # int(f'cam_extrinsics = {cam_extrinsics}, cam_intrinsics = {cam_intrinsics}')
    image_bytes = GSRENDER_INSTANCE.render_set_one_depth_image(
        [cam_extrinsics],
        [cam_intrinsics]
    )
    image_io = io.BytesIO(image_bytes)
    etime = time.time()
    cost_time = (etime - stime) * 1000
    print(f'cost time is : {cost_time} ms')
    return send_file(image_io, mimetype='image/jpeg')

@app.route('/render_gs_local', methods=['POST'])
def render_gs_local():
    cam_extrinsics = request.form.get('cam_extrinsics')
    cam_intrinsics = request.form.get('cam_intrinsics')
    camera_render_gs_path = request.form.get('camera_render_gs_path')
    save_image = request.form.get('save_image')
    return_depth_img = request.form.get('return_depth_img', 'false').lower() == 'true'
    return_depth_npz = request.form.get('return_depth_npz', 'false').lower() == 'true'
    print(f'cam_extrinsics = {cam_extrinsics}, cam_intrinsics={cam_intrinsics}, camera_render_gs_path={camera_render_gs_path}')
    # gsrender = GSRender(MODEL_ARGS, ARGS_ITERATION, pipeline=PIPELINE_EXTRACT, opt=OP_EXTRACT, args=ARGS,
    #                     load_ply_root=ARGS_LOAD_PLY_ROOT)
    render_result = GSRENDER_INSTANCE.render_set_one_image(
        [cam_extrinsics],
        [cam_intrinsics],
        camera_render_gs_path,
        return_depth_img=return_depth_img,
        return_depth_npz=return_depth_npz
    )
    
    # 根据请求返回不同的数据
    if return_depth_img or return_depth_npz:
        # 返回包含所有数据的JSON响应
        response_data = {
            'image': base64.b64encode(render_result['image_bytes']).decode('utf-8')
        }
        if return_depth_img and 'depth_img_bytes' in render_result:
            response_data['depth_img'] = base64.b64encode(render_result['depth_img_bytes']).decode('utf-8')
        if return_depth_npz and 'depth_npz_bytes' in render_result:
            response_data['depth_npz'] = base64.b64encode(render_result['depth_npz_bytes']).decode('utf-8')
        return jsonify(response_data)
    else:
        # 只返回RGB图像
        image_io = io.BytesIO(render_result['image_bytes'])
        return send_file(image_io, mimetype='image/jpeg')

if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--load_ply_root", type=str)
    parser.add_argument("--gpus", default="0", type=str)
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("--use_transmittance_alpha", action="store_true")
    parser.add_argument("--use_transmittance_alpha_thro", type=float, default=0.5)
    parser.add_argument("--check_cam_max_dist_thro", type=float, default=3000.0, help='check far cam center')
    parser.add_argument("--img_proj_check_num_limit", type=int, default=0, help='check pts cam proj num or cam num')

    # glo render
    parser.add_argument("--restore_pth", action="store_true")
    parser.add_argument("--with_glo", action="store_true")
    parser.add_argument("--glo_img_num", type=int, default=0, help='')  # TODO for load pth

    parser.add_argument("--app_opt_mode", type=str, default='dir-sh', help='[dir-sh, dir, wild-gs]')
    parser.add_argument("--app_opt_out_mode", type=str, default='color', help='[color, beta, sh, beta-sh, beta-opacity, opacity-sh, beta-sh-opacity]')
    parser.add_argument("--app_opt_img_embed_dim", type=int, default=16, help='per image embed dim')
    parser.add_argument("--app_opt_gs_feature_dim", type=int, default=32, help='per gs feature dim')
    parser.add_argument("--app_opt_mlp_width", type=int, default=64, help=' mlp hidden layer width')
    parser.add_argument("--app_opt_mlp_depth", type=int, default=2, help='mlp layer num default 2')
    parser.add_argument("--use_label", action="store_true")  # TODO for load pth

    parser.add_argument("--render_T_offset", nargs="+", type=float, default=[])
    parser.add_argument("--render_roll_pitch_yaw_offset", nargs="+", type=float, default=[])

    # depth debug
    parser.add_argument("--depth_debug", action="store_true")
    args = get_combined_args(parser)
    # GPU configs
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # Initialize system state (RNG)
    safe_state(args.quiet)
    print(f"multi_view_num {args.pgsr_multi_view_num}")

    args.app_opt_cfg = None
    model_args = model.extract(args)
    MODEL_ARGS = model_args
    ARGS_ITERATION = args.iteration
    PIPELINE_EXTRACT = pipeline.extract(args)
    OP_EXTRACT = op.extract(args)
    ARGS = args
    ARGS_LOAD_PLY_ROOT = args.load_ply_root
    GSRENDER_INSTANCE = GSRender(MODEL_ARGS, ARGS_ITERATION, pipeline=PIPELINE_EXTRACT, opt=OP_EXTRACT, args=ARGS,
             load_ply_root=ARGS_LOAD_PLY_ROOT)
    # gsrender = GSRender(model_args, args.iteration, pipeline=pipeline.extract(args), opt=op.extract(args), args=args, load_ply_root=args.load_ply_root)
    # gsrender.render_set(
    #     ["138 0.6929961794728361 0.6868527312121395 -0.15314485435568045 0.15664059003608935 2.3592032322679723 -1.128928183474061 -0.29175430476615516 1 image_137.jpg"],
    #     ["1 PINHOLE 1280 720 977.3323824654996 977.3323824654996 640.0 360.0"]
    # )
    print('GS Render video ALL DONE!')

    app.run(host="0.0.0.0", debug=True, port=7001)

