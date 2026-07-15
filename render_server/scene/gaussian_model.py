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
import numpy as np
from utils.general_utils import inverse_sigmoid
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, quaternion_multiply, build_scaling_rotation, build_rotation, quaternion_to_matrix
import json
import e3nn
from scipy.spatial.transform import Rotation

# TODO
# try:
#     from diff_gaussian_rasterization import SparseGaussianAdam
# except:
#     pass


def vector_to_quaternion(from_vector, to_vector):
    from_vector = from_vector / (from_vector.norm(dim=-1).clamp_min(1e-15)[..., None])
    to_vector = to_vector / (to_vector.norm(dim=-1).clamp_min(1e-15)[..., None])
    axis = torch.cross(from_vector, to_vector)
    axis = axis / (axis.norm(dim=-1).clamp_min(1e-15)[..., None])
    angle = torch.acos((from_vector * to_vector).sum(-1))[..., None]

    half_angle = angle / 2.0
    w = torch.cos(half_angle)
    xyz = axis * torch.sin(half_angle)
    return torch.hstack((w, xyz))


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2 """
    a, b = vec1 / np.linalg.norm(vec1, axis=-1)[:, None], vec2 / (np.linalg.norm(vec2, axis=-1)[:, None] + 1e-6)
    v = np.cross(a, b)
    c = (a * b).sum(-1)[:, None]
    s = np.linalg.norm(v, axis=-1)[:, None]

    kmat = np.zeros((vec1.shape[0], 3, 3))
    kmat[:, 0, 1] = -v[:, 2]
    kmat[:, 0, 2] = v[:, 1]
    kmat[:, 1, 0] = v[:, 2]
    kmat[:, 1, 2] = -v[:, 0]
    kmat[:, 2, 0] = -v[:, 1]
    kmat[:, 2, 1] = v[:, 0]
    rotation_matrix = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2 + 1e-8))[:, None]
    return rotation_matrix


def normal2grotation(normal, wxyz=True):
    z = np.zeros_like(normal)
    z[..., 2] = 1
    # rotate vector from z to normal
    sv, c = np.cross(z, normal), (z * normal).sum(-1, keepdims=True)
    s = np.linalg.norm(sv, axis=-1, keepdims=True)
    theta = np.arctan2(s, c)
    v = sv / np.where(s > 1e-6, s, 1)  # Be Careful, invalid value (nan) when divide 0
    sv, c = np.sin(theta / 2) * v, np.cos(theta / 2)
    q = np.concatenate([sv, c], axis=-1)
    if wxyz:
        q = np.roll(q, 1, axis=-1)  # xyzw -> wxyz(gaussian style)
    return q


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def set_scale_clip_max(self, scale_clip_max):
        self.scale_clip_max = scale_clip_max

    def __init__(self, sh_degree, asg_degree=0, opacity_init_value=0.1, scale_init=1,
                 use_c3d_mask=False, use_label=False, ground_cfg=None, record_pts_status=False,
                 use_min_cam_dis_weight=False, min_cam_dis_spatial_scale=None, use_opacity_distance_decay=False,
                 opacity_distance_decay_init_mode='same_value',
                 use_norm=False, use_norm_init=False, scale_clip_max=0.0, pts_label_enum=None,
                 use_color_affine=False, use_position_affine=False,
                 use_scene_box=False, scene_box_scale_multiplier=1.5,
                 c3d_mask_gs_num_limit=1, use_asg=False, app_opt_cfg=None, bilateral_grid_cfg=None,
                 densify_split_backend='gs', densify_clone_backend='gs', use_rot_angles=False,
                 use_vignet=False, use_blur_opt=False):

        assert densify_split_backend in ['gs', 'abs-gs', 'mcmc'], 'GaussianModel unsupport densify_split_backend {}'.format(
            densify_split_backend)
        assert densify_clone_backend in ['gs', 'gof', 'mcmc'], 'GaussianModel unsupport densify_clone_backend {}'.format(
            densify_clone_backend)

        if densify_clone_backend == 'mcmc' or densify_split_backend == 'mcmc':
            from utils.reloc_utils import compute_relocation_cuda
            self.compute_relocation_cuda = compute_relocation_cuda



        self.densify_split_backend = densify_split_backend
        self.densify_clone_backend = densify_clone_backend
        self.use_rot_angles = use_rot_angles  # single freedom rot(z-axis)

        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        if self.use_rot_angles:
            self._rot_angles = torch.empty(0)
            self.norm_quat = None
        else:
            self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)

        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)

        if densify_split_backend == 'abs-gs':
            self.xyz_gradient_accum_abs = torch.empty(0)
            self.denom_abs = torch.empty(0)

        self.optimizer = None
        self.percent_dense_split = 0
        self.percent_dense_clone = 0
        self.spatial_lr_scale = 0

        self.setup_functions()

        # add configs
        self.use_label = use_label
        if self.use_label:
            self.points_label = torch.empty(0)
        self.pts_label_enum = pts_label_enum

        self.use_scene_box = use_scene_box
        self.scene_box_scale_multiplier = scene_box_scale_multiplier
        self.scale_clip_max = scale_clip_max
        self.use_norm = use_norm
        self.use_norm_init = use_norm_init
        self.use_color_affine = use_color_affine
        self.use_position_affine = use_position_affine
        self.ground_cfg = ground_cfg

        # c3d config
        self.use_c3d_mask = use_c3d_mask
        self.c3d_mask_gs_num_limit = c3d_mask_gs_num_limit
        if self.use_c3d_mask:
            self._c3d_mask = torch.empty(0)

        # opacity_distance_decay config
        self.use_opacity_distance_decay = use_opacity_distance_decay
        if self.use_opacity_distance_decay:
            self._opacity_distance_decay = torch.empty(0)
        self.opacity_distance_decay_init_mode = opacity_distance_decay_init_mode
        assert self.opacity_distance_decay_init_mode in ['same_value', 'min_cam_dis'], \
            'Error, unsupport opacity_distance_decay_init_mode!!!'
        self.opacity_init_value = opacity_init_value
        self.scale_init = scale_init

        # load global min_cam_dis from ply to make depth mask
        self.use_min_cam_dis_weight = use_min_cam_dis_weight
        self.points_min_cam_dis_unique = None
        self.min_cam_dis_spatial_scale = None
        if self.use_min_cam_dis_weight:
            self.points_min_cam_dis = torch.empty(0)
            self.min_cam_dis_spatial_scale = min_cam_dis_spatial_scale  # for gs scale init
        self.points_min_dis_cam_uid = None
        self.points_min_dis_cam_center = None

        # record split/clone and prune for debug
        self.record_pts_status = record_pts_status
        if self.record_pts_status:
            self.points_prune = None  # ori and densify
            self.points_prune_ori = None  # ori

        self.use_asg = use_asg
        if self.use_asg:
            self.max_asg_degree = asg_degree
            self._features_asg = torch.empty(0)

        self.app_opt_cfg = app_opt_cfg
        if self.app_opt_cfg is not None:
            self._features_app = torch.empty(0)
            self.app_module = None
            self.app_optimizers = list()

        self.bilateral_grid_cfg = bilateral_grid_cfg
        if self.bilateral_grid_cfg is not None:
            self.bilateral_grid_module = None
            self.bilateral_grid_optimizers = list()
            self.bilateral_grid_scheduler_args = None

            from utils.lib_bilagrid import (
                bil_grids_slice,
                bil_grids_color_correct,
                total_variation_loss,
            )

            self.bil_grids_slice = bil_grids_slice
            self.bil_grids_color_correct = bil_grids_color_correct
            self.total_variation_loss = total_variation_loss

        self.use_blur_opt = use_blur_opt

        # ntc
        self._d_color = None

        # 2d img modify
        self.use_vignet = use_vignet

        # others
        self.warning_times = 3
        self._xyz_bound_min = None
        self._xyz_bound_max = None


    @property
    def get_scaling(self):
        if self.scale_clip_max is not None and self.scale_clip_max > 0:
            if not self.use_scene_box:
                return torch.clip(self.scaling_activation(self._scaling), max=self.scale_clip_max)
            else:
                scenebox_mask = self.points_label == self.pts_label_enum.Scenebox
                return torch.where(scenebox_mask, self.scaling_activation(self._scaling),
                                   torch.clip(self.scaling_activation(self._scaling), max=self.scale_clip_max))
        else:
            return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        if self.use_rot_angles:
            # return self.rotation_activation(self._rotation)
            axis = torch.zeros((len(self._rot_angles), 3)).float().cuda()
            axis[:, 2] = 1
            rot_angles = e3nn.o3.axis_angle_to_quaternion(axis, self._rot_angles[:, 0])  # wxyz
            # self.rots: wxyz, from gaussian to world
            rotation = e3nn.o3.compose_quaternion(self.norm_quat, rot_angles)
            return self.rotation_activation(rotation)
        else:
            return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_features_dc(self):
        return self._features_dc

    @property
    def get_features_rest(self):
        return self._features_rest

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        if self.use_rot_angles:
            return self.covariance_activation(self.get_scaling, scaling_modifier, self.get_rotation)
        else:
            return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    @property
    def get_app_features(self):
        return self._features_app

    @property
    def get_asg_features(self):
        return self._features_asg

    @property
    def get_norm_cfd(self):
        return torch.sigmoid(self._norm_cfd)

    @property
    def get_min_scale_index(self):
        return self._scaling.argmin(-1)


    @property
    def get_poly_points(self, in_poly_label=1, ground_label=3):
        poly_flag = torch.logical_or(self.points_label % 100 == in_poly_label,
                                     self.points_label % 100 == ground_label).squeeze()
        # poly_flag = (self.points_label % 100 == in_poly_label).squeeze()

        if poly_flag.sum() > 0:
            return poly_flag
        else:
            if self.warning_times > 0:
                print("WARNING: get_poly_points, in poly flag is all False and make all True for use")
                self.warning_times -= 1

            return ~poly_flag

    def clip_grad(self, norm=1.0, value=0):
        for group in self.optimizer.param_groups:
            torch.nn.utils.clip_grad_norm_(group["params"][0], norm)  # norm_type=2
            if value > 0:
                torch.nn.utils.clip_grad_value_(group["params"][0], value)

    def check_grad_nan(self):
        for group in self.optimizer.param_groups:
            # print('check_grad_nan:', group['name'], torch.isnan(group["params"][0]).any())
            if torch.isnan(group["params"][0]).any():
                print('WARNING gaussian_model check_grad_nan: nan in grad: ', group)
                return True
        return False

    def check_gs_nan(self):
        nan_mask = None

        if torch.isnan(self.get_xyz).any().item():
            nan_mask_xyz = torch.sum(torch.isnan(self.get_xyz), dim=-1) > 0
            nan_mask = nan_mask_xyz
            print('gaussian model check_gs_nan xyz have nan: ({}/{})'
                  .format(torch.sum(nan_mask_xyz), self.get_xyz.shape[0]))
        if torch.isnan(self.get_scaling).any().item():
            nan_mask_scale = torch.sum(torch.isnan(self.get_scaling), dim=-1) > 0
            nan_mask = torch.logical_and(nan_mask, nan_mask_scale)
            print('gaussian model check_gs_nan scaling have nan: ({}/{})'
                  .format(torch.sum(nan_mask_scale), self.get_xyz.shape[0]))
        if torch.isnan(self.get_features).any().item():
            nan_mask_fts = torch.sum(torch.reshape(torch.isnan(self.get_features),
                                                   (self.get_features.shape[0], -1)), dim=-1) > 0
            nan_mask = torch.logical_and(nan_mask, nan_mask_fts)
            print('gaussian model check_gs_nan features have nan: ({}/{})'
                  .format(torch.sum(nan_mask_fts), self.get_xyz.shape[0]))
        if torch.isnan(self.get_opacity).any().item():
            nan_mask_opacity = torch.sum(torch.isnan(self.get_opacity), dim=-1) > 0
            nan_mask = torch.logical_and(nan_mask, nan_mask_opacity)

            print('gaussian model check_gs_nan opacity have nan: ({}/{})'
                  .format(torch.sum(nan_mask_opacity), self.get_xyz.shape[0]))
        if torch.isnan(self.get_rotation).any().item():
            nan_mask_rotation = torch.sum(torch.isnan(self.get_rotation), dim=-1) > 0
            nan_mask = torch.logical_and(nan_mask, nan_mask_rotation)
            print('gaussian model check_gs_nan rotation have nan: ({}/{})'
                  .format(torch.sum(nan_mask_rotation), self.get_xyz.shape[0]))

        if self.app_opt_cfg is not None and torch.isnan(self.get_app_features).any().item():
            nan_mask_app_fts = torch.sum(torch.isnan(self.get_app_features), dim=-1) > 0
            nan_mask = torch.logical_and(nan_mask, nan_mask_app_fts)
            print('gaussian model check_gs_nan app feature have nan: ({}/{})'
                  .format(torch.sum(nan_mask_app_fts), self.get_xyz.shape[0]))

        if nan_mask is None:
            return False, nan_mask
        else:
            return True, nan_mask

    def get_contracted_xyz(self):
        with torch.no_grad():
            xyz = self.get_xyz
            xyz_bound_min, xyz_bound_max = self.get_xyz_bound(86.6)
            normalzied_xyz = (xyz - xyz_bound_min) / (xyz_bound_max - xyz_bound_min)
            return normalzied_xyz

    def get_xyz_bound(self, percentile=86.6):
        with torch.no_grad():
            if self._xyz_bound_min is None:
                half_percentile = (100 - percentile) / 200
                self._xyz_bound_min = torch.quantile(self._xyz, half_percentile, dim=0)
                self._xyz_bound_max = torch.quantile(self._xyz, 1 - half_percentile, dim=0)
            return self._xyz_bound_min, self._xyz_bound_max
    
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        # for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
        #     l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        if self.use_rot_angles:
            for i in range(self.get_rotation.shape[1]):
                l.append('rot_{}'.format(i))
        else:
            for i in range(self._rotation.shape[1]):
                l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path, save_asg=False, use_app=False, app_img_id=-1, app_cam_center=None,
                 app_min_dis_cam_save=False):

        print('gaussian_model save ply start, path: {}, use_app: {}, app_min_dis_cam_save: {}'
              .format(path, use_app, app_min_dis_cam_save))
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = self.norm.cpu().numpy() if self.use_norm else np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        #f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()

        rotation = self._rotation.detach().cpu().numpy()

        data_list = [xyz, normals, f_dc, opacities, scale, rotation]
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        if self.use_label:
            print('gaussian_model save ply, add labels', torch.unique(self.points_label, return_counts=True))
            dtype_full.append(('labels', 'i4'))
            data_list.append(self.points_label.detach().cpu().numpy())

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(data_list, axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        PlyData([el]).write(path)
        print('Save Gaussian Ply: {}, pts_num: {}'.format(path, xyz.shape[0]))

    def load_ply(self, path, cam_img_map=None):
        print('gaussian_model load_ply {}'.format(path))
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        if self.max_sh_degree == 0:
            extra_f_names = []
        else:
            extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
            extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

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

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))

        if self.use_rot_angles:
            self.norm_quat = torch.tensor(rots).float().cuda()
            rot_angles = torch.zeros(len(self.norm_quat), 1).float().cuda()
            self._rot_angles = nn.Parameter(rot_angles.requires_grad_(True))
        else:
            self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        if self.use_label:
            self.points_label = torch.tensor(
                np.asarray(plydata.elements[0]["labels"])[..., np.newaxis]).cuda() if self.use_label else torch.empty(0)
        if self.use_min_cam_dis_weight:
            self.points_min_cam_dis = \
                torch.tensor(np.asarray(plydata.elements[0]["min_cam_dis"])[..., np.newaxis]).cuda() \
                    if self.use_min_cam_dis_weight else torch.empty(0)
            self.points_min_cam_dis_unique = list(torch.unique(self.points_min_cam_dis).cpu().numpy())
            self.points_min_cam_dis_unique.sort()

        if self.use_opacity_distance_decay:
            self._opacity_distance_decay = \
                torch.tensor(np.asarray(plydata.elements[0]["opacity_distance_decay"])[..., np.newaxis]).cuda() \
                    if self.use_opacity_distance_decay else torch.empty(0)

        if self.use_norm:
            self.norm = np.stack((np.asarray(plydata.elements[0]["nx"]), np.asarray(plydata.elements[0]["ny"]),
                                  np.asarray(plydata.elements[0]["nz"])), axis=1)

        if self.use_asg:
            asg_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_asg_")]
            if len(asg_names) > 0:
                f_asgs = np.zeros((xyz.shape[0], len(asg_names)))
                for idx, attr_name in enumerate(asg_names):
                    f_asgs[:, idx] = np.asarray(plydata.elements[0][attr_name])
                self._features_asg = nn.Parameter(
                    torch.tensor(f_asgs, dtype=torch.float, device="cuda").requires_grad_(True))
            else:
                print('WARNING gaussian_model load_ply use asg but no asg in ply!')

        if self.app_opt_cfg is not None:
            app_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_app_")]
            if len(app_names) > 0:
                f_apps = np.zeros((xyz.shape[0], len(app_names)))
                for idx, attr_name in enumerate(app_names):
                    f_apps[:, idx] = np.asarray(plydata.elements[0][attr_name])
                self._features_app = nn.Parameter(
                    torch.tensor(f_apps, dtype=torch.float).requires_grad_(True))
            else:
                print('WARNING gaussian_model load_ply use app but no app in ply! init!')
                app_features_init = torch.rand((xyz.shape[0], self.app_opt_cfg['app_opt_gs_feature_dim'])).float().cuda()
                self._features_app = nn.Parameter(app_features_init.requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # vignetting model
        if self.use_vignet:
            assert cam_img_map is not None and len(cam_img_map) > 0, "Error! use_vignet but cam_img_map is None!"
            self.vignet_model = nn.ParameterDict()
            print('use_vignet init by cam_img_map, cams:', cam_img_map.keys())
            for camera_id in cam_img_map.keys():
                self.vignet_model[str(camera_id)] = nn.Parameter(torch.zeros(3).requires_grad_(True))
            self.vignet_model.cuda()

    def merge(self, pc):
        self._xyz = torch.concat([self._xyz, pc._xyz], dim=0)
        self._features_dc = torch.concat([self._features_dc, pc._features_dc], dim=0)
        self._scaling = torch.concat([self._scaling, pc._scaling], dim=0)
        self._opacity = torch.concat([self._opacity, pc._opacity], dim=0)
        self._rotation = torch.concat([self._rotation, pc._rotation], dim=0)
        if self.use_norm:
            self.norm = torch.concat([self.norm, pc.norm], dim=0)

            
    @staticmethod
    def rx(theta):
        return torch.tensor([[1, 0, 0],
                          [0, torch.cos(theta), -torch.sin(theta)],
                          [0, torch.sin(theta), torch.cos(theta)]])

    @staticmethod
    def ry(theta):
        return torch.tensor([[torch.cos(theta), 0, torch.sin(theta)],
                          [0, 1, 0],
                          [-torch.sin(theta), 0, torch.cos(theta)]])

    @staticmethod
    def rz(theta):
        return torch.tensor([[torch.cos(theta), -torch.sin(theta), 0],
                          [torch.sin(theta), torch.cos(theta), 0],
                          [0, 0, 1]])

    def rescale(self, scale: float):
        if scale != 1.:
            self._xyz =  self._xyz * scale
            self._scaling = self._scaling + torch.log(scale)

            print("rescaled with factor {}".format(scale))

    def rotate_xyz(self, rotmat, mean_xyz):
        new_xyz = self.get_xyz
        #mean_xyz = torch.mean(new_xyz,0)
        new_xyz = new_xyz - mean_xyz
        new_xyz = new_xyz @ rotmat.T
        self._xyz = new_xyz + mean_xyz

    def rotate_rot(self, rotmat):
        new_rotation = build_rotation(self._rotation)
        new_rotation = rotmat @ new_rotation
        new_quat = np.array(Rotation.from_matrix(new_rotation.detach().cpu().numpy()).as_quat())
        new_quat[:, [0,1,2,3]] = new_quat[:, [3,0,1,2]] # xyzw -> wxyz
        self._rotation = torch.from_numpy(new_quat).to(self._rotation.device).float()
    def get_rotation_matrix(self):
        # same as from pytorch3d.transforms import quaternion_to_matrix
        return quaternion_to_matrix(self.get_rotation)
    def get_smallest_axis(self, return_idx=False):
        rotation_matrices = self.get_rotation_matrix()
        smallest_axis_idx = self.get_scaling.min(dim=-1)[1][..., None, None].expand(-1, 3, -1)
        smallest_axis = rotation_matrices.gather(2, smallest_axis_idx)
        if return_idx:
            return smallest_axis.squeeze(dim=2), smallest_axis_idx[..., 0, 0]
        return smallest_axis.squeeze(dim=2)
    def get_normal_global(self, view_cam):
        normal_global = self.get_smallest_axis()
        gaussian_to_cam_global = view_cam.camera_center.cuda() - self._xyz
        neg_mask = (normal_global * gaussian_to_cam_global).sum(-1) < 0.0
        normal_global[neg_mask] = -normal_global[neg_mask]
        return normal_global
    def rotate_by_euler_angles(self, x: float, y: float, z: float, origin):
        """
        rotate in z-y-x order, radians as unit
        """

        if x == 0. and y == 0. and z == 0.:
            return

        rotation_matrix = self.rx(x) @ self.ry(y) @ self.rz(z)
        rotation_matrix = rotation_matrix.to(self._xyz.device).to(self._xyz.dtype)

        self.rotate_xyz(rotation_matrix, origin)
        self.rotate_rot(rotation_matrix)


    def rotate_by_matrix(self, rotation_matrix, keep_sh_degree: bool = True):
        # rotate _xyz
        self._xyz = torch.matmul(self._xyz, rotation_matrix.T).T

        # rotate gaussian
        # rotate via quaternions
        
        new_rotation = build_rotation(self._rotation)
        new_rotation = rotation_matrix @ new_rotation
        self._rotation  = self.rotmat2qvec(new_rotation)

        # rotate via rotation matrix
        # gaussian_rotation = build_rotation(torch.from_numpy(self.rotations)).cpu()
        # gaussian_rotation = torch.from_numpy(rotation_matrix) @ gaussian_rotation
        # xyzw_quaternions = R.from_matrix(gaussian_rotation.numpy()).as_quat(canonical=False)
        # wxyz_quaternions = xyzw_quaternions
        # wxyz_quaternions[:, [0, 1, 2, 3]] = wxyz_quaternions[:, [3, 0, 1, 2]]
        # rotations_from_matrix = wxyz_quaternions
        # self.rotations = rotations_from_matrix

        # TODO: rotate shs
        if keep_sh_degree is False:
            print("set sh_degree=0 when rotation transform enabled")
            self.sh_degrees = 0

    def translation(self, translation_vector):

        self._xyz += translation_vector.to(self._xyz.device).to(self._xyz.dtype)
        print("translation transform applied")
        
    def transform_point_cloud(self, rotation_angles, translation_vector, scale_factors, origin):
        """
        Apply transformations to xyz.

        Args:
            points (torch.Tensor): Point cloud data of shape (N, 3) where N is the number of points.
            rotation_angles (list or tuple): Rotation angles around x, y, and z axes in radians.
            translation_vector (list or tuple): Translation vector (dx, dy, dz).
            scale_factors (list or tuple): Scale factors for each axis (sx, sy, sz).

        Returns:
            torch.Tensor: Transformed point cloud.
        """

        # Convert parameters to tensors if they aren't already
        self.rescale(scale_factors)
        self.rotate_by_euler_angles(rotation_angles[0], rotation_angles[1], rotation_angles[2], origin)
        self.translation(translation_vector)
        

    def get_points_depth_in_depth_map(self, fov_camera, depth, points_in_camera_space, scale=1, return_pts_projections = False):
        # TODO scale_x scale_y
        # st = max(int(scale / 2) - 1, 0)
        # depth_view = depth[None, :, st::scale, st::scale]
        depth_view = depth[None, :, :, :]
        W, H = int(fov_camera.image_width / scale), int(fov_camera.image_height / scale)
        depth_view = depth_view[:H, :W]
        pts_projections = torch.stack(
            [points_in_camera_space[:, 0] * fov_camera.fx / points_in_camera_space[:, 2] + fov_camera.cx_pixel,
             points_in_camera_space[:, 1] * fov_camera.fy / points_in_camera_space[:, 2] + fov_camera.cy_pixel],
            -1).float() / scale
        mask = (pts_projections[:, 0] > 0) & (pts_projections[:, 0] < W) & \
               (pts_projections[:, 1] > 0) & (pts_projections[:, 1] < H) & (points_in_camera_space[:, 2] > 0.1)

        if return_pts_projections:
            pts_projections_return = pts_projections.clone().detach()

        pts_projections[..., 0] /= ((W - 1) / 2)
        pts_projections[..., 1] /= ((H - 1) / 2)
        pts_projections -= 1
        pts_projections = pts_projections.view(1, -1, 1, 2)
        map_z = torch.nn.functional.grid_sample(input=depth_view,
                                                grid=pts_projections,
                                                mode='bilinear',
                                                padding_mode='border',
                                                align_corners=True
                                                )[0, :, :, 0]
        if return_pts_projections:
            return map_z, mask, pts_projections_return
        else:
            return map_z, mask

    def get_points_from_depth(self, fov_camera, depth, scale=1, mask = None):
        # TODO scale_x scale_y
        # st = int(max(int(scale / 2) - 1, 0))
        # depth_view = depth.squeeze()[st::scale, st::scale]
        depth_view = depth.squeeze()
        rays_d = fov_camera.get_rays(scale=scale, mask = mask)
        if mask is not None:
            depth_view = depth_view[mask]
        else:
            depth_view = depth_view[:rays_d.shape[0], :rays_d.shape[1]]
        pts = (rays_d * depth_view[..., None]).reshape(-1, 3)
        R = torch.tensor(fov_camera.R).float().cuda()
        T = torch.tensor(fov_camera.T).float().cuda()
        pts = (pts - T) @ R.transpose(-1, -2)
        return pts

    def get_norm_back_mask(self, camera):
        zray = (e3nn.o3.quaternion_to_matrix(self.get_rotation) @ torch.tensor([0,0,1]).float().cuda()[:,None])[...,0]
        ray = self.get_xyz - camera.camera_center[None,:].cuda()
        ray = ray / torch.linalg.norm(ray, dim=-1, keepdims=True)
        back = (zray * ray).sum(-1) > 0
        return ~back





    


