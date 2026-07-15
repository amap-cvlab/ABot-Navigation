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
import sys

import torch
import math
from diff_plane_rasterization import GaussianRasterizationSettings as PlaneGaussianRasterizationSettings
from diff_plane_rasterization import GaussianRasterizer as PlaneGaussianRasterizer
from utils.sh_utils import eval_sh
from utils.graphics_utils import normal_from_depth_image
from utils.general_utils import build_rotation, quaternion_multiply

import numpy as np
import cv2

def render_normal(viewpoint_cam, depth, offset=None, normal=None, scale=1):
    # depth: (H, W), bg_color: (3), alpha: (H, W)
    # normal_ref: (3, H, W)
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf(scale=scale)
    st = max(int(scale / 2) - 1, 0)
    if offset is not None:
        offset = offset[st::scale, st::scale]
    normal_ref = normal_from_depth_image(depth[st::scale, st::scale],
                                         intrinsic_matrix.to(depth.device),
                                         extrinsic_matrix.to(depth.device), offset)
    normal_ref = normal_ref.permute(2, 0, 1)
    return normal_ref


def render(viewpoint_camera, pc, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
           override_color=None, use_c3d_mask=False, use_opacity_distance_decay=False, use_min_cam_dis_filter=False,
           valid_mask=None, img_res_scale=0,
           render_mode=0, min_alpha=1.0 / 255.0, depth_offset=0.0, cov_2d_bias=0.3, global_ges_beta=1,
           render_depth=False, render_depth_mode=0, return_render_pts=False, opacity_distance_decay_grad=True,
           ntc_color_ratio=0.0, asg_color=None, app_color=None, app_color_beta=None, app_opacity=None,
           app_color_zero_img_embed=None,
           ntc_asg_app_debug=False, app_color_debug_vis=False, lod_debug=False,
           return_plane=True, return_depth_normal=True, return_global_rendered_normal=False, return_rendered_alpha = False,
           world_rotation_transfrom=None, world_translation_transfrom=None, render_z_depth=False,
           render_T_offset=None, render_roll_pitch_yaw_offset=None, use_blur_opt=False, blur_opt_debug=False):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    use_ntc = ntc_color_ratio > 0
    use_app = app_color is not None
    use_asg = asg_color is not None

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0

    # for abs
    screenspace_points_abs = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
        screenspace_points_abs.retain_grad()
    except:
        pass

    if render_T_offset is not None or render_roll_pitch_yaw_offset is not None:
        world_view_transform, full_proj_transform, camera_center = (
            viewpoint_camera.get_offset_matrix(T_offset=render_T_offset,
                                               roll_pitch_yaw_offset=render_roll_pitch_yaw_offset))
    else:
        world_view_transform, full_proj_transform, camera_center =\
            (viewpoint_camera.world_view_transform.cuda(), viewpoint_camera.full_proj_transform.cuda(),
             viewpoint_camera.camera_center.cuda())

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = PlaneGaussianRasterizationSettings(
        image_height=int(
            viewpoint_camera.image_height * img_res_scale if img_res_scale > 0 else viewpoint_camera.image_height),
        image_width=int(
            viewpoint_camera.image_width * img_res_scale if img_res_scale > 0 else viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=world_view_transform.cuda(),
        projmatrix=full_proj_transform.cuda(),
        sh_degree=pc.active_sh_degree,
        campos=camera_center.cuda(),
        prefiltered=False,
        render_geo=return_plane,
        pixel_cx=int(viewpoint_camera.cx_pixel * img_res_scale) if img_res_scale > 0 else viewpoint_camera.cx_pixel,
        pixel_cy=int(viewpoint_camera.cy_pixel * img_res_scale) if img_res_scale > 0 else viewpoint_camera.cy_pixel,
        debug=pipe.debug
    )

    rasterizer = PlaneGaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    means2D_abs = screenspace_points_abs  # for abs

    if world_rotation_transfrom is not None and world_translation_transfrom is not None:
        means3D = (build_rotation(world_rotation_transfrom) @ means3D.T).squeeze().T + world_translation_transfrom

    if use_c3d_mask:
        c3d_mask = ((torch.sigmoid(pc._c3d_mask) > 0.01).float() - torch.sigmoid(
            pc._c3d_mask)).detach() + torch.sigmoid(pc._c3d_mask)
        # print('render c3d_mask points_label:', torch.unique(pc.points_label[c3d_mask < 1.0], return_counts=True))
        opacity = pc.get_opacity * c3d_mask
    else:
        opacity = pc.get_opacity

    depth = None
    if use_opacity_distance_decay:
        xyz_n4 = torch.cat([means3D, torch.ones([means3D.shape[0], 1], device='cuda')], axis=-1).unsqueeze(-1)
        depth = torch.matmul(viewpoint_camera.world_view_transform.transpose(0, 1).cuda(), xyz_n4)[:, 2]
        del xyz_n4

        if depth_offset > 0:
            depth[depth > 0] = depth[depth > 0] + depth_offset

        # _opacity_distance_decay = 1 depth: 0.7788
        # _opacity_distance_decay = 2 depth: 0.9394
        # _opacity_distance_decay = 3 depth: 0.9726
        # depth 贴近 3 * gaussians.points_min_cam_dis
        if opacity_distance_decay_grad:
            opacity_distance_decay = torch.exp(-(depth ** 2) / (2 * torch.relu(pc._opacity_distance_decay) ** 2 + 1e-6))
        else:
            opacity_distance_decay = torch.exp(
                -(depth ** 2) / (2 * torch.relu(pc._opacity_distance_decay.detach()) ** 2 + 1e-6))
        opacity = opacity * opacity_distance_decay
    elif use_min_cam_dis_filter:
        xyz_n4 = torch.cat([means3D, torch.ones([means3D.shape[0], 1], device='cuda')], axis=-1).unsqueeze(-1)
        depth = torch.matmul(viewpoint_camera.world_view_transform.transpose(0, 1).cuda(), xyz_n4)[:, 2].squeeze()
        del xyz_n4

        points_min_cam_dis = pc.points_min_cam_dis
        clip_levels = pc.points_min_cam_dis_unique

        opacity_distance_filters = torch.zeros_like(opacity).squeeze(-1)
        last_clip_level = 0
        for clip_level in clip_levels:
            # for near level to far level
            level_pts_mask = torch.logical_and(last_clip_level < points_min_cam_dis, points_min_cam_dis <= clip_level).squeeze(-1)
            level_pick_mask = 1.0 * torch.logical_and(last_clip_level < depth[level_pts_mask], depth[level_pts_mask] <= clip_level)
            opacity_distance_filters[level_pts_mask] = level_pick_mask
            # print(last_clip_level, clip_level, torch.sum(opacity_distance_filters), torch.sum(level_pick_mask), '/', torch.sum(level_pts_mask))
            last_clip_level = clip_level
        level_pts_mask = (last_clip_level <= points_min_cam_dis).squeeze(-1)
        level_pick_mask = 1.0 * torch.logical_or(opacity_distance_filters[level_pts_mask] > 0, last_clip_level < depth[level_pts_mask])
        opacity_distance_filters[level_pts_mask] = level_pick_mask
        # print(last_clip_level, 'inf', torch.sum(opacity_distance_filters), torch.sum(level_pick_mask), '/', torch.sum(level_pts_mask))

        if valid_mask is None:
            valid_mask = torch.ones_like(opacity).squeeze(-1)
        valid_mask = valid_mask * opacity_distance_filters

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        if use_c3d_mask:
            scales = pc.get_scaling * c3d_mask
        else:
            scales = pc.get_scaling
        rotations = pc.get_rotation
        if world_rotation_transfrom is not None:
            rotations = quaternion_multiply(world_rotation_transfrom, rotations)

    if valid_mask is not None:
        opacity = opacity * valid_mask.unsqueeze(-1)
        scales = scales * valid_mask.unsqueeze(-1)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    colors_precomp_zero_img_embed = None # for app-zero-embed vis
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.cuda().unsqueeze(0))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            if use_ntc:
                pc.query_ntc(viewpoint_camera.camera_center.cuda())
                colors_precomp = colors_precomp + pc._d_color * ntc_color_ratio
            if use_asg and asg_color is not None:
                colors_precomp = colors_precomp + asg_color
            if use_app and app_color is not None:

                # for vis
                if app_color_zero_img_embed is not None:
                    colors_precomp_zero_img_embed = torch.clamp_min(
                        colors_precomp + torch.sigmoid(app_color_zero_img_embed) * 2 - 1, 0.0)

                if app_color_beta is not None:
                    # app_color_beta range 0 ~ 2
                    colors_precomp = colors_precomp * app_color_beta
                colors_precomp = torch.clamp_min(colors_precomp + torch.sigmoid(app_color) * 2 - 1, 0.0)

                if app_opacity is not None:
                    opacity = torch.clamp(opacity + app_opacity, min=0.0, max=1.0)
        else:
            assert not use_ntc, 'use ntc but override_color or not convert_SHs_python!'
            assert not use_asg, 'use asg but override_color or not convert_SHs_python!'
            assert not use_app, 'use app opt but override_color or not convert_SHs_python!'
            shs = pc.get_features
    else:
        colors_precomp = override_color

    if use_blur_opt:
        blur_opt_lambda_s, blur_opt_max_clamp = 0.01, 1.1
        scales_delta, rotations_delta, pos_delta = pc.blur_module(means3D.detach(), scales.detach(), rotations.detach(),
                                                            viewpoint_camera.camera_center.cuda().repeat(means3D.shape[0], 1))
        scales_delta = torch.clamp(blur_opt_lambda_s * scales_delta + (1 - blur_opt_lambda_s), min=1.0, max=blur_opt_max_clamp)
        rotations_delta = torch.clamp(blur_opt_lambda_s * rotations_delta + (1 - blur_opt_lambda_s), min=1.0, max=blur_opt_max_clamp)

        scales = scales * scales_delta
        rotations = rotations * rotations_delta

    if not return_plane:
        rendered_image, radii, out_observe, _, _ , _, _= rasterizer(
            means3D=means3D,
            means2D=means2D,
            means2D_abs=means2D_abs,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)

        return_dict = {"render": rendered_image,
                       "render_depth": None,
                       "viewspace_points": screenspace_points,
                       "viewspace_points_abs": screenspace_points_abs,
                       "visibility_filter": radii > 0,
                       "radii": radii,
                       "out_observe": out_observe,
                       "valid_mask": valid_mask}

    else:
        global_normal = pc.get_normal_global(viewpoint_camera)
        # for debug global_normal = torch.clip(global_normal * torch.tensor([0.0, 0.0, 100.0]).cuda(), 0, 1)
        local_normal = global_normal @ viewpoint_camera.world_view_transform[:3, :3].cuda()
        pts_in_cam = means3D @ viewpoint_camera.world_view_transform[:3, :3].cuda() + viewpoint_camera.world_view_transform[3, :3].cuda()
        depth_z = pts_in_cam[:, 2]
        local_distance = (local_normal * pts_in_cam).sum(-1).abs()
        input_all_map = torch.zeros((means3D.shape[0], 5)).cuda().float()
        input_all_map[:, :3] = local_normal
        input_all_map[:, 3] = 1.0
        input_all_map[:, 4] = local_distance

        rendered_image, radii, out_observe, out_all_map, plane_depth, out_peak_index, out_peak_weight = rasterizer(
            means3D=means3D,
            means2D=means2D,
            means2D_abs=means2D_abs,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            all_map=input_all_map,
            cov3D_precomp=cov3D_precomp)

        rendered_normal = out_all_map[0:3]
        rendered_alpha = out_all_map[3:4, ]
        rendered_distance = out_all_map[4:5, ]
        # out_plane_depth[pix_id] = All_map[4] / -(All_map[0] * ray.x + All_map[1] * ray.y + All_map[2] + 1.0e-8);
        # rendered_distance / -(rendered_normal[0] * x + rendered_normal[1] * y + rendered_normal[2])
        # rendered_normal.permute(1, 2, 0)
        # ((rendered_normal + 1.0) * 0.5).permute(1, 2, 0)

        return_dict = {"render": rendered_image,
                       "render_depth": plane_depth,
                       "viewspace_points": screenspace_points,
                       "viewspace_points_abs": screenspace_points_abs,
                       "visibility_filter": radii > 0,
                       "radii": radii,
                       "out_observe": out_observe,
                       "rendered_normal": rendered_normal,
                       "plane_depth": plane_depth,
                       "rendered_distance": rendered_distance,
                       "valid_mask": valid_mask,
                       "out_peak_index": out_peak_index,
                       "out_peak_weight": out_peak_weight,
                       "local_distance": local_distance,
                       "local_normal": local_normal,
                       }

        if return_rendered_alpha:
            return_dict.update({"rendered_alpha": rendered_alpha})
            
        if return_depth_normal:
            depth_normal = render_normal(viewpoint_camera, plane_depth.squeeze()) * (rendered_alpha).detach()
            return_dict.update({"depth_normal": depth_normal})

        # rendered_normal to global
        if return_global_rendered_normal:
            rendered_normal_global = (rendered_normal.permute(1, 2, 0) @ viewpoint_camera.world_view_transform[:3,
                                                                         :3].cuda().T).permute(2, 0, 1)
            return_dict.update({"rendered_normal_global": rendered_normal_global})

    if render_z_depth:
        xyz_cam = torch.cat([pc.get_xyz, torch.ones([pc.get_xyz.shape[0], 1], device='cuda')],
                            axis=-1).unsqueeze(-1)
        xyz_cam = torch.matmul(viewpoint_camera.world_view_transform.transpose(0, 1).cuda(), xyz_cam)
        z_depth = xyz_cam.squeeze()[:, 2:3].repeat(1, 3)  # N 3
        global_normal = pc.get_normal_global(viewpoint_camera)
        local_normal = global_normal @ viewpoint_camera.world_view_transform[:3, :3].cuda()
        pts_in_cam = means3D @ viewpoint_camera.world_view_transform[:3,
                               :3].cuda() + viewpoint_camera.world_view_transform[3, :3].cuda()
        local_distance = (local_normal * pts_in_cam).sum(-1).abs()
        input_all_map = torch.zeros((means3D.shape[0], 5)).cuda().float()
        input_all_map[:, :3] = local_normal
        input_all_map[:, 3] = 1.0
        input_all_map[:, 4] = local_distance
        depth_image, _, _, _, _, _, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            means2D_abs=means2D_abs,
            shs=None,
            colors_precomp=z_depth,
            opacities=torch.ones_like(pc.get_xyz[..., :1]),
            scales=scales,
            rotations=rotations,
            all_map=input_all_map,
            cov3D_precomp=cov3D_precomp)
        render_z_depth = depth_image[0]
        return_dict.update({"render_z_depth": render_z_depth})
        del z_depth

    # debug vis render
    with torch.no_grad():
        ntc_rendered_image = None
        if use_ntc and ntc_asg_app_debug:
            pc.query_ntc(viewpoint_camera.camera_center.cuda())
            colors_precomp = pc._d_color.float()
            colors_precomp = torch.clamp_min(colors_precomp + 0.5, 0.0)
            if not return_plane:
                ntc_rendered_image, _, _, _, _, _, _ = rasterizer(
                    means3D=means3D,
                    means2D=means2D,
                    means2D_abs=means2D_abs,
                    shs=shs,
                    colors_precomp=colors_precomp,
                    opacities=opacity,
                    scales=scales,
                    rotations=rotations,
                    cov3D_precomp=cov3D_precomp)
            else:
                global_normal = pc.get_normal_global(viewpoint_camera)
                local_normal = global_normal @ viewpoint_camera.world_view_transform[:3, :3].cuda()
                pts_in_cam = means3D @ viewpoint_camera.world_view_transform[:3,
                                       :3].cuda() + viewpoint_camera.world_view_transform[3, :3].cuda()
                local_distance = (local_normal * pts_in_cam).sum(-1).abs()
                input_all_map = torch.zeros((means3D.shape[0], 5)).cuda().float()
                input_all_map[:, :3] = local_normal
                input_all_map[:, 3] = 1.0
                input_all_map[:, 4] = local_distance
                ntc_rendered_image, _, _, _, _, _, _ = rasterizer(
                    means3D=means3D,
                    means2D=means2D,
                    means2D_abs=means2D_abs,
                    shs=shs,
                    colors_precomp=colors_precomp,
                    opacities=opacity,
                    scales=scales,
                    rotations=rotations,
                    all_map=input_all_map,
                    cov3D_precomp=cov3D_precomp)


        asg_rendered_image = None
        if use_asg and ntc_asg_app_debug:
            # asg_color_norm = (asg_color - torch.min(asg_color)) / (torch.max(asg_color) - torch.min(asg_color))
            colors_precomp = torch.clamp_min(asg_color, 0.0)
            if not return_plane:
                asg_rendered_image, _, _, _, _, _, _ = rasterizer(
                    means3D=means3D,
                    means2D=means2D,
                    means2D_abs=means2D_abs,
                    shs=shs,
                    colors_precomp=colors_precomp,
                    opacities=opacity,
                    scales=scales,
                    rotations=rotations,
                    cov3D_precomp=cov3D_precomp)
            else:
                global_normal = pc.get_normal_global(viewpoint_camera)
                local_normal = global_normal @ viewpoint_camera.world_view_transform[:3, :3].cuda()
                pts_in_cam = means3D @ viewpoint_camera.world_view_transform[:3,
                                       :3].cuda() + viewpoint_camera.world_view_transform[3, :3].cuda()
                local_distance = (local_normal * pts_in_cam).sum(-1).abs()
                input_all_map = torch.zeros((means3D.shape[0], 5)).cuda().float()
                input_all_map[:, :3] = local_normal
                input_all_map[:, 3] = 1.0
                input_all_map[:, 4] = local_distance
                asg_rendered_image, _, _, _, _, _, _ = rasterizer(
                    means3D=means3D,
                    means2D=means2D,
                    means2D_abs=means2D_abs,
                    shs=shs,
                    colors_precomp=colors_precomp,
                    opacities=opacity,
                    scales=scales,
                    rotations=rotations,
                    all_map=input_all_map,
                    cov3D_precomp=cov3D_precomp)

        app_rendered_image = None
        app_rendered_image_zero_img_embed = None
        if use_app and ntc_asg_app_debug and app_color_debug_vis:
            colors_precomp = torch.sigmoid(app_color)
            if not return_plane:
                app_rendered_image, _, _, _, _, _, _ = rasterizer(
                    means3D=means3D,
                    means2D=means2D,
                    means2D_abs=means2D_abs,
                    shs=shs,
                    colors_precomp=colors_precomp,
                    opacities=opacity,
                    scales=scales,
                    rotations=rotations,
                    cov3D_precomp=cov3D_precomp)
            else:
                global_normal = pc.get_normal_global(viewpoint_camera)
                local_normal = global_normal @ viewpoint_camera.world_view_transform[:3, :3].cuda()
                pts_in_cam = means3D @ viewpoint_camera.world_view_transform[:3,
                                       :3].cuda() + viewpoint_camera.world_view_transform[3, :3].cuda()
                local_distance = (local_normal * pts_in_cam).sum(-1).abs()
                input_all_map = torch.zeros((means3D.shape[0], 5)).cuda().float()
                input_all_map[:, :3] = local_normal
                input_all_map[:, 3] = 1.0
                input_all_map[:, 4] = local_distance
                app_rendered_image, _, _, _, _, _, _ = rasterizer(
                    means3D=means3D,
                    means2D=means2D,
                    means2D_abs=means2D_abs,
                    shs=shs,
                    colors_precomp=colors_precomp,
                    opacities=opacity,
                    scales=scales,
                    rotations=rotations,
                    all_map=input_all_map,
                    cov3D_precomp=cov3D_precomp)

            if colors_precomp_zero_img_embed is not None:
                if not return_plane:
                    app_rendered_image_zero_img_embed, _, _, _, _, _, _ = rasterizer(
                        means3D=means3D,
                        means2D=means2D,
                        means2D_abs=means2D_abs,
                        shs=shs,
                        colors_precomp=colors_precomp_zero_img_embed,
                        opacities=opacity,
                        scales=scales,
                        rotations=rotations,
                        cov3D_precomp=cov3D_precomp)
                else:
                    global_normal = pc.get_normal_global(viewpoint_camera)
                    local_normal = global_normal @ viewpoint_camera.world_view_transform[:3, :3].cuda()
                    pts_in_cam = means3D @ viewpoint_camera.world_view_transform[:3,
                                           :3].cuda() + viewpoint_camera.world_view_transform[3, :3].cuda()
                    local_distance = (local_normal * pts_in_cam).sum(-1).abs()
                    input_all_map = torch.zeros((means3D.shape[0], 5)).cuda().float()
                    input_all_map[:, :3] = local_normal
                    input_all_map[:, 3] = 1.0
                    input_all_map[:, 4] = local_distance
                    app_rendered_image_zero_img_embed, _, _, _, _, _, _ = rasterizer(
                        means3D=means3D,
                        means2D=means2D,
                        means2D_abs=means2D_abs,
                        shs=shs,
                        colors_precomp=colors_precomp_zero_img_embed,
                        opacities=opacity,
                        scales=scales,
                        rotations=rotations,
                        all_map=input_all_map,
                        cov3D_precomp=cov3D_precomp)

        sh_rendered_image = None
        if (use_ntc and ntc_asg_app_debug) or (use_asg and ntc_asg_app_debug) or (use_app and ntc_asg_app_debug):
            if not return_plane:
                sh_rendered_image, _, _, _, _, _, _ = rasterizer(
                    means3D=means3D,
                    means2D=means2D,
                    means2D_abs=means2D_abs,
                    shs=pc.get_features,
                    colors_precomp=None,
                    opacities=pc.get_opacity,
                    scales=scales,
                    rotations=rotations,
                    cov3D_precomp=cov3D_precomp)
            else:
                global_normal = pc.get_normal_global(viewpoint_camera)
                local_normal = global_normal @ viewpoint_camera.world_view_transform[:3, :3].cuda()
                pts_in_cam = means3D @ viewpoint_camera.world_view_transform[:3,
                                       :3].cuda() + viewpoint_camera.world_view_transform[3, :3].cuda()
                local_distance = (local_normal * pts_in_cam).sum(-1).abs()
                input_all_map = torch.zeros((means3D.shape[0], 5)).cuda().float()
                input_all_map[:, :3] = local_normal
                input_all_map[:, 3] = 1.0
                input_all_map[:, 4] = local_distance
                sh_rendered_image, _, _, _, _, _, _ = rasterizer(
                    means3D=means3D,
                    means2D=means2D,
                    means2D_abs=means2D_abs,
                    shs=pc.get_features,
                    colors_precomp=None,
                    opacities=pc.get_opacity,
                    scales=scales,
                    rotations=rotations,
                    all_map=input_all_map,
                    cov3D_precomp=cov3D_precomp)

        sh_rendered_image_wo_blur_opt = None
        if use_blur_opt and blur_opt_debug:
            if not return_plane:
                sh_rendered_image_wo_blur_opt, _, _, _, _ = rasterizer(
                    means3D=means3D,
                    means2D=means2D,
                    means2D_abs=means2D_abs,
                    shs=pc.get_features,
                    colors_precomp=None,
                    opacities=opacity,
                    scales=scales/ scales_delta,
                    rotations=rotations / rotations_delta,
                    cov3D_precomp=cov3D_precomp)
            else:
                global_normal = pc.get_normal_global(viewpoint_camera)
                local_normal = global_normal @ viewpoint_camera.world_view_transform[:3, :3].cuda()
                pts_in_cam = means3D @ viewpoint_camera.world_view_transform[:3,
                                       :3].cuda() + viewpoint_camera.world_view_transform[3, :3].cuda()
                local_distance = (local_normal * pts_in_cam).sum(-1).abs()
                input_all_map = torch.zeros((means3D.shape[0], 5)).cuda().float()
                input_all_map[:, :3] = local_normal
                input_all_map[:, 3] = 1.0
                input_all_map[:, 4] = local_distance
                sh_rendered_image_wo_blur_opt, _, _, _, _ = rasterizer(
                    means3D=means3D,
                    means2D=means2D,
                    means2D_abs=means2D_abs,
                    shs=pc.get_features,
                    colors_precomp=None,
                    opacities=opacity,
                    scales=scales/ scales_delta,
                    rotations=rotations / rotations_delta,
                    all_map=input_all_map,
                    cov3D_precomp=cov3D_precomp)

        lod_renderd_image_i = None
        if lod_debug and use_min_cam_dis_filter:
            lod_color = points_min_cam_dis.repeat(1, 3)

            if not return_plane:
                lod_renderd_image, _, _, _, _, _, _ = rasterizer(
                    means3D=means3D,
                    means2D=means2D,
                    means2D_abs=means2D_abs,
                    shs=None,
                    colors_precomp=lod_color,
                    opacities=opacity,
                    scales=scales,
                    rotations=rotations,
                    cov3D_precomp=cov3D_precomp)
            else:
                global_normal = pc.get_normal_global(viewpoint_camera)
                local_normal = global_normal @ viewpoint_camera.world_view_transform[:3, :3].cuda()
                pts_in_cam = means3D @ viewpoint_camera.world_view_transform[:3,
                                       :3].cuda() + viewpoint_camera.world_view_transform[3, :3].cuda()
                local_distance = (local_normal * pts_in_cam).sum(-1).abs()
                input_all_map = torch.zeros((means3D.shape[0], 5)).cuda().float()
                input_all_map[:, :3] = local_normal
                input_all_map[:, 3] = 1.0
                input_all_map[:, 4] = local_distance
                lod_renderd_image, _, _, _, _, _, _ = rasterizer(
                    means3D=means3D,
                    means2D=means2D,
                    means2D_abs=means2D_abs,
                    shs=None,
                    colors_precomp=lod_color,
                    opacities=opacity,
                    scales=scales,
                    rotations=rotations,
                    all_map=input_all_map,
                    cov3D_precomp=cov3D_precomp)

            lod_renderd_image_i = lod_renderd_image.permute(1, 2, 0).squeeze().detach().cpu().numpy()
            lod_renderd_image_i = (lod_renderd_image_i - lod_renderd_image_i.min()) / (lod_renderd_image_i.max() - lod_renderd_image_i.min() + 1e-10)
            lod_renderd_image_i = (lod_renderd_image_i * 255).clip(0, 255).astype(np.uint8)
            lod_renderd_image_i = cv2.applyColorMap(lod_renderd_image_i, cv2.COLORMAP_JET)
            lod_renderd_image_i = torch.tensor(lod_renderd_image_i).permute(2, 0, 1) / 255.0

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.

    return_dict["app_render"] = app_rendered_image
    return_dict["app_render_zero_img_embed"] = app_rendered_image_zero_img_embed
    return_dict["ntc_render"] = ntc_rendered_image
    return_dict["asg_render"] = asg_rendered_image
    return_dict["sh_render"] = sh_rendered_image
    return_dict["lod_render"] = lod_renderd_image_i
    return_dict["sh_rendered_image_wo_blur_opt"] = sh_rendered_image_wo_blur_opt

    return return_dict
