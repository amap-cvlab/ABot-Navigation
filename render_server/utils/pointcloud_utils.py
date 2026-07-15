import os
from plyfile import PlyData, PlyElement
import numpy as np
import cv2
import torch
import PIL.Image
import PIL.ImageDraw
from argparse import ArgumentParser
from typing import Tuple
def open3d_get_norm(points_np, radius=5.0, max_nn=50):
    # points_np = np.array([ply_vertex['x'], ply_vertex['y'], ply_vertex['z']]).T

    print('open3d_get_norm: open3d compute normals, radius: {}, max_nn: {}'.format(radius, max_nn))
    import open3d as o3d
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(np.array(points_np, np.float32))
    o3d.geometry.PointCloud.estimate_normals(
        pointcloud,
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)  # 0.66 320
    )

    # print('oopen3d_get_norm: open3d compute normals location')
    # o3d.geometry.PointCloud.orient_normals_towards_camera_location(pointcloud, camera_location=np.array([0, 0, 200.0]))

    print('open3d_get_norm: open3d compute normals Done!')
    c_norms = np.asarray(pointcloud.normals, np.float32)

    return c_norms[:, 0], c_norms[:, 1], c_norms[:, 2]  # nx ny nz
def generate_spherical_point_cloud(num_points=200000, radius=None, output=None, offset=None,
                                   scene_box_label=2, add_norm=False):
    if radius is None:
        radius = [2000]

    spherical_point_cloud_list = list()
    spherical_point_cloud_nxyz_list = list()
    for point_r in radius:
        phi = np.random.uniform(0, 2*np.pi, num_points)
        cos_theta = np.random.uniform(-1, 1, num_points)
        theta = np.arccos(cos_theta)

        x = point_r * np.sin(theta) * np.cos(phi)
        y = point_r * np.sin(theta) * np.sin(phi)
        z = point_r * np.cos(theta)

        xyz = np.column_stack((x, y, z))

        if add_norm:
            nx, ny, nz = open3d_get_norm(xyz)
        else:
            nx, ny, nz = np.zeros_like(x), np.zeros_like(y), np.zeros_like(z)
        nxyz = np.column_stack((nx, ny, nz))

        spherical_point_cloud_list.append(xyz)
        spherical_point_cloud_nxyz_list.append(nxyz)

    spherical_point_cloud = np.concatenate(spherical_point_cloud_list, axis=0)
    spherical_point_cloud_norm = np.concatenate(spherical_point_cloud_nxyz_list, axis=0)

    if offset is not None:
        spherical_point_cloud += offset

    if output is not None:
        dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
                 ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'), ('labels', 'i4')]
        elements = np.empty(spherical_point_cloud.shape[0], dtype=dtype)
        elements['x'] = np.array(spherical_point_cloud[:, 0])
        elements['y'] = np.array(spherical_point_cloud[:, 1])
        elements['z'] = np.array(spherical_point_cloud[:, 2])
        elements['nx'] = np.array(spherical_point_cloud_norm[:, 0])
        elements['ny'] = np.array(spherical_point_cloud_norm[:, 1])
        elements['nz'] = np.array(spherical_point_cloud_norm[:, 2])
        elements['red'] = np.zeros_like(spherical_point_cloud[:, 0])
        elements['green'] = np.zeros_like(spherical_point_cloud[:, 0])
        elements['blue'] = np.zeros_like(spherical_point_cloud[:, 0])
        elements['labels'] = np.ones_like(spherical_point_cloud[:, 0]) * scene_box_label
        vertex_element = PlyElement.describe(elements, 'vertex')
        ply_data = PlyData([vertex_element], text=False)
        ply_data.write(output)
        print('generate_spherical_point_cloud save ply', output)
    return spherical_point_cloud