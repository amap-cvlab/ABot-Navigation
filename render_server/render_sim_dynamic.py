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
# 动态场景加载版本 - 支持多场景、多GPU、LRU淘汰
#
import io
import os
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
import threading
import time
from collections import OrderedDict
from typing import Dict, Optional, List, Tuple

from flask import Flask, jsonify, request, send_file, make_response
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor
from global_config import cuda_init_lock
from utils.xyz_euler_trans_gs_colmap_data import xyz_euler_trans_gs_colmap

# 从原文件导入必要的类和函数
from render_sim import (
    CameraModel, GSRender, read_images_str, read_cameras_str,
    GaussianData, load_gs_ply, save_gs_ply, save_gs_mini_ply,
    add_logo_to_img, searchForMaxIteration, make_get_request
)


class SceneManager:
    """
    动态场景管理器
    - 支持多场景动态加载
    - 支持多GPU分配
    - 支持LRU淘汰策略
    """
    
    def __init__(
        self,
        scenes_root: str,
        gpus: List[int],
        max_scenes_per_gpu: int,
        model_args,
        pipeline,
        op,
        args
    ):
        """
        初始化场景管理器
        
        Args:
            scenes_root: 所有场景的根目录，每个子目录为一个场景
            gpus: 可用GPU列表，如 [0, 1, 2]
            max_scenes_per_gpu: 每个GPU最大加载场景数
            model_args: 模型参数
            pipeline: 管线参数
            op: 优化参数
            args: 其他参数
        """
        self.scenes_root = scenes_root
        self.gpus = gpus
        self.max_scenes_per_gpu = max_scenes_per_gpu
        self.max_total_scenes = len(gpus) * max_scenes_per_gpu
        
        self.model_args = model_args
        self.pipeline = pipeline
        self.op = op
        self.args = args
        
        # 场景缓存: scene_id -> (GSRender实例, gpu_id, 最后访问时间)
        self._scene_cache: OrderedDict[str, Tuple[GSRender, int, float]] = OrderedDict()
        
        # GPU场景计数: gpu_id -> 当前加载的场景数
        self._gpu_scene_count: Dict[int, int] = {gpu: 0 for gpu in gpus}
        
        # 线程锁，保证线程安全
        self._lock = threading.RLock()
        
        # 扫描可用场景
        self._available_scenes = self._scan_available_scenes()
        print(f"[SceneManager] 初始化完成")
        print(f"  - 场景根目录: {scenes_root}")
        print(f"  - 可用GPU: {gpus}")
        print(f"  - 每GPU最大场景数: {max_scenes_per_gpu}")
        print(f"  - 总最大场景数: {self.max_total_scenes}")
        print(f"  - 发现可用场景: {list(self._available_scenes.keys())}")
    
    def _scan_available_scenes(self) -> Dict[str, str]:
        """扫描可用场景目录"""
        available = {}
        if not os.path.exists(self.scenes_root):
            print(f"[SceneManager] 警告: 场景根目录不存在: {self.scenes_root}")
            return available
            
        for scene_id in os.listdir(self.scenes_root):
            scene_path = os.path.join(self.scenes_root, scene_id)
            if os.path.isdir(scene_path):
                # 检查是否有gs文件夹，gs文件夹下至少有一个子文件夹，且子文件夹中有ply文件
                gs_folder = os.path.join(scene_path, 'gs')
                if os.path.isdir(gs_folder):
                    has_valid_subfolder = False
                    for subfolder in os.listdir(gs_folder):
                        subfolder_path = os.path.join(gs_folder, subfolder)
                        if os.path.isdir(subfolder_path):
                            # 检查子文件夹中是否有ply文件
                            if any(f.endswith('.ply') for f in os.listdir(subfolder_path)
                                   if os.path.isfile(os.path.join(subfolder_path, f))):
                                has_valid_subfolder = True
                                break
                    if has_valid_subfolder:
                        available[scene_id] = scene_path
        return available
    
    def _select_gpu_for_new_scene(self) -> int:
        """选择一个GPU来加载新场景（选择负载最低的GPU）"""
        min_count = float('inf')
        selected_gpu = self.gpus[0]
        for gpu in self.gpus:
            if self._gpu_scene_count[gpu] < min_count:
                min_count = self._gpu_scene_count[gpu]
                selected_gpu = gpu
        return selected_gpu
    
    def _evict_oldest_scene(self) -> Optional[str]:
        """淘汰最久未使用的场景"""
        if not self._scene_cache:
            return None
        
        # OrderedDict保持插入顺序，但我们需要按最后访问时间排序
        oldest_scene_id = None
        oldest_time = float('inf')
        
        for scene_id, (_, _, last_access) in self._scene_cache.items():
            if last_access < oldest_time:
                oldest_time = last_access
                oldest_scene_id = scene_id
        
        if oldest_scene_id:
            self._unload_scene(oldest_scene_id)
            return oldest_scene_id
        return None
    
    def _unload_scene(self, scene_id: str):
        """卸载指定场景"""
        if scene_id in self._scene_cache:
            gs_render, gpu_id, last_access = self._scene_cache[scene_id]
            last_access_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_access))
            
            # 释放GPU内存
            del gs_render.gaussians_list
            del gs_render.polygon_list
            del gs_render.background
            torch.cuda.empty_cache()
            
            # 更新计数
            self._gpu_scene_count[gpu_id] -= 1
            del self._scene_cache[scene_id]
            
            remaining_scenes = len(self._scene_cache)
            print(f"[SceneManager] ✖ 卸载场景: {scene_id}")
            print(f"  - GPU: {gpu_id}")
            print(f"  - 最后访问时间: {last_access_str}")
            print(f"  - 剩余已加载场景数: {remaining_scenes}/{self.max_total_scenes}")
            print(f"  - GPU场景分布: {dict(self._gpu_scene_count)}")
    
    def _load_scene(self, scene_id: str, gpu_id: int) -> GSRender:
        """加载指定场景到指定GPU"""
        scene_path = self._available_scenes.get(scene_id)
        scene_path = os.path.join(scene_path, 'gs')
        if not scene_path:
            raise ValueError(f"场景不存在: {scene_id}")
        
        print(f"[SceneManager] 开始加载场景: {scene_id} 到 GPU {gpu_id}")
        
        # 设置当前CUDA设备
        original_device = torch.cuda.current_device() if torch.cuda.is_available() else None
        torch.cuda.set_device(gpu_id)
        
        # 创建场景特定的args副本
        scene_args = self._create_scene_args(gpu_id)
        
        try:
            load_start_time = time.time()
            gs_render = GSRender(
                self.model_args,
                scene_args.iteration if hasattr(scene_args, 'iteration') else -1,
                pipeline=self.pipeline,
                opt=self.op,
                args=scene_args,
                load_ply_root=scene_path
            )
            load_duration = (time.time() - load_start_time) * 1000
            print(f"[SceneManager] ✔ 场景加载完成: {scene_id}")
            print(f"  - GPU: {gpu_id}")
            print(f"  - 加载路径: {scene_path}")
            print(f"  - 加载耗时: {load_duration:.2f} ms")
            return gs_render
        finally:
            # 恢复原始设备
            if original_device is not None:
                torch.cuda.set_device(original_device)
    
    def _create_scene_args(self, gpu_id: int):
        """为特定GPU创建参数副本"""
        import copy
        scene_args = copy.copy(self.args)
        scene_args.device = f"cuda:{gpu_id}"
        return scene_args
    
    def get_scene(self, scene_id: str) -> GSRender:
        """
        获取场景的GSRender实例
        如果场景未加载则自动加载，如果超过最大数量则淘汰最久未使用的场景
        
        Args:
            scene_id: 场景ID
            
        Returns:
            GSRender实例
        """
        with self._lock:
            # 检查场景是否存在
            if scene_id not in self._available_scenes:
                raise ValueError(f"场景不存在: {scene_id}，可用场景: {list(self._available_scenes.keys())}")
            
            # 如果场景已加载，更新访问时间并返回
            if scene_id in self._scene_cache:
                gs_render, gpu_id, _ = self._scene_cache[scene_id]
                self._scene_cache[scene_id] = (gs_render, gpu_id, time.time())
                print(f"[SceneManager] 使用缓存场景: {scene_id} (GPU {gpu_id})")
                return gs_render
            
            # 需要加载新场景
            print(f"[SceneManager] 准备加载新场景: {scene_id}")
            
            # 检查是否需要淘汰旧场景
            while len(self._scene_cache) >= self.max_total_scenes:
                print(f"[SceneManager] 已达到最大场景数限制({self.max_total_scenes})，需要淘汰旧场景")
                evicted = self._evict_oldest_scene()
                if not evicted:
                    break
            
            # 选择GPU并加载场景
            gpu_id = self._select_gpu_for_new_scene()
            gs_render = self._load_scene(scene_id, gpu_id)
            
            # 缓存场景
            self._scene_cache[scene_id] = (gs_render, gpu_id, time.time())
            self._gpu_scene_count[gpu_id] += 1
            
            loaded_count = len(self._scene_cache)
            print(f"  - 当前已加载场景数: {loaded_count}/{self.max_total_scenes}")
            print(f"  - GPU场景分布: {dict(self._gpu_scene_count)}")
            
            return gs_render
    
    def list_loaded_scenes(self) -> List[dict]:
        """列出所有已加载的场景"""
        with self._lock:
            result = []
            for scene_id, (_, gpu_id, last_access) in self._scene_cache.items():
                result.append({
                    "scene_id": scene_id,
                    "gpu_id": gpu_id,
                    "last_access": last_access,
                    "last_access_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(last_access))
                })
            return result
    
    def list_available_scenes(self) -> List[str]:
        """列出所有可用场景"""
        return list(self._available_scenes.keys())
    
    def get_status(self) -> dict:
        """获取管理器状态"""
        with self._lock:
            return {
                "scenes_root": self.scenes_root,
                "gpus": self.gpus,
                "max_scenes_per_gpu": self.max_scenes_per_gpu,
                "max_total_scenes": self.max_total_scenes,
                "loaded_scenes_count": len(self._scene_cache),
                "gpu_scene_count": dict(self._gpu_scene_count),
                "available_scenes": list(self._available_scenes.keys()),
                "loaded_scenes": self.list_loaded_scenes()
            }
    
    def preload_scene(self, scene_id: str) -> bool:
        """预加载场景"""
        try:
            self.get_scene(scene_id)
            return True
        except Exception as e:
            print(f"[SceneManager] 预加载场景失败: {scene_id}, 错误: {e}")
            return False
    
    def unload_scene(self, scene_id: str) -> bool:
        """手动卸载场景"""
        with self._lock:
            if scene_id in self._scene_cache:
                self._unload_scene(scene_id)
                return True
            return False


# 全局变量
SCENE_MANAGER: Optional[SceneManager] = None
MODEL_ARGS = None
PIPELINE_EXTRACT = None
OP_EXTRACT = None
ARGS = None

# Flask应用
thread_pool_executor = ThreadPoolExecutor(max_workers=4)
app = Flask(__name__)
CORS(app)

@app.after_request
def log_loaded_scenes_count(response):
    """每次请求后打印当前已加载场景数量"""
    if SCENE_MANAGER is None:
        print("[SceneManager] 当前已加载场景数: 0 (管理器未初始化)")
        return response
    try:
        loaded_count = len(SCENE_MANAGER.list_loaded_scenes())
        print(f"[SceneManager] 当前已加载场景数: {loaded_count}")
    except Exception as e:
        print(f"[SceneManager] 当前已加载场景数打印失败: {e}")
    return response


def get_scene_from_request() -> Tuple[str, GSRender]:
    """从请求中获取场景ID并返回对应的GSRender实例"""
    data = request.get_json() if request.is_json else {}
    scene_id = data.get('scene_id') or request.args.get('scene_id')
    
    if not scene_id:
        raise ValueError("缺少必需参数: scene_id")
    
    gs_render = SCENE_MANAGER.get_scene(scene_id)
    return scene_id, gs_render


@app.route('/ping', methods=['GET'])
def ping():
    return "pong"


@app.route('/status', methods=['GET'])
def status():
    """获取场景管理器状态"""
    return jsonify(SCENE_MANAGER.get_status())


@app.route('/scenes', methods=['GET'])
def list_scenes():
    """列出所有可用场景"""
    return jsonify({
        "available_scenes": SCENE_MANAGER.list_available_scenes(),
        "loaded_scenes": SCENE_MANAGER.list_loaded_scenes()
    })


@app.route('/preload', methods=['POST'])
def preload_scene():
    """预加载场景"""
    data = request.get_json()
    scene_id = data.get('scene_id')
    if not scene_id:
        return jsonify({"error": "缺少scene_id参数"}), 400
    
    success = SCENE_MANAGER.preload_scene(scene_id)
    return jsonify({"success": success, "scene_id": scene_id})


@app.route('/unload', methods=['POST'])
def unload_scene():
    """卸载场景"""
    data = request.get_json()
    scene_id = data.get('scene_id')
    if not scene_id:
        return jsonify({"error": "缺少scene_id参数"}), 400
    
    success = SCENE_MANAGER.unload_scene(scene_id)
    return jsonify({"success": success, "scene_id": scene_id})


@app.route('/render_gs', methods=['POST'])
def render_gs():
    """渲染单张图片（需要传入scene_id）"""
    stime = time.time()
    try:
        scene_id, gs_render = get_scene_from_request()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"加载场景失败: {str(e)}"}), 500
    
    data = request.get_json()
    cam_extrinsics = data.get('cam_extrinsics')
    cam_intrinsics = data.get('cam_intrinsics')
    camera_render_gs_path = data.get('camera_render_gs_path', './')
    return_depth_img = data.get('return_depth_img', False)
    return_depth_npz = data.get('return_depth_npz', False)
    render_scale = float(data.get('render_scale', getattr(ARGS, 'render_scale', 1.5)))  # 超采样倍率，1.0=关闭
    
    render_result = gs_render.render_set_one_image(
        [cam_extrinsics],
        [cam_intrinsics],
        camera_render_gs_path,
        return_depth_img,
        return_depth_npz,
        render_scale=render_scale,
    )
    
    etime = time.time()
    cost_time = (etime - stime) * 1000
    print(f'[{scene_id}] render_gs cost time: {cost_time:.2f} ms')
    
    if return_depth_img or return_depth_npz:
        import base64
        response_data = {
            'image': base64.b64encode(render_result['image_bytes']).decode('utf-8')
        }
        if return_depth_img and 'depth_img_bytes' in render_result:
            response_data['depth_img'] = base64.b64encode(render_result['depth_img_bytes']).decode('utf-8')
        if return_depth_npz and 'depth_npz_bytes' in render_result:
            response_data['depth_npz'] = base64.b64encode(render_result['depth_npz_bytes']).decode('utf-8')
        return jsonify(response_data)
    else:
        image_io = io.BytesIO(render_result['image_bytes'])
        return send_file(image_io, mimetype='image/jpeg')


@app.route('/render_gs_xyz_euler', methods=['POST'])
def render_gs_xyz_euler():
    """使用xyz和欧拉角渲染（需要传入scene_id）"""
    stime = time.time()
    try:
        scene_id, gs_render = get_scene_from_request()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"加载场景失败: {str(e)}"}), 500
    
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
    camera_render_gs_path = data.get('camera_render_gs_path', './')
    return_depth_img = data.get('return_depth_img', False)
    return_depth_npz = data.get('return_depth_npz', False)
    render_scale = float(data.get('render_scale', getattr(ARGS, 'render_scale', 1.5)))  # 超采样倍率，1.0=关闭
    
    render_result = gs_render.render_set_one_image(
        [cam_extrinsics],
        [cam_intrinsics],
        camera_render_gs_path,
        return_depth_img,
        return_depth_npz,
        render_scale=render_scale,
    )
    
    etime = time.time()
    cost_time = (etime - stime) * 1000
    print(f'[{scene_id}] render_gs_xyz_euler cost time: {cost_time:.2f} ms')
    
    if return_depth_img or return_depth_npz:
        import base64
        response_data = {
            'image': base64.b64encode(render_result['image_bytes']).decode('utf-8')
        }
        if return_depth_img and 'depth_img_bytes' in render_result:
            response_data['depth_img'] = base64.b64encode(render_result['depth_img_bytes']).decode('utf-8')
        if return_depth_npz and 'depth_npz_bytes' in render_result:
            response_data['depth_npz'] = base64.b64encode(render_result['depth_npz_bytes']).decode('utf-8')
        return jsonify(response_data)
    else:
        image_io = io.BytesIO(render_result['image_bytes'])
        return send_file(image_io, mimetype='image/jpeg')


@app.route('/render_gs_depth', methods=['POST'])
def render_gs_depth():
    """渲染深度图（需要传入scene_id）"""
    stime = time.time()
    try:
        scene_id, gs_render = get_scene_from_request()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"加载场景失败: {str(e)}"}), 500
    
    data = request.get_json()
    cam_extrinsics = data.get('cam_extrinsics')
    cam_intrinsics = data.get('cam_intrinsics')
    
    image_bytes = gs_render.render_set_one_depth_image(
        [cam_extrinsics],
        [cam_intrinsics]
    )
    image_io = io.BytesIO(image_bytes)
    etime = time.time()
    cost_time = (etime - stime) * 1000
    print(f'[{scene_id}] render_gs_depth cost time: {cost_time:.2f} ms')
    return send_file(image_io, mimetype='image/jpeg')


@app.route('/render_gs_multi', methods=['POST'])
def render_gs_multi():
    """多视角渲染（需要传入scene_id）"""
    stime = time.time()
    try:
        scene_id, gs_render = get_scene_from_request()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"加载场景失败: {str(e)}"}), 500
    
    data = request.get_json()
    cam_extrinsics_list = data.get('cam_extrinsics_pair_list')
    cam_intrinsics_list = data.get('cam_intrinsics_pair_list')
    camera_render_gs_path = data.get('camera_render_gs_path', './')
    return_depth_img = data.get('return_depth_img', False)
    return_depth_npz = data.get('return_depth_npz', False)
    render_scale = float(data.get('render_scale', getattr(ARGS, 'render_scale', 1.5)))  # 超采样倍率，1.0=关闭
    
    result_futures = []
    for i in range(len(cam_extrinsics_list)):
        cam_extrinsics = cam_extrinsics_list[i]
        cam_intrinsics = cam_intrinsics_list[i]
        future_res = thread_pool_executor.submit(
            gs_render.render_set_one_image,
            [cam_extrinsics],
            [cam_intrinsics],
            camera_render_gs_path,
            return_depth_img,
            return_depth_npz,
            render_scale,
        )
        result_futures.append(future_res)
    
    # 等待所有结果
    render_results = []
    for future_res in result_futures:
        render_results.append(future_res.result())
    
    etime = time.time()
    cost_time = (etime - stime) * 1000
    print(f'[{scene_id}] render_gs_multi cost time: {cost_time:.2f} ms')
    
    if return_depth_img or return_depth_npz:
        import base64
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


# 保留原有的Java交互接口
@app.route('/gs_status', methods=['GET'])
def gs_status():
    taskId = request.args.get('taskId')
    url = 'http://127.0.0.1:8001/gs/status'
    params = {'taskId': str(taskId)}
    result = make_get_request(url, params)
    return jsonify(result)


@app.route('/gs_end_status', methods=['GET'])
def gs_end_status():
    taskId = request.args.get('taskId')
    url = 'http://127.0.0.1:8001/gs/endStatus'
    params = {'taskId': str(taskId)}
    result = make_get_request(url, params)
    return jsonify(result)


@app.route('/gs_end', methods=['GET'])
def gs_end():
    taskId = request.args.get('taskId')
    ip = request.args.get('ip')
    url = 'http://127.0.0.1:8001/gs/gsEnd'
    params = {'taskId': str(taskId), 'ip': str(ip)}
    result = make_get_request(url, params)
    return jsonify(result)


@app.route('/gs_status_now', methods=['GET'])
def gs_status_now():
    taskId = request.args.get('taskId')
    ip = request.args.get('ip')
    url = 'http://127.0.0.1:8001/gs/gsStatus'
    params = {'taskId': str(taskId), 'ip': str(ip)}
    result = make_get_request(url, params)
    return jsonify(result)


@app.route('/init_task', methods=['GET'])
def init_task():
    url = 'http://127.0.0.1:8001/gs/initTask'
    params = {}
    result = make_get_request(url, params)
    return jsonify(result)


def parse_gpu_list(gpus_str: str) -> List[int]:
    """解析GPU列表字符串，如 "0,1,2" -> [0, 1, 2]"""
    return [int(g.strip()) for g in gpus_str.split(',') if g.strip()]


if __name__ == "__main__":
    torch.set_num_threads(8)
    
    # 设置命令行参数解析器
    parser = ArgumentParser(description="动态场景加载的3DGS渲染服务")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    
    # 原有参数
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--sh_degree", default=3, type=int)
    parser.add_argument("--use_transmittance_alpha", action="store_true")
    parser.add_argument("--use_transmittance_alpha_thro", type=float, default=0.5)
    parser.add_argument("--check_cam_max_dist_thro", type=float, default=3000.0)
    parser.add_argument("--img_proj_check_num_limit", type=int, default=0)
    
    # glo render
    parser.add_argument("--restore_pth", action="store_true")
    parser.add_argument("--with_glo", action="store_true")
    parser.add_argument("--glo_img_num", type=int, default=0)
    
    parser.add_argument("--app_opt_mode", type=str, default='dir-sh')
    parser.add_argument("--app_opt_out_mode", type=str, default='color')
    parser.add_argument("--app_opt_img_embed_dim", type=int, default=16)
    parser.add_argument("--app_opt_gs_feature_dim", type=int, default=32)
    parser.add_argument("--app_opt_mlp_width", type=int, default=64)
    parser.add_argument("--app_opt_mlp_depth", type=int, default=2)
    parser.add_argument("--use_label", action="store_true")
    
    parser.add_argument("--render_T_offset", nargs="+", type=float, default=[])
    parser.add_argument("--render_roll_pitch_yaw_offset", nargs="+", type=float, default=[])
    parser.add_argument("--depth_debug", action="store_true")
    
    # 新增动态场景相关参数
    parser.add_argument("--scenes_root", type=str, default='/path/to/scenes',
                        help="所有场景的根目录，每个子目录为一个场景")
    parser.add_argument("--port", type=int, default=7001,
                        help="服务端口号")
    parser.add_argument("--gpus", type=str, default="0",
                        help="使用的GPU列表，用逗号分隔，如 '0,1,2'")
    parser.add_argument("--max_scenes_per_gpu", type=int, default=10,
                        help="每个GPU最大加载场景数")
    parser.add_argument("--preload_scenes", type=str, default="",
                        #"0801_840108,0802_840243,0803_840265,0804_840284,0805_840440,0806_840803,0807_841163,0808_841200,0809_841211,0810_841216",
                        #"park2,cross_2512_1,20260316153248,cross_2512_12,cross3,20260212160022,park1,cross1,cross2,20260318lvdisoho,20260317150652,20260227170203,20260319120452,20260212163520,20260227165628",
                        help="启动时预加载的场景ID列表，用逗号分隔")
    parser.add_argument("--render_scale", type=float, default=1.5,
                        help="渲染超采样倍率，1.0=关闭")
    
    args = get_combined_args(parser)
    
    # 解析GPU列表
    gpu_list = parse_gpu_list(args.gpus)
    if not gpu_list:
        print("错误: 未指定有效的GPU")
        exit(1)
    
    # 设置CUDA可见设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    
    # 初始化系统状态
    safe_state(args.quiet)
    
    print(f"=" * 60)
    print(f"动态场景3DGS渲染服务")
    print(f"=" * 60)
    print(f"场景根目录: {args.scenes_root}")
    print(f"端口: {args.port}")
    print(f"GPU列表: {gpu_list}")
    print(f"每GPU最大场景数: {args.max_scenes_per_gpu}")
    print(f"总最大场景数: {len(gpu_list) * args.max_scenes_per_gpu}")
    print(f"=" * 60)
    
    # 提取参数
    args.app_opt_cfg = None
    model_args = model.extract(args)
    MODEL_ARGS = model_args
    PIPELINE_EXTRACT = pipeline.extract(args)
    OP_EXTRACT = op.extract(args)
    ARGS = args
    
    # 创建场景管理器
    # 注意：由于设置了CUDA_VISIBLE_DEVICES，GPU索引需要重映射
    remapped_gpus = list(range(len(gpu_list)))
    SCENE_MANAGER = SceneManager(
        scenes_root=args.scenes_root,
        gpus=remapped_gpus,
        max_scenes_per_gpu=args.max_scenes_per_gpu,
        model_args=model_args,
        pipeline=PIPELINE_EXTRACT,
        op=OP_EXTRACT,
        args=args
    )
    
    # 预加载场景
    if args.preload_scenes:
        preload_list = [s.strip() for s in args.preload_scenes.split(',') if s.strip()]
        print(f"预加载场景: {preload_list}")
        for scene_id in preload_list:
            SCENE_MANAGER.preload_scene(scene_id)
    
    print(f"\n服务启动中... 监听端口 {args.port}")
    print(f"API端点:")
    print(f"  - GET  /ping           - 健康检查")
    print(f"  - GET  /status         - 获取管理器状态")
    print(f"  - GET  /scenes         - 列出所有场景")
    print(f"  - POST /preload        - 预加载场景")
    print(f"  - POST /unload         - 卸载场景")
    print(f"  - POST /render_gs      - 渲染图片 (需要scene_id)")
    print(f"  - POST /render_gs_xyz_euler - 使用xyz欧拉角渲染")
    print(f"  - POST /render_gs_depth - 渲染深度图")
    print(f"  - POST /render_gs_multi - 多视角渲染")
    print(f"=" * 60)
    
    app.run(host="0.0.0.0", debug=False, port=args.port, threaded=True)

# 使用示例:
# conda activate ../../../envs/vllm_gw/
# python render_sim_dynamic.py --port 7001 --gpus 0 --max_scenes_per_gpu 1
# python render_sim_dynamic.py --scenes_root /path/to/all/scenes --port 7001 --gpus 0,1 --max_scenes_per_gpu 3
#
# 请求示例:
# curl -X POST http://localhost:7001/render_gs \
#   -H "Content-Type: application/json" \
#   -d '{
#     "scene_id": "scene_001",
#     "cam_extrinsics": "...",
#     "cam_intrinsics": "..."
#   }'
