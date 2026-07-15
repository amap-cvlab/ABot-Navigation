# 3DGS 渲染服务

本目录包含基于 [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) 的 3D 高斯泼溅渲染服务。它提供 HTTP API，从高斯泼溅场景模型渲染多视角 RGB/深度图像，供 Point-Goal 和 POI-Goal 评测器使用。

## 硬件要求

| 项目 | 要求 |
|:-----|:-----|
| GPU | NVIDIA，Compute Capability 7.0+（Volta / Turing / Ampere） |
| 显存 | 每 GPU >= 24 GB |
| CUDA | 与你的 PyTorch 版本匹配即可 |
| 操作系统 | Linux（Ubuntu 20.04+） |

## 环境搭建

### 1. 创建 conda 环境

```bash
conda env create -f render_server/environment.yml
conda activate abotn_render
```

此步骤安装 Python 及非 GPU 依赖（numpy、scipy、flask 等）。

### 2. 安装 PyTorch

安装匹配你 GPU 的 PyTorch（带 CUDA 支持）。访问 [pytorch.org](https://pytorch.org/get-started/locally/) 获取适合你 CUDA 版本的安装命令：

```bash
# CUDA 11.8 示例：
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1 示例：
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

验证：`python -c "import torch; print(torch.cuda.is_available())"`

### 3. 编译 CUDA 扩展

两个 CUDA 扩展必须使用 `--no-build-isolation` 编译（其 `setup.py` 在顶层导入了 torch）：

```bash
cd render_server
pip install --no-build-isolation ./submodules/diff-plane-rasterization
pip install --no-build-isolation ./submodules/simple-knn
```

验证：`python -c "import diff_plane_rasterization; import simple_knn; print('OK')"`

## 快速启动

```bash
conda activate abotn_render
python render_sim_dynamic.py --port 7036 --gpus 0 --scenes_root /path/to/3dgs_scenes
```

日志显示 `Listening on http://localhost:7036/render_gs` 即表示就绪。

## 启动脚本

`scripts/` 目录下的便捷脚本已为各评测任务预配置：

| 脚本 | 任务 | RENDER_SCALE |
|:-----|:-----|:-------------|
| `scripts/start_PointGoal_outdoor_render_server.sh` | Point-Goal Outdoor | 1.0 |
| `scripts/start_PointGoal_indoor_render_server.sh` | Point-Goal Indoor | 1.0 |
| `scripts/start_POIGoal_render_server.sh` | POI-Goal | 1.5 |

运行前需编辑脚本中的 `SCENES_ROOT`：

```bash
bash scripts/start_PointGoal_outdoor_render_server.sh
```

### 可配置变量

| 变量 | 默认值 | 说明 |
|:-----|:-------|:-----|
| `SCENES_ROOT` | — | 3DGS 场景目录（必填） |
| `PORT` | 7036 | HTTP 监听端口 |
| `GPUS` | `"0"` | GPU 编号，逗号分隔，如 `"0,1,2"` |
| `MAX_SCENES_PER_GPU` | 1 | 每 GPU 最大加载场景数 |
| `RENDER_SCALE` | 1.0 / 1.5 | 渲染超采样倍率 |

## 多 GPU 部署

传入多个 GPU 编号将场景分布到多张显卡：

```bash
python render_sim_dynamic.py --port 7036 --gpus 0,1,2 --max_scenes_per_gpu 2 --scenes_root /path/to/scenes
```

场景以轮询方式分配到各 GPU，当单 GPU 加载场景数达到上限时使用 LRU 策略淘汰。

## API 端点

### 核心渲染

| 方法 | 路径 | 说明 |
|:-----|:-----|:-----|
| POST | `/render_gs` | 渲染单张场景图片 |
| POST | `/render_gs_xyz_euler` | 通过 XYZ 坐标 + 欧拉角渲染 |
| POST | `/render_gs_multi` | 单次请求渲染多视角 |
| POST | `/render_gs_depth` | 渲染深度图 |

### 场景管理

| 方法 | 路径 | 说明 |
|:-----|:-----|:-----|
| GET | `/ping` | 健康检查（返回 `"pong"`） |
| GET | `/status` | 场景管理器状态（GPU 使用情况、已加载场景） |
| GET | `/scenes` | 列出可用场景和已加载场景 |
| POST | `/preload` | 预加载场景（`{"scene_id": "..."}`） |
| POST | `/unload` | 卸载场景（`{"scene_id": "..."}`） |

### 旧版接口

| 方法 | 路径 | 说明 |
|:-----|:-----|:-----|
| GET | `/gs_status` | 任务状态查询 |
| GET | `/gs_end` | 结束任务会话 |
| GET | `/init_task` | 初始化任务 |

## 架构说明

渲染服务运行在独立的 conda 环境中，与评测器分离。它通过 HTTP 与 `abotn_evaluator` 通信——这种分离是设计上的考量：渲染服务需要 PyTorch + CUDA 进行 GPU 渲染，而评测器是纯 Python 轻量包，可运行在任意环境中。

## 相关文档

- [快速上手](../docs/zh-CN/getting-started.md) — 完整安装指引
- [Point-Goal 评测](../docs/zh-CN/point-goal.md)
- [POI-Goal 评测](../docs/zh-CN/poi-goal.md)
