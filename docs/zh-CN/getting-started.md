# 快速上手

[English](../getting-started.md) | [中文](getting-started.md)

## 环境要求

- Linux (Ubuntu 20.04+)
- NVIDIA GPU，Compute Capability 7.0+，显存 >= 24 GB
- CUDA toolkit（与你的 PyTorch 版本匹配即可）
- [Conda](https://docs.conda.io/en/latest/miniconda.html)

## 1. 安装评测框架

从源码安装：

```bash
git clone <仓库地址>
cd <仓库目录名> && pip install -e .
```

验证：`python -c "import abotn_evaluator; print('OK')"`

## 2. 部署渲染服务

渲染服务从 3DGS 场景生成多视角 RGB 观测，运行在独立 conda 环境中（需 PyTorch + CUDA）。

```bash
# 创建基础环境
conda env create -f render_server/environment.yml
conda activate abotn_render

# 安装 PyTorch（按你的 CUDA 版本选择，参考 https://pytorch.org）
pip install torch --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8 示例

# 编译 CUDA 扩展
cd render_server
pip install --no-build-isolation ./submodules/diff-plane-rasterization
pip install --no-build-isolation ./submodules/simple-knn
```

编辑启动脚本设置 `SCENES_ROOT`（3DGS `.ply` 场景文件目录），然后启动：

```bash
bash scripts/start_PointGoal_outdoor_render_server.sh
```

| 脚本 | 任务 | RENDER_SCALE |
|:-----|:-----|:-------------|
| `scripts/start_PointGoal_outdoor_render_server.sh` | Point-Goal Outdoor | 1.0 |
| `scripts/start_PointGoal_indoor_render_server.sh` | Point-Goal Indoor | 1.0 |
| `scripts/start_POIGoal_render_server.sh` | POI-Goal | 1.5 |

脚本可配置变量：

| 变量 | 默认值 | 说明 |
|:-----|:-------|:-----|
| `SCENES_ROOT` | — | 3DGS 场景目录（必填） |
| `PORT` | 7036 | HTTP 监听端口 |
| `GPUS` | `"0"` | GPU 编号，如 `"0,1,2"` |
| `MAX_SCENES_PER_GPU` | 1 | 每 GPU 加载场景数 |

日志显示 `Listening on http://localhost:7036/render_gs` 即就绪。

## 3. 下载数据

| 数据集 | 内容 | 下载 |
|:-------|:-----|:-----|
| **ABotN-PointBench** | 31 个场景，465 条轨迹，占用栅格图 | [🤗 HuggingFace](https://huggingface.co/datasets/acvlab/ABotN-PointBench) \| [🤖 ModelScope](https://www.modelscope.cn/datasets/amap_cvlab/ABotN-PointBench) |
| **ABotN-POIBench** | 11 个区域，163 个 POI，占用栅格图 | [🤗 HuggingFace](https://huggingface.co/datasets/acvlab/ABotN-POIBench) \| [🤖 ModelScope](https://www.modelscope.cn/datasets/amap_cvlab/ABotN-POIBench) |

解压后目录结构：

```
ABotN-PointBench/
├── Indoor/
│   ├── annotations/          # 16 个室内场景，每个含 traj_*.json
│   └── occmaps/              # 占用栅格图（场景结构同 annotations）
└── Outdoor/
    ├── annotations/          # 15 个室外场景，每个含 traj_*.json
    └── occmaps/

ABotN-POIBench/
├── annotations/              # 11 个场景，每个含 traj_*.json
└── occmaps/
```

3DGS 场景 `.ply` 文件单独放置，路径用作 `SCENES_ROOT`。

运行评测时，将 `annotations/` 作为 `--data-dir`，`occmaps/` 作为 `--map-dir`。

## 后续步骤

- [Point-Goal 评测](point-goal.md)
- [POI-Goal 评测](poi-goal.md)
- [API 参考](api-reference.md)
