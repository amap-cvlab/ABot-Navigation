# 3DGS Render Server

This directory contains a 3D Gaussian Splatting render server derived from [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting). It provides HTTP APIs that render multi-view RGB/depth images from Gaussian Splatting scene models, used by the Point-Goal and POI-Goal evaluators.

## Hardware Requirements

| Item | Requirement |
|:-----|:------------|
| GPU | NVIDIA, Compute Capability 7.0+ (Volta / Turing / Ampere) |
| VRAM | >= 24 GB per GPU |
| CUDA | Any version supported by your PyTorch build |
| OS | Linux (Ubuntu 20.04+) |

## Environment Setup

### 1. Create conda environment

```bash
conda env create -f render_server/environment.yml
conda activate abotn_render
```

This installs Python and non-GPU dependencies (numpy, scipy, flask, etc.).

### 2. Install PyTorch

Install PyTorch with CUDA support matching your GPU. Visit [pytorch.org](https://pytorch.org/get-started/locally/) to get the correct command for your CUDA version:

```bash
# Example for CUDA 11.8:
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Example for CUDA 12.1:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Verify: `python -c "import torch; print(torch.cuda.is_available())"`

### 3. Compile CUDA extensions

The two CUDA extensions must be compiled with `--no-build-isolation` (their `setup.py` imports torch at the top level):

```bash
cd render_server
pip install --no-build-isolation ./submodules/diff-plane-rasterization
pip install --no-build-isolation ./submodules/simple-knn
```

Verify: `python -c "import diff_plane_rasterization; import simple_knn; print('OK')"`

## Quick Start

```bash
conda activate abotn_render
python render_sim_dynamic.py --port 7036 --gpus 0 --scenes_root /path/to/3dgs_scenes
```

Ready when logs show `Listening on http://localhost:7036/render_gs`.

## Startup Scripts

Convenience scripts in `scripts/` pre-configure each evaluation task:

| Script | Task | RENDER_SCALE |
|:-------|:-----|:-------------|
| `scripts/start_PointGoal_outdoor_render_server.sh` | Point-Goal Outdoor | 1.0 |
| `scripts/start_PointGoal_indoor_render_server.sh` | Point-Goal Indoor | 1.0 |
| `scripts/start_POIGoal_render_server.sh` | POI-Goal | 1.5 |

Edit `SCENES_ROOT` in the script before running:

```bash
bash scripts/start_PointGoal_outdoor_render_server.sh
```

### Configurable Variables

| Variable | Default | Description |
|:---------|:--------|:------------|
| `SCENES_ROOT` | — | Path to 3DGS scene directory (required) |
| `PORT` | 7036 | HTTP listen port |
| `GPUS` | `"0"` | GPU IDs, comma-separated, e.g. `"0,1,2"` |
| `MAX_SCENES_PER_GPU` | 1 | Max scenes loaded per GPU |
| `RENDER_SCALE` | 1.0 / 1.5 | Supersampling factor |

## Multi-GPU

Pass multiple GPU IDs to distribute scenes across GPUs:

```bash
python render_sim_dynamic.py --port 7036 --gpus 0,1,2 --max_scenes_per_gpu 2 --scenes_root /path/to/scenes
```

Scenes are assigned to GPUs in round-robin order and managed with LRU eviction when the per-GPU limit is reached.

## API Endpoints

### Core Rendering

| Method | Path | Description |
|:-------|:-----|:------------|
| POST | `/render_gs` | Render a single image for a scene |
| POST | `/render_gs_xyz_euler` | Render by XYZ position + Euler angles |
| POST | `/render_gs_multi` | Render multiple views in one request |
| POST | `/render_gs_depth` | Render depth map |

### Scene Management

| Method | Path | Description |
|:-------|:-----|:------------|
| GET | `/ping` | Health check (returns `"pong"`) |
| GET | `/status` | Scene manager status (GPU usage, loaded scenes) |
| GET | `/scenes` | List available and loaded scenes |
| POST | `/preload` | Preload a scene (`{"scene_id": "..."}`) |
| POST | `/unload` | Unload a scene (`{"scene_id": "..."}`) |

### Legacy

| Method | Path | Description |
|:-------|:-----|:------------|
| GET | `/gs_status` | Task status query |
| GET | `/gs_end` | End task session |
| GET | `/init_task` | Initialize task |

## Architecture Note

The render server runs in a separate conda environment from the evaluator. It communicates with `abotn_evaluator` via HTTP — this separation is by design: the render server needs PyTorch + CUDA for GPU rendering, while the evaluator is a pure-Python lightweight package that can run in any environment.

## Further Reading

- [Getting Started](../docs/getting-started.md) — full installation walkthrough
- [Point-Goal Evaluation](../docs/point-goal.md)
- [POI-Goal Evaluation](../docs/poi-goal.md)
