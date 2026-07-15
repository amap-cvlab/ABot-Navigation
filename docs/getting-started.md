# Getting Started

[English](getting-started.md) | [中文](zh-CN/getting-started.md)

## Prerequisites

- Linux (Ubuntu 20.04+)
- NVIDIA GPU, Compute Capability 7.0+, >= 24 GB VRAM
- CUDA toolkit (any version supported by your PyTorch)
- [Conda](https://docs.conda.io/en/latest/miniconda.html)

## 1. Install the Evaluator

```bash
pip install abotn-bench
```

Or from source:

```bash
git clone <repository_url>
cd <repository_name> && pip install -e .
```

Verify: `python -c "import abotn_evaluator; print('OK')"`

## 2. Deploy the Render Server

The render server provides multi-view RGB observations from 3DGS scenes. It runs in a separate conda environment with PyTorch + CUDA.

```bash
# Create env with base dependencies
conda env create -f render_server/environment.yml
conda activate abotn_render

# Install PyTorch matching your CUDA (see https://pytorch.org)
pip install torch --index-url https://download.pytorch.org/whl/cu118  # example for CUDA 11.8

# Compile CUDA extensions
cd render_server
pip install --no-build-isolation ./submodules/diff-plane-rasterization
pip install --no-build-isolation ./submodules/simple-knn
```

Edit the startup script to set `SCENES_ROOT` (path to your 3DGS `.ply` scene files), then start:

```bash
bash scripts/start_PointGoal_outdoor_render_server.sh
```

| Script | Task | RENDER_SCALE |
|:-------|:-----|:-------------|
| `scripts/start_PointGoal_outdoor_render_server.sh` | Point-Goal Outdoor | 1.0 |
| `scripts/start_PointGoal_indoor_render_server.sh` | Point-Goal Indoor | 1.0 |
| `scripts/start_POIGoal_render_server.sh` | POI-Goal | 1.5 |

Configurable variables in each script:

| Variable | Default | Description |
|:---------|:--------|:------------|
| `SCENES_ROOT` | — | Path to 3DGS scene directory (required) |
| `PORT` | 7036 | HTTP listen port |
| `GPUS` | `"0"` | GPU IDs, e.g. `"0,1,2"` |
| `MAX_SCENES_PER_GPU` | 1 | Scenes loaded per GPU |

Ready when logs show `Listening on http://localhost:7036/render_gs`.

## 3. Download Data

| Dataset | Contents | Download |
|:--------|:---------|:---------|
| **ABotN-PointBench** | 31 scenes, 465 trajectories, occupancy maps | [🤗 HuggingFace](https://huggingface.co/datasets/acvlab/ABotN-PointBench) \| [🤖 ModelScope](https://www.modelscope.cn/datasets/amap_cvlab/ABotN-PointBench) |
| **ABotN-POIBench** | 11 areas, 163 POIs, occupancy maps | [🤗 HuggingFace](https://huggingface.co/datasets/acvlab/ABotN-POIBench) \| [🤖 ModelScope](https://www.modelscope.cn/datasets/amap_cvlab/ABotN-POIBench) |

Expected layout after extraction:

```
ABotN-PointBench/
├── Indoor/
│   ├── annotations/          # 16 indoor scenes, each with traj_*.json
│   └── occmaps/              # occupancy maps (same scene structure)
└── Outdoor/
    ├── annotations/          # 15 outdoor scenes, each with traj_*.json
    └── occmaps/

ABotN-POIBench/
├── annotations/              # 11 scenes, each with traj_*.json
└── occmaps/
```

3DGS scene `.ply` files should be extracted to a separate directory used as `SCENES_ROOT`.

When running evaluation, pass `annotations/` as `--data-dir` and `occmaps/` as `--map-dir`.

## Next Steps

- [Point-Goal Evaluation](point-goal.md)
- [POI-Goal Evaluation](poi-goal.md)
- [API Reference](api-reference.md)
