<div align="center">

# ABotN-Bench

[![arXiv](https://img.shields.io/badge/arXiv-2607.10383-b31b1b.svg)](https://arxiv.org/abs/2607.10383)
[![PDF](https://img.shields.io/badge/PDF-Technical_Report-blue)](https://arxiv.org/pdf/2607.10383)
[![Project Page](https://img.shields.io/badge/🌐-Project_Page-green)](https://amap-cvlab.github.io/ABot-Navigation/ABot-N1/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![🤗 HuggingFace](https://img.shields.io/badge/🤗-Datasets-yellow)](https://huggingface.co/acvlab)
[![🤖 ModelScope](https://img.shields.io/badge/ModelScope-Datasets-purple)](https://www.modelscope.cn/organization/amap_cvlab)

[English](README.md) | [中文](README_zh-CN.md)

</div>

We introduce two complementary benchmarks built on the same high-fidelity 3D Gaussian Splatting (3DGS) reconstruction stack — **ABotN-PointBench** for coordinate-conditioned navigation and **ABotN-POIBench** for name-conditioned POI navigation — to advance evaluation of closed-loop, social-rule-aware visual navigation in real-world indoor and outdoor environments. We additionally release a re-curated **Short-Horizon OVON** variant that isolates the recognition-and-approach phase of object-goal navigation. All benchmarks and the evaluation toolkit are released as part of the [ABot-N1](https://arxiv.org/abs/2607.10383) project.

<p align="center">
  <img src="assets/benchmark_overview.jpg" alt="ABotN-PointBench and ABotN-POIBench overview">
</p>

## Benchmarks

| Benchmark | Task | Goal | Scenes | Episodes |
|:----------|:-----|:-----|:-------|:---------|
| **ABotN-PointBench** | Point-Goal | Navigate to (x, y) coordinates | 31 real-world 3DGS scenes (16 indoor + 15 outdoor) | 465 |
| **ABotN-POIBench** | POI-Goal | Navigate to a named POI entrance | 11 commercial areas, 126k m² | 163 POIs |
| **Short-Horizon OVON** | Object-Goal | Find and approach a target object category | 36 HM3D scenes (OVON Val-Unseen) | 2,443 |

ABotN-PointBench and ABotN-POIBench are newly introduced benchmarks with full evaluation tooling in this repository. Short-Horizon OVON is a visibility-filtered variant of the [HM3D-OVON](https://github.com/naokiyokoyama/ovon) benchmark; see [its documentation](docs/short-horizon-ovon.md) for integration details.

## 📦 Datasets — Released

All benchmark data is publicly available on 🤗 HuggingFace and 🤖 ModelScope:

| Dataset | Download |
|:--------|:---------|
| ABotN-PointBench | [🤗 HuggingFace](https://huggingface.co/datasets/acvlab/ABotN-PointBench) \| [🤖 ModelScope](https://www.modelscope.cn/datasets/amap_cvlab/ABotN-PointBench) |
| ABotN-POIBench | [🤗 HuggingFace](https://huggingface.co/datasets/acvlab/ABotN-POIBench) \| [🤖 ModelScope](https://www.modelscope.cn/datasets/amap_cvlab/ABotN-POIBench) |
| ABotN-Short-Horizon-OVON | [🤗 HuggingFace](https://huggingface.co/datasets/acvlab/ABotN-Short-Horizon-OVON) \| [🤖 ModelScope](https://www.modelscope.cn/datasets/amap_cvlab/ABotN-Short-Horizon-OVON) |

See [Getting Started](docs/getting-started.md) for download and setup instructions.

**Key features:**

- Photorealistic multi-view RGB observations from real-world 3DGS scene reconstructions
- Closed-loop evaluation with collision-aware success metrics (SR<sub>&lt;3col</sub> outdoor, SR<sub>&lt;1col</sub> indoor)
- Social-rule-aware traversability scoring via annotated walkability maps
- Standardized evaluation protocols with fixed thresholds for reproducible comparison
- Minimal agent interface: implement `reset()` + `predict()` to evaluate any model

## Benchmark Results

### Point-Goal — ABotN-PointBench (Outdoor)

| Method | SR<sub>&lt;3col</sub>↑ | SPL↑ |
|:-------|:---:|:----:|
| GNM | 39.1 | 36.7 |
| ViNT | 62.2 | 62.2 |
| NoMaD | 56.0 | 55.7 |
| CityWalker | 48.9 | 48.3 |
| SocialNav | 72.0 | 71.9 |
| ABot-N1 | **92.9** | **91.4** |

### Point-Goal — ABotN-PointBench (Indoor)

| Method | SR<sub>&lt;1col</sub>↑ | SPL↑ |
|:-------|:---:|:----:|
| GNM | 26.7 | 26.6 |
| ViNT | 27.9 | 27.9 |
| NoMaD | 20.0 | 19.6 |
| CityWalker | 21.7 | 21.6 |
| SocialNav | 42.5 | 42.5 |
| ABot-N1 | **95.4** | **93.7** |

### POI-Goal — ABotN-POIBench

| Method | SR<sub>&lt;2m</sub>↑ | SPL↑ |
|:-------|:---:|:----:|
| ViNT | 19.0 | 18.2 |
| OmniNav (vanilla) | 23.9 | 22.4 |
| OmniNav (BridgeNav) | 34.4 | 31.5 |
| POINav | 42.3 | 40.3 |
| ABot-N1 | **77.3** | **72.6** |

### Object-Goal — Short-Horizon OVON

| Method | SR↑ | SPL↑ | DTG↓ |
|:-------|:---:|:----:|:----:|
| StreamVLN | 39.7 | 15.8 | 2.368 |
| NaVILA | 55.4 | 26.1 | 1.811 |
| Uni-NaVid | 68.7 | 34.5 | 1.495 |
| ABot-N1 | **84.9** | **51.8** | **0.822** |

> Full results and per-difficulty breakdowns available in the [technical report](https://arxiv.org/abs/2607.10383).

## Architecture

```
┌──────────────────────────┐         ┌──────────────────────────────┐
│  3DGS Render Server      │  HTTP   │  Evaluation Environment      │
│  Python 3.8, CUDA 11     │◄───────►│  pip install abotn-bench     │
│  render_server/           │         │  import abotn_evaluator      │
└──────────────────────────┘         └──────────────────────────────┘
```

## Quick Start

```bash
# Install
pip install abotn-bench

# Deploy render server (separate conda env, CUDA 11 required)
conda env create -f render_server/environment.yml
conda activate abotn_render
bash scripts/start_PointGoal_outdoor_render_server.sh   # set SCENES_ROOT first

# Evaluate your agent
python -m abotn_evaluator.point_goal.runner \
    --agent-module your_agent:YourAgent \
    --data-dir /path/to/pointbench/outdoor/trajectory \
    --render-url http://localhost:7036/render_gs \
    --mode outdoor
```

## Agent Interface

```python
from abotn_evaluator.interface.point_goal import BasePointGoalAgent, Observation, WaypointPrediction

class YourAgent(BasePointGoalAgent):
    def reset(self): ...
    def predict(self, observation: Observation) -> WaypointPrediction:
        # observation.images: Dict[str, ndarray] — multi-view RGB (left/front/right)
        # observation.target_position: ndarray — [front, left] in metres
        # observation.distance_to_goal: float
        return WaypointPrediction(waypoint=..., arrive=...)
```

For POI-Goal, use `BasePoiGoalAgent` — the observation adds a `poi_name: str` field.

## Metrics

| Task | Metrics | Description |
|:-----|:--------|:------------|
| Point-Goal (Outdoor) | SR<sub>&lt;3col</sub>, SPL | Success rate under a 3-collision budget; path efficiency vs. A* reference |
| Point-Goal (Indoor) | SR<sub>&lt;1col</sub>, SPL | Success rate under strict zero-collision criterion |
| POI-Goal | SR<sub>&lt;2m</sub>, SPL | Entrance arrival within 2 m; global and per-POI |
| Object-Goal (Short-Horizon OVON) | SR, SPL, DTG | Success rate, path efficiency, distance-to-goal at termination |

## Documentation

| | |
|:--|:--|
| [Getting Started](docs/getting-started.md) | Installation, render server setup, data download |
| [Point-Goal Evaluation](docs/point-goal.md) | Outdoor/indoor protocol, evaluation commands |
| [POI-Goal Evaluation](docs/poi-goal.md) | POI-Goal protocol, evaluation commands |
| [Short-Horizon OVON](docs/short-horizon-ovon.md) | Visibility-filtered Object-Goal variant for Habitat-sim |
| [API Reference](docs/api-reference.md) | Observation/WaypointPrediction fields, coordinate system, CLI flags |
| [Custom Agents](docs/custom-agents.md) | Coordinate adaptation, wrapper pattern |

## Citation

If you find ABotN-Bench useful in your research, please cite the [technical report](https://arxiv.org/abs/2607.10383):

```bibtex
@misc{gong2026abotn1,
  title={ABot-N1: Toward a General Visual Language Navigation Foundation Model},
  author={Ruiyan Gong and Yingnan Guo and Junjun Hu and Jintao Kong and Xiaoxu Leng and Tianlun Li and Weize Li and Fei Liu and Zhicheng Liu and Jia Lu and Minghua Luo and Chenlin Ming and Yanfen Shen and Jiyue Tao and Zhengbo Wang and Mingyang Yin and Minqi Gu and Zihao Guan and Wei Guo and Guoqing Liu and Huachong Pang and Menglin Yang and Zeqian Ye and Xiaoxiao Geng and Zhining Gu and Honglin Han and Di Jing and Hongyu Pan and Mingchao Sun and Kuan Yang and Jianfang Zhang and Yanghong Chen and Ye He and Wei Mei and Jiahao Shi and Xiangpo Yang and Yanqing Zhu and Zedong Chu and Xiaolong Wu and Mu Xu},
  year={2026},
  eprint={2607.10383},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2607.10383},
}
```

## License

Apache-2.0. `render_server/` retains the [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) license.

## Acknowledgments

ABotN-Bench is developed by **AMAP CV Lab** as part of the [ABot-N1](https://amap-cvlab.github.io/ABot-Navigation/ABot-N1/) project. We thank the open-source community for [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), [Habitat](https://aihabitat.org/), and [HM3D-OVON](https://github.com/naokiyokoyama/ovon).
