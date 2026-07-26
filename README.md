<div align="center">

# ABot-N0

### A Unified VLA Foundation Model for Versatile Embodied Navigation

[![Paper](https://img.shields.io/badge/📄-Technical_Report-blue)](https://arxiv.org/pdf/2602.11598)
[![Project Page](https://img.shields.io/badge/🌐-Project_Page-green)](https://amap-cvlab.github.io/ABot-Navigation/ABot-N0/)

**AMAP CV Lab**

</div>

---

## 🔥 News

- **[2026/02]** 🎉 Technical Report released! [Read the paper](https://arxiv.org/pdf/2602.11598)

## 📖 Introduction

<div align="center">
<video src="https://github.com/user-attachments/assets/a2c2772a-c86e-454a-9d00-a232b71b8a87" width="80%" controls autoplay loop muted>
</video>
</div>

**ABot-N0** is a unified **Vision-Language-Action (VLA)** foundation model that achieves a **"Grand Unification"** across **5 core embodied navigation tasks**:

| Task | Description |
|:---:|:---|
| 🎯 **Point-Goal** | Reach precise metric coordinates for robust locomotion and obstacle avoidance |
| 🔍 **Object-Goal** | Search for and navigate to a specific object category in unseen environments |
| 📝 **Instruction-Following** | Execute complex natural language navigation instructions |
| 📍 **POI-Goal** | Navigate to specific Points of Interest and their physical entrances |
| 🚶 **Person-Following** | Real-time tracking and following of dynamic human targets |

ABot-N0 adopts a hierarchical **"Brain-Action"** architecture:
- **Universal Multi-Modal Encoder** — unifies heterogeneous inputs (RGB, visual history, goals) into a shared latent space
- **Cognitive Brain** — a pre-trained LLM (Qwen3-4B) for deep semantic understanding and spatial reasoning
- **Action Expert** — Flow Matching-based trajectory generator for precise, continuous control

## 📊 Key Highlights

| | |
|:---|:---|
| **Unified Tasks** | 5 core navigation paradigms in a single model |
| **SOTA Benchmarks** | New state-of-the-art on **7** authoritative benchmarks |
| **Data Scale** | **16.9M** expert trajectories + **5.0M** reasoning samples |
| **3D Scenes** | **7,802** high-fidelity scenes covering **10.3 km²** |
| **Real-world Deployment** | Deployed on Unitree Go2 with NVIDIA Jetson Orin NX, achieving 2Hz VLA inference |

## 🏗️ Architecture

ABot-N0 follows a hierarchical "Brain-Action" design comprising three pillars:

1. **Universal Multi-Modal Encoder**: Supports flexible vision inputs (panoramic / front-view), heterogeneous goal definitions (text-based semantic goals & point-based geometric goals), and reasoning task encoding.

2. **Cognitive Brain**: Built upon a pre-trained LLM, it supports dual-mode operation — a *Reasoning Head* for high-level semantic understanding and an *Action Head* for motion planning.

3. **Action Expert**: Employs Flow Matching to generate multi-modal trajectory distributions (5 waypoints with position + yaw), enabling precise continuous control.

## 📦 Data Engine

<div align="center">
<a href="https://www.youtube.com/watch?v=EonbUL-1uPo">
<img src="https://img.youtube.com/vi/EonbUL-1uPo/maxresdefault.jpg" width="80%" alt="ABot-N0 Data Engine Video">
</a>
<p><em>▶️ Click to watch the Data Engine video</em></p>
</div>

The **ABot-N0 Data Engine** is the largest embodied navigation data pipeline, integrating three synergistic layers:

- **High-Fidelity 3D Scene Ecosystem**: 7,802 scenes (indoor: homes, offices, malls, stations; outdoor: intersections, parks, virtual city) covering 10.3 km²
- **Universal Trajectories Dataset**: ~16.9M expert trajectories across 5 navigation paradigms
- **Cognitive Reasoning Dataset**: ~5.0M reasoning samples grounding decision-making in spatial-social logic

## 🏋️ Training Recipe

ABot-N0 is trained via a three-stage curriculum:

1. **Phase 1 — Cognitive Warm-up**: Fine-tune the LLM backbone on reasoning tasks to learn "what to see" and "how to reason"
2. **Phase 2 — Unified Sensorimotor SFT**: Joint multi-task training with dual-head optimization (AR reasoning + Flow Matching actions)
3. **Phase 3 — SAFE-GRPO**: Post-training value alignment via socially-aware reinforcement learning for social compliance

## 🤖 Agentic Navigation System

Beyond the foundation model, we propose an **Agentic Navigation System** for real-world deployment:

- **Agentic Planner**: VLM-powered intent decomposition with CoT reasoning and closed-loop self-reflection
- **Topo-Memory (Map-as-Memory)**: Hierarchical topological memory for cross-scale spatial knowledge (Block → Road → Function → Object/POI layers)
- **Neural Controller**: High-speed reactive control (>10Hz) bridging strategic waypoints and real-time execution
- **Hardware**: Unitree Go2 quadrupedal robot + NVIDIA Jetson Orin NX (157 TOPS)

## 🎬 Video Demos

### 🌍 Long-Horizon Agentic Missions

<table>
<tr>
<td align="center" width="50%">
<a href="https://www.youtube.com/watch?v=pm3SP6HgwkI">
<img src="https://img.youtube.com/vi/pm3SP6HgwkI/maxresdefault.jpg" width="100%" alt="Indoor Long-Range Mission">
</a>
<br><b>▶️ Indoor Long-Range Mission</b>
</td>
<td align="center" width="50%">
<a href="https://www.youtube.com/watch?v=TZl9OfadPJg">
<img src="https://img.youtube.com/vi/TZl9OfadPJg/maxresdefault.jpg" width="100%" alt="Outdoor Long-Range Mission">
</a>
<br><b>▶️ Outdoor Long-Range Mission</b>
</td>
</tr>
</table>

### 🤝 Real-World Applications

<table>
<tr>
<td align="center" width="50%">
<a href="https://www.youtube.com/watch?v=7DIoE-g671g">
<img src="https://img.youtube.com/vi/7DIoE-g671g/maxresdefault.jpg" width="100%" alt="Guide Dog Assistance">
</a>
<br><b>▶️ Guide Dog Assistance</b>
</td>
<td align="center" width="50%">
<a href="https://www.youtube.com/watch?v=7UEisQYBF9k">
<img src="https://img.youtube.com/vi/7UEisQYBF9k/maxresdefault.jpg" width="100%" alt="Interactive Companion">
</a>
<br><b>▶️ Interactive Companion</b>
</td>
</tr>
</table>

### 🎯 Single-Task Capabilities

<table>
<tr>
<td align="center" width="33%">
<a href="https://www.youtube.com/watch?v=--pVvU9BOX4">
<img src="https://img.youtube.com/vi/--pVvU9BOX4/maxresdefault.jpg" width="100%" alt="Point-Goal Navigation">
</a>
<br><b>▶️ Point-Goal</b>
</td>
<td align="center" width="33%">
<a href="https://www.youtube.com/watch?v=tfv4L3knYlk">
<img src="https://img.youtube.com/vi/tfv4L3knYlk/maxresdefault.jpg" width="100%" alt="Object-Goal Navigation">
</a>
<br><b>▶️ Object-Goal</b>
</td>
<td align="center" width="33%">
<a href="https://www.youtube.com/watch?v=WdNuaxVtb3A">
<img src="https://img.youtube.com/vi/WdNuaxVtb3A/maxresdefault.jpg" width="100%" alt="Instruction-Following">
</a>
<br><b>▶️ Instruction-Following</b>
</td>
</tr>
<tr>
<td align="center" width="33%">
<a href="https://www.youtube.com/watch?v=6YAANa1Oico">
<img src="https://img.youtube.com/vi/6YAANa1Oico/maxresdefault.jpg" width="100%" alt="POI-Goal Navigation">
</a>
<br><b>▶️ POI-Goal</b>
</td>
<td align="center" width="33%">
<a href="https://www.youtube.com/watch?v=dZftcxjN9f8">
<img src="https://img.youtube.com/vi/dZftcxjN9f8/maxresdefault.jpg" width="100%" alt="Person-Following">
</a>
<br><b>▶️ Person-Following</b>
</td>
<td align="center" width="33%">
</td>
</tr>
</table>

## 📈 Benchmark Results

ABot-N0 achieves **new SOTA** on 7 benchmarks:

- **CityWalker** (Point-Goal, Open-Loop)
- **SocNav** (Point-Goal, Closed-Loop)
- **VLN-CE R2R** (Instruction-Following)
- **VLN-CE RxR** (Instruction-Following)
- **HM3D-OVON** (Object-Goal)
- **BridgeNav** (POI-Goal)
- **EVT-Bench** (Person-Following)

## 📄 Citation

If you find this work useful, please consider citing:

```bibtex
@misc{chu2026abotn0technicalreportvla,
      title={ABot-N0: Technical Report on the VLA Foundation Model for Versatile Embodied Navigation}, 
      author={Zedong Chu and Shichao Xie and Xiaolong Wu and Yanfen Shen and Minghua Luo and Zhengbo Wang and Fei Liu and Xiaoxu Leng and Junjun Hu and Mingyang Yin and Jia Lu and Yingnan Guo and Kai Yang and Jiawei Han and Xu Chen and Yanqing Zhu and Yuxiang Zhao and Xin Liu and Yirong Yang and Ye He and Jiahang Wang and Yang Cai and Tianlin Zhang and Li Gao and Liu Liu and Mingchao Sun and Fan Jiang and Chiyu Wang and Zhicheng Liu and Hongyu Pan and Honglin Han and Zhining Gu and Kuan Yang and Jianfang Zhang and Di Jing and Zihao Guan and Wei Guo and Guoqing Liu and Di Yang and Xiangpo Yang and Menglin Yang and Hongguang Xing and Weiguo Li and Mu Xu},
      year={2026},
      eprint={2602.11598},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2602.11598}, 
}
```

## 📜 License

This project is released under the [Apache 2.0 License](LICENSE).

## 🙏 Acknowledgments

This work is developed by **AMAP CV Lab**. See the [Technical Report](https://arxiv.org/pdf/2602.11598) for a full list of contributors.
