# Short-Horizon OVON

[English](../short-horizon-ovon.md) | [中文](short-horizon-ovon.md)

Short-Horizon OVON 是 [HM3D-OVON](https://github.com/naokiyokoyama/ovon) Val-Unseen split 的重构变体，用于隔离物体目标导航中的**识别与接近阶段**。标准 OVON episode 将"找到物体"与"到达物体"混为一体，导致识别错误与执行失败相互混淆。Short-Horizon OVON 通过将每个 episode 的起始位姿重定位至目标物体首次可见帧来解决这一问题，提供对开放词汇泛化能力和最后几米到达精度的独立评测。

- **来源**：OVON Val-Unseen split（HM3D 场景）
- **Episode 数**：2,443 个，覆盖 36 个场景
- **改动**：起始位姿通过人工标注重定位至首次可见帧
- **指标**：SR、SPL、DTG（终止时距目标距离）
- **模拟器**：[Habitat-sim](https://aihabitat.org/)

## 数据

下载定义重定位起始位姿的可见性 JSON：

| 文件 | 说明 | 下载 |
|:-----|:-----|:-----|
| `short_horizon_ovon_visibility.json` | 2,443 个首次可见帧 episode 的起始位姿 | [🤗 HuggingFace](https://huggingface.co/datasets/acvlab/ABotN-Short-Horizon-OVON) \| [🤖 ModelScope](https://www.modelscope.cn/datasets/amap_cvlab/ABotN-Short-Horizon-OVON) |

每条记录包含：

```json
{
  "scene_id": "00873-bxsVRursffK",
  "traj_id": "traj_3078",
  "image_name": "image_12_front.jpg",
  "habitat_start_position": [-6.583, 0.524, -4.307],
  "habitat_quaternion_wxyz": [0.923, 0.0, 0.384, 0.0]
}
```

还需要标准 OVON Val-Unseen episode 和 HM3D 场景网格，参见 [HM3D-OVON 仓库](https://github.com/naokiyokoyama/ovon)。

## 接入方式

在 Habitat 中加载标准 OVON 数据集后，插入以下过滤代码替换 episode 起始位姿。在数据集加载与环境创建之间插入：

```python
import json
import numpy as np

def apply_short_horizon_filter(dataset, visibility_json_path):
    """将 OVON episode 过滤为 short-horizon 变体。

    用可见性标注中的首次可见帧位姿替换每个 episode 的起始位姿，
    仅保留可见性 JSON 中存在的 episode。

    Args:
        dataset: 已加载 OVON episode 的 Habitat 数据集。
        visibility_json_path: 下载的可见性 JSON 文件路径。

    Returns:
        过滤并重定位后的数据集。
    """
    visibility_data = json.load(open(visibility_json_path))

    # 构建查找表：scene_id + traj_id -> 起始位姿
    pose_lookup = {}
    for entry in visibility_data:
        key = entry["scene_id"] + "_" + entry["traj_id"]
        pose_lookup[key] = {
            "position": entry["habitat_start_position"],
            "quaternion_wxyz": entry["habitat_quaternion_wxyz"],
        }

    filtered_episodes = []
    for episode in dataset.episodes:
        scene_id = episode.scene_id.split("/")[-2]
        key = scene_id + "_traj_" + episode.episode_id
        if key not in pose_lookup:
            continue

        pose = pose_lookup[key]
        position = pose["position"]
        quat_wxyz = pose["quaternion_wxyz"]

        # 减去机器狗自身高度：可见性标注记录的是相机高度处的位姿，
        # 需下移 0.45m 转换为机器人基座的落地起始位姿
        position[1] = position[1] - 0.45

        # 四元数从 wxyz 转为 xyzw（Habitat 约定）
        episode.start_rotation = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        episode.start_position = position

        # 清除缓存的最短路径数据（起始位姿已变更）
        episode.info["geodesic_distance"] = 10000
        episode.shortest_paths = None

        filtered_episodes.append(episode)

    dataset.episodes = filtered_episodes
    dataset.episodes.sort(key=lambda ep: ep.episode_id)
    return dataset
```

在评测脚本中使用：

```python
import habitat

config = habitat.get_config(your_config_path)
dataset = habitat.make_dataset(config.habitat.dataset.type, config=config.habitat.dataset)

# 应用 short-horizon 过滤
dataset = apply_short_horizon_filter(dataset, "/path/to/short_horizon_ovon_visibility.json")

env = habitat.Env(config, dataset)
# ... 正常运行评测
```

## 结果

| 方法 | SR↑ | SPL↑ | DTG↓ |
|:-----|:---:|:----:|:----:|
| StreamVLN | 39.7 | 15.8 | 2.368 |
| NaVILA | 55.4 | 26.1 | 1.811 |
| Uni-NaVid | 68.7 | 34.5 | 1.495 |
| ABot-N1 | **84.9** | **51.8** | **0.822** |

> 完整分析见[技术报告](https://arxiv.org/abs/2607.10383) Section 6.1.2。
