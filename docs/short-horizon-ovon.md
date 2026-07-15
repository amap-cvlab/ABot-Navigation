# Short-Horizon OVON

[English](short-horizon-ovon.md) | [中文](zh-CN/short-horizon-ovon.md)

Short-Horizon OVON is a re-curated variant of the [HM3D-OVON](https://github.com/naokiyokoyama/ovon) Val-Unseen split that isolates the **recognition-and-approach phase** of object-goal navigation. Standard OVON episodes conflate "find the object" with "reach the object", confounding recognition errors with execution failures. Short-Horizon OVON addresses this by re-anchoring each episode's start pose to the first frame in which the target object is visible, providing a clean measurement of out-of-vocabulary generalization and final-meters arrival.

- **Source**: OVON Val-Unseen split (HM3D scenes)
- **Episodes**: 2,443 across 36 scenes
- **Modification**: Start poses relocated to first-visible frames via manual annotation
- **Metrics**: SR, SPL, DTG (distance-to-goal at termination)
- **Simulator**: [Habitat-sim](https://aihabitat.org/)

## Data

Download the visibility JSON that defines the re-anchored start poses:

| File | Description | Download |
|:-----|:------------|:---------|
| `short_horizon_ovon_visibility.json` | Start poses for 2,443 first-visible-frame episodes | [🤗 HuggingFace](https://huggingface.co/datasets/acvlab/ABotN-Short-Horizon-OVON) \| [🤖 ModelScope](https://www.modelscope.cn/datasets/amap_cvlab/ABotN-Short-Horizon-OVON) |

Each entry contains:

```json
{
  "scene_id": "00873-bxsVRursffK",
  "traj_id": "traj_3078",
  "image_name": "image_12_front.jpg",
  "habitat_start_position": [-6.583, 0.524, -4.307],
  "habitat_quaternion_wxyz": [0.923, 0.0, 0.384, 0.0]
}
```

You also need the standard OVON Val-Unseen episodes and HM3D scene meshes. See the [HM3D-OVON repo](https://github.com/naokiyokoyama/ovon) for setup.

## Integration

After loading the standard OVON dataset in Habitat, apply the following filter to replace episode start poses with the first-visible-frame positions. Insert this code between dataset loading and environment creation:

```python
import json
import numpy as np

def apply_short_horizon_filter(dataset, visibility_json_path):
    """Filter OVON episodes to short-horizon variants using visibility annotations.

    Replaces each episode's start pose with the first frame where the target
    object is visible, and retains only episodes present in the visibility JSON.

    Args:
        dataset: Habitat dataset with loaded OVON episodes.
        visibility_json_path: Path to the downloaded visibility JSON.

    Returns:
        The dataset with filtered and re-anchored episodes.
    """
    visibility_data = json.load(open(visibility_json_path))

    # Build lookup: scene_id + traj_id -> start pose
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

        # Subtract the robot dog's height: the visibility annotation records the
        # pose at camera height, shift down 0.45m to the robot base's ground start pose
        position[1] = position[1] - 0.45

        # Convert quaternion from wxyz to xyzw (Habitat convention)
        episode.start_rotation = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
        episode.start_position = position

        # Clear cached shortest-path data (start pose has changed)
        episode.info["geodesic_distance"] = 10000
        episode.shortest_paths = None

        filtered_episodes.append(episode)

    dataset.episodes = filtered_episodes
    dataset.episodes.sort(key=lambda ep: ep.episode_id)
    return dataset
```

Usage in your evaluation script:

```python
import habitat

config = habitat.get_config(your_config_path)
dataset = habitat.make_dataset(config.habitat.dataset.type, config=config.habitat.dataset)

# Apply short-horizon filter
dataset = apply_short_horizon_filter(dataset, "/path/to/short_horizon_ovon_visibility.json")

env = habitat.Env(config, dataset)
# ... run evaluation as usual
```

## Results

| Method | SR↑ | SPL↑ | DTG↓ |
|:-------|:---:|:----:|:----:|
| StreamVLN | 39.7 | 15.8 | 2.368 |
| NaVILA | 55.4 | 26.1 | 1.811 |
| Uni-NaVid | 68.7 | 34.5 | 1.495 |
| ABot-N1 | **84.9** | **51.8** | **0.822** |

> Full analysis in the [technical report](https://arxiv.org/abs/2607.10383), Section 6.1.2.
