# API 参考

[English](../api-reference.md) | [中文](api-reference.md)

`abotn_evaluator` 包的完整接口规范。

---

## 评测循环

评测器按黑盒循环驱动每个 episode：

```
agent.reset()
for step in range(max_steps):
    obs = evaluator.build_observation()
    pred = agent.predict(obs)
    evaluator.step(pred)
    if pred.arrive or step + 1 == max_steps:
        break
evaluator.save_result()
```

- 评测器仅调用 `reset()` 和 `predict()`
- Agent 内部架构（记忆、规划等）对评测器不可见
- `arrive=True` 时，评测器检查实际距离是否 ≤ `arrive_threshold`

---

## Agent 接口

### Point-Goal

```python
from abotn_evaluator.interface.point_goal import BasePointGoalAgent, Observation, WaypointPrediction

class YourAgent(BasePointGoalAgent):
    def reset(self) -> None: ...
    def predict(self, observation: Observation) -> WaypointPrediction: ...
```

### POI-Goal

```python
from abotn_evaluator.interface.poi_goal import BasePoiGoalAgent, PoiGoalObservation
from abotn_evaluator.interface.point_goal import WaypointPrediction

class YourPoiAgent(BasePoiGoalAgent):
    def reset(self) -> None: ...
    def predict(self, observation: PoiGoalObservation) -> WaypointPrediction: ...
```

`PoiGoalObservation` 继承 `Observation` 全部字段，额外增加 `poi_name: str`。

---

## Observation 字段

| 字段 | 类型 | 必填 | 格式 |
|:-----|:-----|:----:|:-----|
| `images` | `Dict[str, ndarray]` | 是 | 三视角 RGB，键：`"left"`、`"front"`、`"right"`，shape `(640, 720, 3)`，dtype `uint8`，RGB。视角分别为 -90°、0°、+90°。 |
| `target_position` | `ndarray` | 是 | Agent 局部坐标系 `[front, left]`，单位米 |
| `distance_to_goal` | `float` | 是 | 到目标的 XY 欧氏距离（米） |
| `position` | `ndarray` | 是 | Agent 世界坐标 `[x, y, z]`（米） |
| `rotation` | `ndarray` | 是 | Camera-to-world 4×4 齐次变换矩阵 |
| `heading` | `float` | 是 | Agent 偏航角 yaw（弧度） |
| `step_count` | `int` | 是 | 当前 episode 已执行步数（从 0 起） |
| `goal_world` | `ndarray` | 否 | 目标点的世界坐标 `[x, y]`（米）。仅 Point-Goal 提供；POI-Goal 中为 `None`。 |
| `history_images` | `List[Dict[str, ndarray]]` | 否 | 历史帧图像（需 `provide_history=True`） |
| `history_poses` | `List[ndarray]` | 否 | 历史帧 4×4 pose（需 `provide_history=True`） |
| `occ_map` | `ndarray` | 否 | 占用栅格图（需 `provide_occ_map=True`） |
| `height_map` | `ndarray` | 否 | 高度图（需 `provide_height_map=True`） |
| `meta_data` | `dict` | 否 | 场景元数据（坐标转换参数等） |
| `extra` | `dict` | 是 | 扩展字段，默认空 |

**POI-Goal 额外字段：**

| 字段 | 类型 | 说明 |
|:-----|:-----|:-----|
| `poi_name` | `str` | 目标 POI 名称（如 `"星巴克"`） |

## WaypointPrediction 字段

| 字段 | 类型 | 必填 | 格式 |
|:-----|:-----|:----:|:-----|
| `waypoint` | `ndarray` | 是 | Agent 局部坐标系 `[front, left]`（米）。接受 shape `(2,)`、`(1,2)` 或 `(N,2)`。 |
| `arrive` | `bool` | 是 | 是否宣称到达。`distance_to_goal < arrive_threshold` 时设为 `True`。 |
| `directions` | `ndarray` | 否 | 方向向量，shape `(N,2)`。未提供时从 waypoint 自动计算。 |
| `confidence` | `float` | 否 | 置信度 `[0,1]`，仅记录日志，不影响评测。 |
| `extra` | `dict` | 否 | 调试/可视化数据（见[可视化扩展](#可视化扩展)）。 |

---

## 坐标系

### Agent 局部坐标系（`target_position` 和 `waypoint` 使用）

- 原点：Agent 当前位置
- 第 0 维（`front`）：正前方为正
- 第 1 维（`left`）：左侧为正

```
     front (+)
        |
        |
  <-----o
left (+) agent
```

右手系，单位米。

### 世界坐标系（`position` 和 `rotation` 使用）

场景固定的全局坐标系。`rotation` 矩阵将局部向量变换至世界：

```
world_point = rotation[:3, :3] @ local_point + rotation[:3, 3]
```

只需确保 `waypoint` 在局部坐标系、单位米即可——评测器内部完成坐标转换。

> 如模型输出的坐标系不同（如 `[right, forward]` 或极坐标 `(r, theta)`），在 `predict()` 中转换。见[自定义 Agent 接入](custom-agents.md)。

---

## 配置参数

### Point-Goal：`make_eval_config`

```python
from abotn_evaluator.point_goal.evaluator import make_eval_config

config = make_eval_config(
    mode="outdoor",  # 或 "indoor"
    render_url="http://localhost:7036/render_gs",
    max_steps=100,
)
```

协议参数由 `mode` 自动设定：

| Mode | `arrive_threshold` | `collision_threshold` | `occ_dilation_meters` |
|:-----|:-------------------|:----------------------|:----------------------|
| `outdoor` | 0.5 | 3 | 0.5 |
| `indoor` | 0.5 | 1 | 0.2 |

运行参数（通过 `**kwargs` 传递）：

| 参数 | 默认值 | 说明 |
|:-----|:-------|:-----|
| `max_steps` | 100 | 每 episode 最大步数 |
| `save_render_images` | `True` | 保存渲染图像至磁盘 |
| `provide_history` | `False` | 在观测中包含 `history_images` / `history_poses` |
| `provide_occ_map` | `False` | 在观测中包含 `occ_map` |
| `provide_height_map` | `False` | 在观测中包含 `height_map` |
| `enable_visualization` | `False` | 处理 `WaypointPrediction.extra` 进行可视化 |

消融研究可显式覆盖协议参数：`make_eval_config(mode="outdoor", collision_threshold=5)`。

### POI-Goal：`PoiGoalEvalConfig`

```python
from abotn_evaluator.poi_goal.evaluator import PoiGoalEvalConfig

config = PoiGoalEvalConfig(
    render_url="http://localhost:7036/render_gs",
    max_steps=100,
)
```

单一协议——默认值即标准：

| 参数 | 默认值 | 说明 |
|:-----|:-------|:-----|
| `arrive_threshold` | 2.0 | 到达距离（米） |
| `collision_mode` | `"hard"` | `"off"` / `"soft"` / `"hard"` |
| `occ_dilation_meters` | 0.5 | 自由空间膨胀（米） |
| `robot_radius` | 0.0 | 机器人足迹半径（米） |
| `occ_obstacle_polarity` | `"dark"` | `"dark"` = 深色像素为障碍 |
| `occ_dark_threshold` | 64 | 灰度障碍阈值 |

`collision_mode` 取值：
- `"off"`：不进行碰撞检测
- `"soft"`：统计碰撞次数，不影响成功判定和终止
- `"hard"`：首次非豁免碰撞即终止（目标圈内碰撞豁免）

---

## CLI 参数

| 参数 | Point-Goal | POI-Goal | 说明 |
|:-----|:-----------|:---------|:-----|
| `--agent-module` | 是 | 是 | Agent 类，格式 `package.module:ClassName` |
| `--agent-config` | 是 | 是 | YAML 配置（键值作为 `**kwargs` 传入 `__init__`） |
| `--data-dir` | 是 | 是 | 轨迹数据目录 |
| `--map-dir` | 是 | 是 | 占用栅格图目录 |
| `--render-url` | 是 | 是 | 渲染服务地址 |
| `--output-dir` | 是 | 是 | 输出目录 |
| `--max-steps` | 100 | 100 | 每 episode 最大步数 |
| `--mode` | `outdoor` / `indoor` | `indoor` | 选择协议参数（Point-Goal）或指标分组（POI-Goal） |
| `--arrive-threshold` | 自动 | 2.0 | 覆盖到达距离 |
| `--collision-threshold` | 自动 | 3 | 碰撞次数阈值 |
| `--collision-mode` | -- | `hard` | POI-Goal 碰撞模式 |
| `--occ-dilation-meters` | 自动 | 0.5 | 覆盖自由空间膨胀 |
| `--provide-history` | Flag | Flag | 启用历史观测 |
| `--save-render-images` | Flag | Flag | 保存图像（默认开启） |
| `--enable-visualization` | Flag | Flag | 启用 extra 字段可视化 |
| `--resume-dir` | 是 | 是 | 从上次运行目录续跑 |
| `--skip-metrics` | Flag | Flag | 跳过评测后指标分析 |

Point-Goal 未显式传递的协议参数由 `--mode` 自动选择。

---

## GaussianScene

场景管理器，从本地目录树加载 episode，并提供坐标转换、碰撞检测和迭代工具。

### 导入与构造

```python
from abotn_evaluator.scene import GaussianScene

scene = GaussianScene(
    local_data_path="/path/to/ABotN-PointBench/Outdoor/annotations",
    local_map_path="/path/to/ABotN-PointBench/Outdoor/occmaps",   # 可选
)
```

### 构造函数参数

| 参数 | 类型 | 必填 | 说明 |
|:-----|:-----|:----:|:-----|
| `local_data_path` | `str` | 是 | 根目录，包含各场景子目录。每个子目录包含 `traj_*.json` 文件及可选的 `map/` 文件夹。 |
| `local_map_path` | `str` | 否 | 独立的地图数据目录。提供时，占用栅格图从 `{local_map_path}/{episode_id}/` 加载，而非 episode 自身的 `scene_path`。适用于地图数据与轨迹数据分离存储的场景。 |

构造时，`GaussianScene` 自动扫描 `local_data_path` 下的子目录，为每个子目录创建 `Episode`，并加载所有轨迹任务和地图数据。

### Episode 数据类

每个场景目录对应一个 `Episode` 实例。

```python
from abotn_evaluator.scene import Episode

episode = Episode.from_scene_dir("/path/to/scene_001")
```

| 字段 | 类型 | 说明 |
|:-----|:-----|:-----|
| `episode_id` | `str` | 唯一场景标识符（目录名）。 |
| `scene_path` | `str` | 本地场景目录的绝对路径。 |
| `tasks` | `List[Task]` | 从 `traj_*.json` 文件加载的导航任务。 |
| `occ_map` | `ndarray` 或 `None` | 占用栅格图——灰度 `uint8`。像素值 `>=128` 为**自由空间**；`<128` 为**障碍物**。从 `map/occ_map.png` 加载。 |
| `height_map` | `ndarray` 或 `None` | 每像素地面高度（米）。从 `map/occ_map_height.tiff` 加载。 |
| `meta_data` | `dict` 或 `None` | 从 `map/occ_map_meta.txt` 解析的坐标系元数据。键包括 `TOP_LEFT_X`、`TOP_LEFT_Y`、`IMAGE_WIDTH`、`IMAGE_HEIGHT`、`COORDINATE_RANGE_X`、`COORDINATE_RANGE_Y` 等。 |

**期望的目录结构：**

```
{scene_dir}/
    traj_0.json
    traj_1.json
    ...
    map/
        occ_map.png
        occ_map_meta.txt
        occ_map_height.tiff
```

**便捷方法：**

| 方法 / 属性 | 返回类型 | 说明 |
|:------------|:---------|:-----|
| `has_map_data()` | `bool` | 若 `occ_map`、`meta_data`、`height_map` 均已加载则返回 `True`。 |
| `num_tasks` | `int` | 此 episode 的任务数。 |
| `get_meter_per_pixel()` | `float` 或 `None` | 地图平均分辨率（米/像素），根据元数据计算。 |

### Task 数据类

每个 `traj_*.json` 文件对应一个 `Task` 实例。

```python
from abotn_evaluator.scene import Task

task = Task.from_json("/path/to/scene_001/traj_0.json")
```

| 字段 / 属性 | 类型 | 说明 |
|:------------|:-----|:-----|
| `task_id` | `str` | 唯一标识符——JSON 文件名主干（如 `"traj_0"`）。 |
| `trajectory` | `List[Dict[str, float]]` | 路径点序列。每个 dict 至少包含 `x`、`y`、`z`、`roll`、`pitch`、`yaw`（欧拉角，弧度制）。 |
| `label` | `dict` | 原始轨迹 JSON 中的元数据标签。 |
| `start_pose` | `dict` 或 `None` | 属性。`trajectory` 第一个元素，为空时返回 `None`。 |
| `end_pose` | `dict` 或 `None` | 属性。`trajectory` 最后一个元素，为空时返回 `None`。 |
| `goal_label` | `str` | 属性。导航目标的语义标签，从 `label["extend"]["goal_label"]` 提取。不存在时返回 `""`。 |

### 碰撞检测方法

所有碰撞检测方法均为 `GaussianScene` 的类方法。

#### `check_point_collision_status`

```python
is_colliding = GaussianScene.check_point_collision_status(
    pose,                   # ndarray (4x4) 或包含 "x", "y" 的 dict
    episode,                # 包含 occ_map 和 meta_data 的 Episode
    robot_radius_pixel=2,   # 方形足迹半尺寸（像素）
)
```

检查单个位姿是否落在占用栅格图的障碍物上。当 `robot_radius_pixel > 0` 时，检查该点周围的方形区域；区域内任何障碍像素即触发碰撞。超出地图边界的点视为碰撞。

**返回值：** `bool` -- 碰撞时为 `True`。

#### `check_line_collision_fast`

```python
has_collision = GaussianScene.check_line_collision_fast(
    pose_start,             # ndarray (4x4) 或包含 "x", "y" 的 dict
    pose_end,               # ndarray (4x4) 或包含 "x", "y" 的 dict
    episode,                # 包含 occ_map 和 meta_data 的 Episode
    step_size_pixel=2.0,    # 像素空间采样间隔
)
```

检查两个位姿之间的直线路径是否穿过障碍物。在像素空间中沿直线均匀采样。超出地图边界的点视为碰撞。

**返回值：** `bool` -- 直线上任一采样点碰撞时为 `True`。

#### `compute_line_collision_length`

```python
collision_meters = GaussianScene.compute_line_collision_length(
    pose_start,               # ndarray (4x4) 或包含 "x", "y" 的 dict
    pose_end,                 # ndarray (4x4) 或包含 "x", "y" 的 dict
    episode,                  # 包含 occ_map 和 meta_data 的 Episode
    sample_step_meter=0.05,   # 世界坐标系采样间隔（米）
)
```

计算路径段中位于障碍物内部的物理长度（米）。在世界坐标系中密集采样，然后将每个采样点映射到占用栅格图检查碰撞状态。超出边界的点视为障碍物。连续障碍段的长度被累加为总碰撞长度。

**返回值：** `float` -- 总碰撞长度（米）。

### 坐标转换

两个方法均为 `GaussianScene` 的静态方法。

#### `convert_actual_to_pixel`

```python
x_pixel, y_pixel = GaussianScene.convert_actual_to_pixel(
    x_actual,   # float，世界 X 坐标
    y_actual,   # float，世界 Y 坐标
    episode,    # 包含 meta_data 的 Episode
)
```

将世界坐标转换为占用栅格图像素坐标。使用 `meta_data` 中的字段（`TOP_LEFT_X`、`TOP_LEFT_Y`、`IMAGE_WIDTH`、`IMAGE_HEIGHT`、`COORDINATE_RANGE_X`、`COORDINATE_RANGE_Y`），并根据角点坐标自动处理坐标轴方向。

**返回值：** `(int, int)` -- 像素坐标 `(x_pixel, y_pixel)`。

#### `convert_pixel_to_actual`

```python
x_actual, y_actual = GaussianScene.convert_pixel_to_actual(
    x_pixel,    # int，像素 X 坐标
    y_pixel,    # int，像素 Y 坐标
    episode,    # 包含 meta_data 的 Episode
)
```

`convert_actual_to_pixel` 的逆变换。将占用栅格图像素坐标转换回世界坐标。

**返回值：** `(float, float)` -- 世界坐标 `(x_actual, y_actual)`。

### 迭代

```python
# 迭代 episode
for episode in scene:
    print(episode.episode_id, episode.num_tasks)

# 迭代所有 (episode, task) 对
for episode, task in scene.iter_tasks():
    print(episode.episode_id, task.task_id)

# 总任务数
print(scene.total_tasks)
```

| 方法 / 属性 | 返回类型 | 说明 |
|:------------|:---------|:-----|
| `iter_tasks()` | `Generator[(Episode, Task)]` | 遍历所有 episode 的全部 `(episode, task)` 对。 |
| `total_tasks` | `int` | 所有 episode 的总任务数。 |
| `__len__()` | `int` | Episode 数量。 |
| `__iter__()` | `Iterator[Episode]` | 迭代 episode。 |
| `get_episode(idx)` | `Episode` | 按整数索引获取 episode。 |
| `get_episode_by_id(episode_id)` | `Episode` 或 `None` | 按 ID 字符串查找 episode。 |
| `summary()` | `dict` | 返回 `{"num_episodes": ..., "total_tasks": ..., "episode_ids": [...]}`。 |

### 地图操作

#### `dilate_free_space_by_meter`

```python
episode.dilate_free_space_by_meter(dilation_meters=0.5)
```

按物理距离膨胀占用栅格图的自由空间区域。这有效地缩小了障碍物，为标注误差提供容差。内部通过场景元数据（`get_meter_per_pixel()`）将米转换为像素，然后使用椭圆形核进行形态学膨胀。

如果元数据不可用或 `dilation_meters <= 0`，操作将被跳过并输出警告。

另有一个底层方法 `dilate_free_space(dilation_pixels)`，直接以像素为单位操作。

### 位姿转换

```python
pose_matrix = GaussianScene.get_gaussian_pose(cur_info)
```

将位姿字典（键为 `x`、`y`、`z`、`roll`、`pitch`、`yaw`，弧度制）转换为 4x4 camera-to-world 齐次变换矩阵，使用 XYZ 欧拉角约定。

---

## GaussianRenderer

远程高斯泼溅渲染服务的客户端。通过 HTTP 发送渲染请求并返回 RGB 图像。

### 导入与构造

```python
from abotn_evaluator.render_client import GaussianRenderer, CameraConfig

renderer = GaussianRenderer(
    render_url="http://localhost:7036/render_gs",
    camera_config=CameraConfig(),   # 可选，使用默认值
    num_views=3,                    # 3 = 左/右/前，1 = 仅前方
    timeout=30,                     # HTTP 超时（秒）
    max_retries=3,                  # 失败重试次数
    retry_backoff=1.0,              # 重试间隔基础时间
)
```

### CameraConfig 数据类

针孔相机内参与高度偏移。

```python
from abotn_evaluator.render_client import CameraConfig

cam = CameraConfig(
    width=720,
    height=640,
    fx=252.075,
    fy=252.075,
    cx=360.0,
    cy=320.0,
    extrinsic_height=0.65,
)
```

| 字段 | 类型 | 默认值 | 说明 |
|:-----|:-----|:-------|:-----|
| `width` | `int` | `720` | 图像宽度（像素）。 |
| `height` | `int` | `640` | 图像高度（像素）。 |
| `fx` | `float` | `252.075` | x 轴焦距（像素）。 |
| `fy` | `float` | `252.075` | y 轴焦距（像素）。 |
| `cx` | `float` | `360.0` | 主点 x 坐标（像素）。 |
| `cy` | `float` | `320.0` | 主点 y 坐标（像素）。 |
| `extrinsic_height` | `float` | `0.65` | 相机相对地面的高度偏移（米）。正值表示在地面之上。 |

**属性：** `intrinsics_colmap` -- 返回 COLMAP 格式的相机内参字符串：`"1 PINHOLE <width> <height> <fx> <fy> <cx> <cy>"`。

### 构造函数参数

| 参数 | 类型 | 默认值 | 说明 |
|:-----|:-----|:-------|:-----|
| `render_url` | `str` | （必填） | 渲染端点的完整 URL（如 `"http://host:7036/render_gs"`）。 |
| `camera_config` | `CameraConfig` | `CameraConfig()` | 相机内参/外参配置。 |
| `num_views` | `int` | `3` | 每个位姿渲染的视角数。`3` 表示左/右/前，`1` 表示仅前方。 |
| `timeout` | `int` | `30` | HTTP 请求超时时间（秒）。 |
| `max_retries` | `int` | `3` | 渲染失败时的最大重试次数。 |
| `retry_backoff` | `float` | `1.0` | 重试间隔基础时间（秒）。每次重试后翻倍（指数退避）。 |

### `render_at_pose`

```python
images = renderer.render_at_pose(
    pose,           # ndarray，4x4 camera-to-world 变换矩阵
    scene_id,       # str，待渲染场景的标识符
    save_dir=None,  # 可选 str，图像保存目录
    image_id=0,     # int，保存文件名的数字前缀
)
```

在给定的 4x4 位姿矩阵处渲染图像。对于多视角设置（`num_views=3`），渲染器通过对基础位姿添加固定的偏航角偏移生成三个视角。

**视角顺序与偏航角偏移（`num_views=3` 时）：**

| 索引 | 视角名称 | 偏航角偏移 |
|:-----|:---------|:-----------|
| 0 | `left` | +90 度 |
| 1 | `right` | -90 度 |
| 2 | `front` | 0 度 |

提供 `save_dir` 时，图像保存为 `{image_id}_{view_name}.jpg`（如 `0_left.jpg`、`0_right.jpg`、`0_front.jpg`）。

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|:-----|:-----|:-------|:-----|
| `pose` | `ndarray` | （必填） | 4x4 camera-to-world 变换矩阵。 |
| `scene_id` | `str` | （必填） | 传递给渲染服务的场景标识符。 |
| `save_dir` | `str` | `None` | 若提供，渲染图像保存到此目录。 |
| `image_id` | `int` | `0` | 保存文件名的数字前缀。 |
| `save_img_idx` | `List[int]` | `None` | 若设置，仅索引在此列表中的图像被保存。 |

**返回值：** `List[PIL.Image.Image]` -- 渲染的 PIL 图像列表。

### 重试机制

当渲染请求失败（HTTP 错误或异常）时，渲染器以指数退避策略重试：

1. 第一次重试：等待 `retry_backoff` 秒（默认 1.0 秒）
2. 第二次重试：等待 `retry_backoff * 2` 秒（默认 2.0 秒）
3. 第三次重试：等待 `retry_backoff * 4` 秒（默认 4.0 秒）

所有重试耗尽后，`render_at_pose` 抛出 `RenderFailureError`。

```python
from abotn_evaluator.render_client import RenderFailureError

try:
    images = renderer.render_at_pose(pose, scene_id)
except RenderFailureError as e:
    print(f"所有重试后渲染失败：{e}")
```

---

## ShortMemory

滑动窗口记忆缓冲区，管理多视角 RGB 观测与 Agent 位姿。维护固定容量的历史帧，自动缩小历史图像以节省内存。

### 导入与构造

```python
from abotn_evaluator.memory import ShortMemory

memory = ShortMemory(
    max_history_frames=20,
    num_current_views=3,
    input_img_size=(476, 420),
    resize_ratio=0.25,
    reorder_views=True,
    resize_on_add=True,
)
```

### 构造函数参数

| 参数 | 类型 | 默认值 | 说明 |
|:-----|:-----|:-------|:-----|
| `max_history_frames` | `int` | `20` | 保留的最大**历史**帧数（不含当前帧）。 |
| `num_current_views` | `int` | `3` | 每帧的相机视角数（如 3 表示左/前/右）。 |
| `input_img_size` | `tuple(int, int)` | `(476, 420)` | 输入图像的 `(宽, 高)`。用作缩放历史帧时的参考尺寸。 |
| `resize_ratio` | `float` | `0.25` | 历史帧图像的缩放因子。`0.25` 表示图像缩小为原始尺寸的 25%。 |
| `reorder_views` | `bool` | `True` | 若为 `True`，将输入视角从内部顺序 `[left, right, front]` 重排为外部顺序 `[left, front, right]`。 |
| `resize_on_add` | `bool` | `True` | 若为 `True`，新帧添加时自动缩放刚变为历史的帧。外部代码自行处理缩放时设为 `False`。 |

### `add_frame`

```python
frame_index = memory.add_frame(
    rgbs,                    # ndarray 或 PIL.Image 的列表
    pose,                    # ndarray，camera-to-world 变换
    have_memory=False,       # 驱逐策略
    num_current_image=None,  # 可选覆盖 num_current_views
)
```

向缓冲区追加新的多视角观测帧。每次调用时，方法执行：

1. 将输入图像转换为 PIL 格式，并可选地重排视角顺序。
2. **折叠上一个当前帧**为仅前方视角——侧面视角（左、右）被丢弃，使历史仅保留前方视角。
3. 追加新帧的图像和位姿。
4. **缩放**刚变为历史的帧（若 `resize_on_add=True`）。
5. 若缓冲区超过 `max_history_frames`，**驱逐**一帧。

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|:-----|:-----|:-------|:-----|
| `rgbs` | `List` | （必填） | 相机视角图像。每个元素为 NumPy `uint8` HWC 数组或 PIL Image。期望长度等于 `num_current_views`。 |
| `pose` | `ndarray` | （必填） | 与此帧关联的 camera-to-world 变换。 |
| `have_memory` | `bool` | `False` | 窗口满时的驱逐策略（见下文）。 |
| `num_current_image` | `int` | `None` | 覆盖本次调用的 `num_current_views`。 |

**返回值：** `int` -- 分配给此观测的从零开始的帧索引。

### 驱逐策略

当缓冲区超出容量时，需移除一个历史帧：

- **FIFO**（`have_memory=True`）：丢弃最旧的帧。简单且可预测。
- **自适应**（`have_memory=False`）：丢弃与前一帧时间间隔最小的历史帧。这保持了时间分布的均匀性，使保留的帧在时间上更加均匀分布。

### 视角重排

当 `reorder_views=True`（默认）时，内部输入顺序 `[left, right, front]`（索引 0, 1, 2）被重排为外部顺序 `[left, front, right]`（索引 0, 2, 1）。这确保了下游消费者获得一致的左-前-右排序。

### 访问器

| 方法 | 返回类型 | 说明 |
|:-----|:---------|:-----|
| `get_all_images()` | `List[PIL.Image]` | 所有存储的图像（历史前方视角 + 当前多视角）。 |
| `get_current_images()` | `List[PIL.Image]` | 仅当前帧的多视角图像（通常 3 个：左、前、右）。 |
| `get_history_images()` | `List[PIL.Image]` | 仅历史（缩小的、仅前方）图像，不含当前帧。 |
| `get_all_poses()` | `List[ndarray]` | 所有存储的位姿（每个图像条目对应一个）。 |
| `get_history_poses()` | `List[ndarray]` | 仅历史帧对应的位姿。 |
| `get_last_pose()` | `ndarray` 或 `None` | 最近添加的位姿，缓冲区为空时返回 `None`。 |
| `get_image_indices()` | `List[int]` | 每个存储图像条目的帧索引标签。用于跟踪时间间隔。 |
| `reset()` | `None` | 清除所有存储的图像、位姿和计数器。 |
| `reset_history()` | `None` | 清除历史，仅保留当前帧的视角。 |

**属性：**

| 属性 | 类型 | 说明 |
|:-----|:-----|:-----|
| `frame_count` | `int` | 已添加的总帧数（包括被驱逐的帧）。 |
| `__len__()` | `int` | 当前存储的图像条目总数。 |

---

## analyze_and_report

完整的指标分析管线函数。**可在评测过程中随时调用**——它从所有已完成的 `result.json` 文件中聚合指标，适用于监控中间结果。

### Point-Goal 用法

```python
from abotn_evaluator.point_goal.metrics import analyze_and_report

report = analyze_and_report(
    result_dir="./results/outdoor/run1",
    mode="outdoor",
)
```

### POI-Goal 用法

```python
from abotn_evaluator.poi_goal.metrics import analyze_and_report

report = analyze_and_report(
    result_dir="./results/poi/run1",
    mode="indoor",
    arrive_threshold=2.0,
)
```

### Point-Goal 参数

| 参数 | 类型 | 默认值 | 说明 |
|:-----|:-----|:-------|:-----|
| `result_dir` | `str` | （必填） | 评测输出根目录。递归加载此目录下的所有 `result.json` 文件。 |
| `mode` | `str` | `"outdoor"` | `"outdoor"` 或 `"indoor"`。控制难度分类（outdoor 为 short/medium/long；indoor 为 easy/hard）。 |
| `collision_threshold` | `int` | `3` | SR_NEW 的碰撞次数阈值。任务的 SR_NEW 为 `True` 当且仅当成功且 `path_collision_count < collision_threshold`。 |
| `min_distance` | `float` | `5.0` | outdoor 难度分组的最短路径距离下界。 |
| `max_distance` | `float` | `50.0` | outdoor 难度分组的最短路径距离上界。 |
| `exclude_scenes` | `List[str]` | `None` | 从分析中排除的场景 ID。outdoor 模式默认为 `["park3"]`，indoor 模式默认为空。 |
| `output_path` | `str` | `None` | 输出 JSON 文件路径。默认为 `{result_dir}/eval_summary.json`。 |
| `per_scene` | `bool` | `False` | 若为 `True`，打印逐场景指标明细表。 |

### POI-Goal 参数

| 参数 | 类型 | 默认值 | 说明 |
|:-----|:-----|:-------|:-----|
| `result_dir` | `str` | （必填） | 评测输出根目录。 |
| `mode` | `str` | `"indoor"` | `"outdoor"` 或 `"indoor"`。 |
| `collision_threshold` | `int` | `3` | 碰撞次数阈值（保留用于向后兼容）。 |
| `arrive_threshold` | `float` | `2.0` | 评测使用的到达距离阈值。用于计算调整后的 SPL 公式：`effective_shortest = max(shortest - arrive_threshold, 1e-3)`。 |
| `min_distance` | `float` | `5.0` | outdoor 难度分组的最小距离。 |
| `max_distance` | `float` | `50.0` | outdoor 难度分组的最大距离。 |
| `exclude_scenes` | `List[str]` | `None` | 排除的场景 ID。 |
| `output_path` | `str` | `None` | 输出 JSON 路径。默认为 `{result_dir}/poi_goal_analysis.json`。 |
| `per_scene` | `bool` | `False` | 打印逐场景明细。 |
| `per_poi` | `bool` | `True` | 打印逐 POI 明细表。 |

### CLI 用法

```bash
# Point-Goal 指标（可在评测进行中运行）
python -m abotn_evaluator.point_goal.metrics \
    --result-dir ./results/outdoor/run1 \
    --mode outdoor \
    --collision-threshold 3

# POI-Goal 指标
python -m abotn_evaluator.poi_goal.metrics \
    --result-dir ./results/poi/run1 \
    --mode indoor \
    --arrive-threshold 2.0 \
    --per-poi
```

### 返回值结构

函数返回字典（同时保存为 JSON），结构如下：

**Point-Goal（`eval_summary.json`）：**

```json
{
  "task_type": "point_goal",
  "result_dir": "/absolute/path/to/results",
  "mode": "outdoor",
  "collision_threshold": 3,
  "total_count": 150,
  "status_distribution": {
    "stop": 120,
    "max_steps": 25,
    "collision": 5
  },
  "distance_config": {
    "min_distance": 5.0,
    "max_distance": 50.0,
    "step": 15.0
  },
  "overall": {
    "count": 150,
    "success_rate": 0.80,
    "sr_new": 0.75,
    "spl_new": 0.68,
    "tcr_new": 0.72,
    "dcr_new": 0.85,
    "avg_steps": 45.2,
    "...": "..."
  },
  "groups": {
    "short":  { "count": 50, "sr_new": 0.90, "...": "..." },
    "medium": { "count": 55, "sr_new": 0.75, "...": "..." },
    "long":   { "count": 45, "sr_new": 0.58, "...": "..." }
  }
}
```

**POI-Goal（`poi_goal_analysis.json`）：**

```json
{
  "task_type": "poi_goal",
  "result_dir": "/absolute/path/to/results",
  "mode": "indoor",
  "arrive_threshold": 2.0,
  "total_count": 100,
  "overall": {
    "count": 100,
    "success_rate": 0.72,
    "spl": 0.65,
    "collision": { "tasks_evaluated": 100, "...": "..." },
    "poi_stats": {
      "Starbucks": { "total": 10, "success": 8, "success_rate": 0.80, "avg_spl": 0.71, "...": "..." },
      "...": "..."
    },
    "...": "..."
  },
  "groups": { "easy": { "...": "..." }, "hard": { "...": "..." } }
}
```

### 难度分类

**Outdoor 模式**根据 `shortest_path_length` 将任务分为 `short`、`medium`、`long` 三组：

| 分组 | 范围 |
|:-----|:-----|
| `short` | `[min_distance, min_distance + step)` |
| `medium` | `[min_distance + step, min_distance + 2*step)` |
| `long` | `[min_distance + 2*step, max_distance * 1.5]` |

其中 `step = (max_distance - min_distance) / 3`。

**Indoor 模式**根据预定义的场景 ID 列表将任务分为 `easy` 和 `hard` 两组。

---

## Agent 加载

动态类加载工具，用于在运行时导入 Agent 类。

### 导入与用法

```python
from abotn_evaluator.agent_loader import load_class

AgentClass = load_class("your_package.module:YourAgent")
agent = AgentClass(**config)
```

### 格式

`module_path` 字符串必须使用 `"package.module:ClassName"` 格式，用恰好一个冒号（`:`）分隔可导入的模块路径和类名。

**示例：**

```python
# 从已安装的包加载
Agent = load_class("agent_examples.point_goal_random:RandomPointGoalAgent")

# 从嵌套包加载
Agent = load_class("my_agents.vlm.qwen_agent:QwenPointGoalAgent")
```

### 错误处理

| 异常 | 条件 | 示例 |
|:-----|:-----|:-----|
| `ValueError` | 模块路径不包含恰好一个 `:`，或模块名/类名为空。 | `load_class("no_colon_here")`、`load_class("a:b:c")`、`load_class(":MyClass")` |
| `ImportError` | 模块无法导入。错误信息包含原始导入错误以便调试。 | `load_class("nonexistent_package.module:Agent")` |
| `AttributeError` | 在模块中找不到指定的类名。错误信息列出模块中所有可用的类。 | `load_class("agent_examples.point_goal_random:NonexistentClass")` |
| `TypeError` | 解析出的属性存在但不是类（如函数或变量）。 | `load_class("os.path:join")` |

---

## 配置

基于 YAML 的配置加载，支持继承和模式解析。

### 导入与用法

```python
from abotn_evaluator.config import load_agent_config

config = load_agent_config("agent_config.yaml")
```

### `load_agent_config`

加载 Agent 配置 YAML 文件，支持两个特殊功能：

#### 1. `_base_` 继承

子 YAML 可通过 `_base_` 键引用基础 YAML 文件。基础文件先被加载，然后子文件的值深度合并覆盖其上（子值覆盖基础值）。

**base_config.yaml：**
```yaml
model_name: "qwen-vl"
temperature: 0.7
max_tokens: 2048
```

**agent_config.yaml：**
```yaml
_base_: base_config.yaml
temperature: 0.5
custom_param: true
```

**结果：** `{"model_name": "qwen-vl", "temperature": 0.5, "max_tokens": 2048, "custom_param": true}`

`_base_` 路径相对于子 YAML 文件所在目录解析。

#### 2. 模式解析

如果配置包含字符串类型的 `mode` 键，且配置中还包含子字典，则匹配模式的子字典会被展平到顶层。

**config_with_modes.yaml：**
```yaml
mode: outdoor
model_name: "qwen-vl"
outdoor:
  step_size: 2.0
  max_distance: 50.0
indoor:
  step_size: 0.5
  max_distance: 10.0
```

**结果：** `{"mode": "outdoor", "model_name": "qwen-vl", "step_size": 2.0, "max_distance": 50.0}`

不匹配的模式子字典（此例中的 `indoor`）从输出中移除。

### `load_yaml`

底层辅助函数，加载 YAML 文件并返回字典，不进行继承或模式处理。

```python
from abotn_evaluator.config import load_yaml

raw_config = load_yaml("path/to/config.yaml")
```

文件不存在时抛出 `FileNotFoundError`，内容不是顶层映射时抛出 `ValueError`。

### `merge_cli_args`

将命令行参数合并到已有的配置字典中。CLI 值覆盖配置值；仅非 `None` 的参数被合并。

```python
from abotn_evaluator.config import merge_cli_args

merged = merge_cli_args(
    config,                 # 基础配置字典
    args,                   # argparse.Namespace 或 dict
    override_keys=None,     # 可选：仅考虑的键列表
)
```

原始 `config` 字典不会被修改；返回一个新字典。

---

## 可视化扩展

Agent 可在 `WaypointPrediction.extra` 中返回调试数据。`enable_visualization=True` 时，评测器按键分发至已注册的处理器。

内置处理器：

- **`affordance_pixel`**：在渲染图像上叠加像素标记

```python
return WaypointPrediction(
    waypoint=wp,
    arrive=False,
    extra={"affordance_pixel": {"front": [360, 320], "left": [200, 400]}},
)
```

像素坐标支持三种格式：归一化 `[0, 1]`、Qwen 格式 `[0, 1000]`、绝对像素。`extra` 为空时即使启用可视化也自动跳过。

添加自定义可视化类型：在 `abotn_evaluator/visualization.py` 的 `_HANDLERS` 字典中注册处理器。

---

## 输出格式

### `result.json`（每个任务）

```json
{
  "episode_id": "scene_001",
  "task_id": "traj_1",
  "status": "stop",
  "steps": 42,
  "travel_length": 28.5,
  "shortest_path_length": 25.3,
  "distance_to_goal": 0.35,
  "success": true,
  "metrics": {
    "success_new": true,
    "spl_new": 0.887,
    "tcr_new": 0.976,
    "dcr_new": 0.992,
    "path_collision_count": 1,
    "collision_path_length": 0.23,
    "total_distance": 28.5,
    "is_path_collided": true
  }
}
```

`status` 取值：`"stop"`（agent 宣称到达）、`"max_steps"`（达到步数上限）、`"collision"`（hard 模式终止）、`"error"`（运行时错误）。

### 汇总报告

由 `analyze_and_report()` 生成：

- **Point-Goal** -> `eval_summary.json`：overall + 难度分组的 `sr_new` / `spl_new`
- **POI-Goal** -> `poi_goal_analysis.json`：overall + 各 POI 统计

CLI runner 评测后自动生成。API 使用时需手动调用 `analyze_and_report(result_dir=...)`。
