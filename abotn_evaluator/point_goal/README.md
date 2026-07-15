# Point Goal Navigation Task

> **Language / 语言**: [English](#english) | [中文](#中文)

---

<a id="english"></a>

## 1. Task Description

The **Point Goal** task evaluates an agent's ability to navigate to a target coordinate `[x, y]` in a photorealistic 3D scene rendered via **3D Gaussian Splatting (3DGS)**. At each time step the evaluator sends the agent an `Observation` containing multi-view RGB images, the goal position in the agent's local frame, and optional map/history data. The agent must respond with a `WaypointPrediction` indicating the next waypoint and whether it believes it has arrived.

The evaluator treats the agent as a **complete black box**: only two methods are called — `reset()` at episode start and `predict(obs)` every step. Any internal architecture (single-system, dual-system fast/slow reasoning, ensemble, caching, etc.) is entirely the agent's responsibility and invisible to the benchmark.

### Evaluation Loop

```
for each episode:
    agent.reset()
    for step in 0..max_steps:
        obs = build_observation(...)
        prediction = agent.predict(obs)
        execute waypoint, check collision, render new view
        if prediction.arrive or distance_to_goal < arrive_threshold:
            break
    save result.json
run metrics analysis → eval_summary.json (overall + per-difficulty)
```

---

## 2. Observation Fields

Defined in `agent_interface.py` → `Observation` dataclass.

| Field | Type | Always? | Description |
|-------|------|---------|-------------|
| `images` | `Dict[str, ndarray(640,720,3)]` | ✅ Yes | Multi-view RGB images keyed by camera name. Default keys: `"left"`, `"front"`, `"right"`. Each value is a `uint8` array of shape `(640, 720, 3)`. |
| `target_position` | `ndarray(2,)` float32 | ✅ Yes | Goal position in the agent's **local** coordinate frame as `[front, left]` in metres. Positive front = forward; positive left = leftward. |
| `position` | `ndarray(3,)` float64 | ✅ Yes | Agent position in **world** coordinates `[x, y, z]`. |
| `rotation` | `ndarray(4,4)` float64 | ✅ Yes | 4×4 camera-to-world pose matrix (rigid-body transform). |
| `heading` | `float` | ✅ Yes | Agent yaw angle in radians, computed as `arctan2(R[1,0], R[0,0])`. |
| `step_count` | `int` | ✅ Yes | Number of steps taken so far in the current episode (0-indexed). |
| `distance_to_goal` | `float` | ✅ Yes | Euclidean distance (metres) from the agent to the goal in the XY plane. |
| `history_images` | `List[Dict[str, ndarray]]` | ⚠️ Optional | List of per-step image dicts from previous steps. Enabled by `--provide-history`. Each dict contains at least a `"front"` key. Images are resized by `--history-resize-ratio`. |
| `history_poses` | `List[ndarray(4,4)]` | ⚠️ Optional | List of 4×4 pose matrices from previous steps. Enabled by `--provide-history`. Length matches `history_images`. |
| `occ_map` | `ndarray` | ⚠️ Optional | Binary occupancy grid for the current scene (0 = free, 1 = occupied). Enabled by `--provide-occ-map`. Free space may be dilated by `--occ-dilation-meters`. |
| `height_map` | `ndarray` | ⚠️ Optional | Ground height map for the current scene. Enabled by `--provide-height-map`. Used internally for Z-coordinate adjustment. |
| `meta_data` | `Dict` | ⚠️ Optional | Scene metadata including coordinate transform parameters (`TOP_LEFT_X`, `TOP_LEFT_Y`, `IMAGE_WIDTH`, `IMAGE_HEIGHT`, `COORDINATE_RANGE_X`, `COORDINATE_RANGE_Y`). Populated when available from the scene. |
| `extra` | `Dict[str, Any]` | ✅ Yes (empty) | Catch-all dictionary for task-specific or experimental data. Defaults to `{}`. |

---

## 3. WaypointPrediction Output Fields

Defined in `agent_interface.py` → `WaypointPrediction` dataclass.

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `waypoint` | `ndarray` shape `(N, 2)` | ✅ Yes | — | Predicted next waypoint(s) in the agent's **local** coordinate frame as `[front, left]` in metres. Typically a single waypoint `(1, 2)`. |
| `arrive` | `bool` | No | `False` | Set to `True` when the agent believes it has reached the goal and wishes to stop the episode. The evaluator also checks `distance_to_goal < arrive_threshold` independently. |
| `directions` | `ndarray` shape `(N, 2)` | No | `None` | Unit direction vector(s) associated with each predicted waypoint. Used by the evaluator to compute the agent's heading at the new pose. If `None`, directions are inferred from consecutive waypoints. |
| `confidence` | `float` | No | `None` | A scalar confidence score in `[0, 1]` for the prediction. Not used by the evaluator but useful for logging and debugging. |

---

## 4. Runner Parameters

All CLI arguments accepted by `python -m abot_n1.tasks.point_goal.runner`:

### Data & Output

| Argument | Type | Default | Required | Description |
|----------|------|---------|----------|-------------|
| `--data-dir` | str | — | ✅ Yes | Root directory containing per-scene subdirectories with benchmark data. |
| `--map-dir` | str | `None` | No | Separate directory for map data (`occ_map`, `height_map`, etc.). Falls back to `--data-dir` if not set. |
| `--output-dir` | str | `./eval_output` | No | Base directory for evaluation outputs. A timestamped subdirectory is created automatically unless `--resume-dir` is used. |
| `--resume-dir` | str | `None` | No | Directory of a previous run to resume. Scans for completed `result.json` files and skips those tasks. |

### Renderer

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--render-url` | str | `http://127.0.0.1:7001/render_gs` | URL of the Gaussian splatting render service. |

### Agent

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--agent-module` | str | `None` | Agent class path as `package.module:ClassName`. Loaded in-process via importlib. |
| `--agent-config` | str | `None` | Optional YAML config file. Contents are passed as keyword arguments to the agent constructor. |
| `--agent-url` | str | `None` | URL of a remote agent HTTP service. When set, the evaluator calls the agent via HTTP instead of loading it in-process. Mutually exclusive with `--agent-module`. |

### Evaluator

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--evaluator-module` | str | `None` | Custom evaluator class path as `package.module:ClassName`. Must accept `(scene, renderer, config, output_dir)` in its constructor. |
| `--max-steps` | int | `100` | Maximum number of agent steps per task before forced stop. |
| `--arrive-threshold` | float | `0.5` | Goal arrival distance threshold in metres. |
| `--collision-threshold` | int | `3` | Path-collision count threshold for SR_NEW metric computation. A task counts as SR_NEW success only if `path_collision_count < collision_threshold`. Outdoor uses `3`; indoor uses `1` (zero collisions allowed). |

### Camera

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--camera-width` | int | `720` | Camera image width in pixels. |
| `--camera-height` | int | `640` | Camera image height in pixels. |
| `--camera-fx` | float | `252.075` | Camera focal length X. |
| `--camera-fy` | float | `252.075` | Camera focal length Y. |
| `--extrinsic-height` | float | `0.65` | Camera height above ground in metres. |

### Observation Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--provide-history` | flag | off | Include `history_images` and `history_poses` in observations. |
| `--provide-occ-map` | flag | off | Include `occ_map` in observations. |
| `--provide-height-map` | flag | off | Include `height_map` in observations. |
| `--max-history-frames` | int | `20` | Maximum number of history frames kept in ShortMemory. |
| `--history-resize-ratio` | float | `0.25` | Downscale ratio applied to history frame images (e.g., 0.25 → ¼ resolution). |

### Output Options

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--save-render-images` | flag | on | Save rendered images to disk (default behaviour). |
| `--no-save-render-images` | flag | — | Disable saving rendered images to disk. |
| `--occ-dilation-meters` | float | `0.0` | Dilate free space in the occupancy map by this many metres. Useful to tolerate annotation error (typical: `0.5`). |

### Metrics Analysis

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--mode` | str | `outdoor` | Difficulty classification mode: `outdoor` (short/medium/long by path length) or `indoor` (easy/hard by scene name). |
| `--min-distance` | float | `5.0` | Outdoor: minimum path length for difficulty bucketing. |
| `--max-distance` | float | `50.0` | Outdoor: maximum path length for difficulty bucketing. |
| `--exclude-scenes` | str list | `None` | Scene IDs to exclude from metrics. Defaults to excluding `park3` in outdoor mode. |
| `--per-scene` | flag | off | Print per-scene metrics breakdown in addition to group-level summaries. |
| `--skip-metrics` | flag | off | Skip post-evaluation metrics analysis entirely. |

---

## 5. Quick Start (Random Agent)

```bash
# 1. Start the 3DGS render server
bash scripts/start_gs_server.sh --scenes-root /path/to/3dgs_scenes

# 2. Evaluate the built-in random agent
python -m abot_n1.tasks.point_goal.runner \
    --agent-module abot_n1.tasks.point_goal.example_agent:RandomPointGoalAgent \
    --data-dir /path/to/benchmark_point_goal \
    --render-url http://localhost:7007/render_gs \
    --output-dir ./results_pointgoal \
    --max-steps 50 \
    --arrive-threshold 0.5

# 3. Analyze results
python -m abot_n1.tasks.point_goal.metrics \
    --result-dir ./results_pointgoal --mode outdoor
```

---

## 6. Custom Agent Example

A minimal agent that implements the required interface:

```python
import numpy as np
from abot_n1.tasks.point_goal.agent_interface import (
    BasePointGoalAgent,
    Observation,
    WaypointPrediction,
)


class MyAgent(BasePointGoalAgent):
    """Minimal point-goal navigation agent."""

    def reset(self) -> None:
        """Called at the start of each new episode."""
        self.step_count = 0

    def predict(self, observation: Observation) -> WaypointPrediction:
        """Return a waypoint toward the goal."""
        self.step_count += 1
        target = observation.target_position  # [front, left] in metres
        dist = float(np.linalg.norm(target))

        # Declare arrival if close enough
        if observation.distance_to_goal < 0.5:
            return WaypointPrediction(
                waypoint=np.array([[0.0, 0.0]], dtype=np.float32),
                arrive=True,
                confidence=1.0,
            )

        # Move toward target at fixed step size
        step_size = 1.0
        direction = target / max(dist, 1e-6)
        actual_step = min(step_size, dist)
        waypoint = direction * actual_step

        return WaypointPrediction(
            waypoint=waypoint.reshape(1, 2),
            arrive=False,
            directions=direction.reshape(1, 2),
            confidence=0.9,
        )
```

Run with:

```bash
python -m abot_n1.tasks.point_goal.runner \
    --agent-module my_package.my_module:MyAgent \
    --data-dir /path/to/benchmark_point_goal \
    --render-url http://localhost:7007/render_gs \
    --output-dir ./results_my_agent
```

---

## 7. Dual-System (Slow Think) Example

The evaluator is architecture-agnostic. Below is a pattern for implementing a dual-system agent where a "slow thinker" provides high-level plans and a "fast system" generates low-level waypoints:

```python
import numpy as np
from abot_n1.tasks.point_goal.agent_interface import (
    BasePointGoalAgent,
    Observation,
    WaypointPrediction,
)


class DualSystemAgent(BasePointGoalAgent):
    """Dual-system agent with slow planning and fast execution."""

    def __init__(self, plan_interval: int = 5):
        self.plan_interval = plan_interval
        self.current_plan = None

    def reset(self) -> None:
        self.step_count = 0
        self.current_plan = None

    # ---- Slow system (high-level planner) ----
    def _slow_think(self, obs: Observation) -> np.ndarray:
        """Generate a high-level subgoal. Called infrequently.

        This could invoke an LLM, a graph planner, or any expensive
        computation. The result is cached and reused across steps.
        """
        # Example: use occupancy map to plan around obstacles
        if obs.occ_map is not None:
            # ... run A* or similar on occ_map ...
            pass
        # Fallback: direct line to goal
        return obs.target_position.copy()

    # ---- Fast system (low-level controller) ----
    def _fast_act(self, obs: Observation, plan: np.ndarray) -> WaypointPrediction:
        """Generate a waypoint using the cached plan. Called every step."""
        dist = float(np.linalg.norm(plan))
        if obs.distance_to_goal < 0.5:
            return WaypointPrediction(
                waypoint=np.array([[0.0, 0.0]], dtype=np.float32),
                arrive=True,
            )
        direction = plan / max(dist, 1e-6)
        waypoint = direction * min(1.0, dist)
        return WaypointPrediction(
            waypoint=waypoint.reshape(1, 2),
            directions=direction.reshape(1, 2),
        )

    # ---- Unified predict (black-box interface) ----
    def predict(self, observation: Observation) -> WaypointPrediction:
        self.step_count += 1

        # Re-plan every N steps or on first step
        if self.current_plan is None or self.step_count % self.plan_interval == 0:
            self.current_plan = self._slow_think(observation)

        return self._fast_act(observation, self.current_plan)
```

Key points:
- The evaluator **never** sees `_slow_think` or `_fast_act` — only `predict()`.
- Planning frequency, model choice, and memory management are all internal.
- Enable `--provide-occ-map` and/or `--provide-history` to give the slow system richer inputs.

---

## 8. Metrics Explanation

Metrics are computed by `metrics.py` and reported by difficulty group. For outdoor mode, tasks are bucketed into **short**, **medium**, and **long** based on `shortest_path_length`.

### Core Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **SR** (Success Rate) | `mean(success)` | Fraction of tasks where `distance_to_goal ≤ arrive_threshold` at termination. |
| **SR_NEW** | `mean(success AND path_collision_count < collision_threshold)` | Collision-aware success rate. A task counts as successful only if the agent arrives **and** the number of path-collision steps is below the threshold (outdoor: 3, indoor: 1 i.e. zero collisions). |
| **SPL_NEW** | `SR_NEW × shortest_path / max(travel_length, shortest_path)` | Path-efficiency-weighted success using SR_NEW as the success indicator. Zero if the agent collided too many times. |
| **TCR_NEW** (Time Compliance Rate) | Per-task (SR_NEW only): `(steps − path_collision_count) / steps`; otherwise 0 | Fraction of time steps without path (line-segment) collisions, gated by the SR_NEW success criterion. Measures temporal safety. |
| **DCR_NEW** (Distance Compliance Rate) | Per-task (SR_NEW only): `(total_distance − collision_path_length) / total_distance`; otherwise 0 | Fraction of travel distance that was collision-free, gated by the SR_NEW success criterion. |

All reported metrics are averaged per task. Point-collision-based metrics (SR_POINT, TCR based on point collisions), plain SPL, and global-ratio variants are no longer produced.

### Difficulty Classification

**Outdoor mode** (`--mode outdoor`):
- Buckets determined by `shortest_path_length` within `[min_distance, max_distance]` divided into three equal ranges.
- Default: short ∈ [5, 20), medium ∈ [20, 35), long ∈ [35, 50×1.5].

**Indoor mode** (`--mode indoor`):
- Scenes are pre-classified as `easy` or `hard` based on scene ID lists defined in `metrics.py`.

---

## 9. Result File Format

### Per-Task: `result.json`

Located at `{output_dir}/{episode_id}/{task_id}/result.json`.

```json
{
  "episode_id": "scene_001",
  "task_id": "task_0042",
  "status": "stop",              // "stop" (arrived) | "max_steps" | "error"
  "target_label": "point_goal",
  "gt_taget_instance": "point_goal_target",
  "steps": 37,                   // Number of steps taken
  "travel_length": 18.42,        // Actual path length (metres)
  "shortest_path_length": 12.5,  // Ground-truth shortest path (metres)
  "distance_to_goal": 0.31,      // Final distance to goal (metres)
  "success": true,               // distance_to_goal <= arrive_threshold
  "oracle_success": true,        // Same as success (reserved for future use)
  "metrics": {
    "initial_distance_to_goal": 14.2,
    "min_distance_to_goal": 0.31,
    "final_distance_to_goal": 0.31,
    "pointgoal_arrive_threshold": 0.5,
    "success_new": true,          // success AND path_collision_count < threshold
    "spl_new": 0.6786,
    "tcr_new": 0.9730,           // (steps - path_collision_count) / steps, gated by success_new
    "dcr_new": 0.9812,           // (total_dist - collision_path_length) / total_dist, gated by success_new
    "path_collision_count": 1,
    "collision_path_length": 0.35,
    "total_distance": 18.62,
    "is_path_collided": true,
    "collision_path_ratio": 0.0188,
    "collision_steps": [12]       // Step indices where path collisions occurred
  }
}
```

### Summary: `eval_summary.json`

Located at `{output_dir}/eval_summary.json`. The single unified output of a run. Contains `overall` aggregates plus per-difficulty `groups` (short/medium/long for outdoor, easy/hard for indoor), along with `task_type`, `mode`, `collision_threshold`, `total_count`, `status_distribution`, and (outdoor) `distance_config`. Each metric block contains `success_rate`, `oracle_success_rate`, `sr_new`, `spl_new`, `tcr_new`, `dcr_new`, and average distance/step/collision statistics.

---

## 10. Agent Interface Reference

```python
class BasePointGoalAgent(ABC):
    @abstractmethod
    def reset(self) -> None:
        """Reset internal state at the start of a new episode."""

    @abstractmethod
    def predict(self, observation: Observation) -> WaypointPrediction:
        """Predict the next waypoint given the current observation."""
```

Only these two methods are called by the evaluator. The agent may use any internal architecture, external services, or caching strategies.

---
---

<a id="中文"></a>

# 点目标导航任务

> **语言 / Language**: [English](#english) | [中文](#中文)

## 1. 任务描述

**点目标（Point Goal）** 任务评估智能体在通过 **3D 高斯泼溅（3DGS）** 渲染的真实感三维场景中，导航至目标坐标 `[x, y]` 的能力。在每个时间步，评估器向智能体发送一个 `Observation`，包含多视角 RGB 图像、智能体局部坐标系下的目标位置以及可选的地图/历史数据。智能体需返回一个 `WaypointPrediction`，指示下一个路点及是否认为已到达目标。

评估器将智能体视为**完全黑盒**：仅调用两个方法——每轮开始时的 `reset()` 和每步的 `predict(obs)`。任何内部架构（单系统、双系统快慢思考、集成模型、缓存等）完全由智能体自行决定，对基准测试不可见。

### 评估流程

```
对于每个 episode:
    agent.reset()
    对于 step = 0..max_steps:
        obs = 构建观测(...)
        prediction = agent.predict(obs)
        执行路点、检测碰撞、渲染新视角
        如果 prediction.arrive 或 distance_to_goal < arrive_threshold:
            终止
    保存 result.json
运行指标分析 → eval_summary.json（总体 + 分难度）
```

---

## 2. 观测字段（Observation）

定义于 `agent_interface.py` → `Observation` 数据类。

| 字段 | 类型 | 始终提供？ | 说明 |
|------|------|-----------|------|
| `images` | `Dict[str, ndarray(640,720,3)]` | ✅ 是 | 多视角 RGB 图像，按相机名称索引。默认键：`"left"`、`"front"`、`"right"`。每个值为 `uint8` 数组，形状 `(640, 720, 3)`。 |
| `target_position` | `ndarray(2,)` float32 | ✅ 是 | 目标在智能体**局部**坐标系中的位置，格式为 `[前方, 左方]`，单位：米。正前方 = 前进方向；正左方 = 左侧方向。 |
| `position` | `ndarray(3,)` float64 | ✅ 是 | 智能体在**世界**坐标系中的位置 `[x, y, z]`。 |
| `rotation` | `ndarray(4,4)` float64 | ✅ 是 | 4×4 相机到世界的位姿矩阵（刚体变换）。 |
| `heading` | `float` | ✅ 是 | 智能体偏航角（弧度），计算方式为 `arctan2(R[1,0], R[0,0])`。 |
| `step_count` | `int` | ✅ 是 | 当前 episode 中已执行的步数（从 0 开始计数）。 |
| `distance_to_goal` | `float` | ✅ 是 | 智能体到目标的 XY 平面欧几里得距离（米）。 |
| `history_images` | `List[Dict[str, ndarray]]` | ⚠️ 可选 | 前几步的图像字典列表。通过 `--provide-history` 启用。每个字典至少包含 `"front"` 键。图像按 `--history-resize-ratio` 缩放。 |
| `history_poses` | `List[ndarray(4,4)]` | ⚠️ 可选 | 前几步的 4×4 位姿矩阵列表。通过 `--provide-history` 启用。长度与 `history_images` 一致。 |
| `occ_map` | `ndarray` | ⚠️ 可选 | 当前场景的二值占据栅格（0 = 空闲，1 = 占据）。通过 `--provide-occ-map` 启用。可通过 `--occ-dilation-meters` 膨胀空闲区域。 |
| `height_map` | `ndarray` | ⚠️ 可选 | 当前场景的地面高度图。通过 `--provide-height-map` 启用。内部用于 Z 坐标校正。 |
| `meta_data` | `Dict` | ⚠️ 可选 | 场景元数据，包括坐标变换参数（`TOP_LEFT_X`、`TOP_LEFT_Y`、`IMAGE_WIDTH`、`IMAGE_HEIGHT`、`COORDINATE_RANGE_X`、`COORDINATE_RANGE_Y`）。当场景提供时自动填充。 |
| `extra` | `Dict[str, Any]` | ✅ 是（空字典） | 通用扩展字典，用于任务特定或实验性数据。默认为 `{}`。 |

---

## 3. 预测输出字段（WaypointPrediction）

定义于 `agent_interface.py` → `WaypointPrediction` 数据类。

| 字段 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `waypoint` | `ndarray` 形状 `(N, 2)` | ✅ 是 | — | 预测的下一个路点，位于智能体**局部**坐标系，格式 `[前方, 左方]`，单位：米。通常为单个路点 `(1, 2)`。 |
| `arrive` | `bool` | 否 | `False` | 设为 `True` 表示智能体认为已到达目标，希望结束本轮。评估器也会独立检查 `distance_to_goal < arrive_threshold`。 |
| `directions` | `ndarray` 形状 `(N, 2)` | 否 | `None` | 与每个预测路点关联的单位方向向量。评估器用其计算新位姿的朝向。若为 `None`，则从相邻路点推断方向。 |
| `confidence` | `float` | 否 | `None` | 预测置信度分数，范围 `[0, 1]`。评估器不使用此字段，但可用于日志记录和调试。 |

---

## 4. 运行参数

`python -m abot_n1.tasks.point_goal.runner` 接受的所有命令行参数：

### 数据与输出

| 参数 | 类型 | 默认值 | 必填 | 说明 |
|------|------|--------|------|------|
| `--data-dir` | str | — | ✅ 是 | 包含各场景子目录的基准数据根目录。 |
| `--map-dir` | str | `None` | 否 | 地图数据（`occ_map`、`height_map` 等）的独立目录。未设置时回退到 `--data-dir`。 |
| `--output-dir` | str | `./eval_output` | 否 | 评估输出的基础目录。自动创建带时间戳的子目录，除非使用 `--resume-dir`。 |
| `--resume-dir` | str | `None` | 否 | 之前运行的目录，用于断点续评。扫描已完成的 `result.json` 并跳过对应任务。 |

### 渲染器

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--render-url` | str | `http://127.0.0.1:7001/render_gs` | 高斯泼溅渲染服务的 URL。 |

### 智能体

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--agent-module` | str | `None` | 智能体类路径，格式 `package.module:ClassName`。通过 importlib 进程内加载。 |
| `--agent-config` | str | `None` | 可选 YAML 配置文件。内容作为关键字参数传递给智能体构造函数。 |
| `--agent-url` | str | `None` | 远程智能体 HTTP 服务 URL。设置后评估器通过 HTTP 调用智能体，而非进程内加载。与 `--agent-module` 互斥。 |

### 评估器

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--evaluator-module` | str | `None` | 自定义评估器类路径，格式 `package.module:ClassName`。构造函数须接受 `(scene, renderer, config, output_dir)`。 |
| `--max-steps` | int | `100` | 每个任务的最大步数，超出后强制停止。 |
| `--arrive-threshold` | float | `0.5` | 到达目标的距离阈值（米）。 |
| `--collision-threshold` | int | `3` | SR_NEW 指标计算的路径碰撞次数阈值。仅当 `path_collision_count < collision_threshold` 时才算 SR_NEW 成功。室外用 `3`；室内用 `1`（即不允许任何碰撞）。 |

### 相机

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--camera-width` | int | `720` | 相机图像宽度（像素）。 |
| `--camera-height` | int | `640` | 相机图像高度（像素）。 |
| `--camera-fx` | float | `252.075` | 相机 X 方向焦距。 |
| `--camera-fy` | float | `252.075` | 相机 Y 方向焦距。 |
| `--extrinsic-height` | float | `0.65` | 相机离地高度（米）。 |

### 观测选项

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--provide-history` | 标志 | 关闭 | 在观测中包含 `history_images` 和 `history_poses`。 |
| `--provide-occ-map` | 标志 | 关闭 | 在观测中包含 `occ_map`。 |
| `--provide-height-map` | 标志 | 关闭 | 在观测中包含 `height_map`。 |
| `--max-history-frames` | int | `20` | ShortMemory 中保留的最大历史帧数。 |
| `--history-resize-ratio` | float | `0.25` | 历史帧图像的缩放比例（如 0.25 → ¼ 分辨率）。 |

### 输出选项

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--save-render-images` | 标志 | 开启 | 将渲染图像保存到磁盘（默认行为）。 |
| `--no-save-render-images` | 标志 | — | 禁止保存渲染图像到磁盘。 |
| `--occ-dilation-meters` | float | `0.0` | 将占据栅格中的空闲区域膨胀指定米数。用于容忍标注误差（典型值：`0.5`）。 |

### 指标分析

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--mode` | str | `outdoor` | 难度分类模式：`outdoor`（按路径长度分短/中/长）或 `indoor`（按场景名分简单/困难）。 |
| `--min-distance` | float | `5.0` | 室外模式：难度分桶的最小路径长度。 |
| `--max-distance` | float | `50.0` | 室外模式：难度分桶的最大路径长度。 |
| `--exclude-scenes` | str 列表 | `None` | 从指标中排除的场景 ID。室外模式默认排除 `park3`。 |
| `--per-scene` | 标志 | 关闭 | 额外打印每个场景的指标明细。 |
| `--skip-metrics` | 标志 | 关闭 | 跳过评估后的指标分析。 |

---

## 5. 快速开始（随机智能体）

```bash
# 1. 启动 3DGS 渲染服务器
bash scripts/start_gs_server.sh --scenes-root /path/to/3dgs_scenes

# 2. 评估内置随机智能体
python -m abot_n1.tasks.point_goal.runner \
    --agent-module abot_n1.tasks.point_goal.example_agent:RandomPointGoalAgent \
    --data-dir /path/to/benchmark_point_goal \
    --render-url http://localhost:7007/render_gs \
    --output-dir ./results_pointgoal \
    --max-steps 50 \
    --arrive-threshold 0.5

# 3. 分析结果
python -m abot_n1.tasks.point_goal.metrics \
    --result-dir ./results_pointgoal --mode outdoor
```

---

## 6. 自定义智能体示例

实现所需接口的最简智能体：

```python
import numpy as np
from abot_n1.tasks.point_goal.agent_interface import (
    BasePointGoalAgent,
    Observation,
    WaypointPrediction,
)


class MyAgent(BasePointGoalAgent):
    """最简的点目标导航智能体。"""

    def reset(self) -> None:
        """每个新 episode 开始时调用。"""
        self.step_count = 0

    def predict(self, observation: Observation) -> WaypointPrediction:
        """返回朝目标方向的路点。"""
        self.step_count += 1
        target = observation.target_position  # [前方, 左方]，单位：米
        dist = float(np.linalg.norm(target))

        # 足够近时宣布到达
        if observation.distance_to_goal < 0.5:
            return WaypointPrediction(
                waypoint=np.array([[0.0, 0.0]], dtype=np.float32),
                arrive=True,
                confidence=1.0,
            )

        # 以固定步长朝目标移动
        step_size = 1.0
        direction = target / max(dist, 1e-6)
        actual_step = min(step_size, dist)
        waypoint = direction * actual_step

        return WaypointPrediction(
            waypoint=waypoint.reshape(1, 2),
            arrive=False,
            directions=direction.reshape(1, 2),
            confidence=0.9,
        )
```

运行命令：

```bash
python -m abot_n1.tasks.point_goal.runner \
    --agent-module my_package.my_module:MyAgent \
    --data-dir /path/to/benchmark_point_goal \
    --render-url http://localhost:7007/render_gs \
    --output-dir ./results_my_agent
```

---

## 7. 双系统（慢思考）示例

评估器不关心智能体的内部架构。以下展示一种双系统模式：「慢思考器」提供高层规划，「快系统」生成底层路点：

```python
import numpy as np
from abot_n1.tasks.point_goal.agent_interface import (
    BasePointGoalAgent,
    Observation,
    WaypointPrediction,
)


class DualSystemAgent(BasePointGoalAgent):
    """双系统智能体：慢规划 + 快执行。"""

    def __init__(self, plan_interval: int = 5):
        self.plan_interval = plan_interval
        self.current_plan = None

    def reset(self) -> None:
        self.step_count = 0
        self.current_plan = None

    # ---- 慢系统（高层规划器）----
    def _slow_think(self, obs: Observation) -> np.ndarray:
        """生成高层子目标。低频调用。

        可调用 LLM、图规划器或任何开销较大的计算。
        结果被缓存并在多步间复用。
        """
        if obs.occ_map is not None:
            # ... 在 occ_map 上运行 A* 等算法 ...
            pass
        # 回退：直线朝目标
        return obs.target_position.copy()

    # ---- 快系统（底层控制器）----
    def _fast_act(self, obs: Observation, plan: np.ndarray) -> WaypointPrediction:
        """使用缓存的计划生成路点。每步调用。"""
        dist = float(np.linalg.norm(plan))
        if obs.distance_to_goal < 0.5:
            return WaypointPrediction(
                waypoint=np.array([[0.0, 0.0]], dtype=np.float32),
                arrive=True,
            )
        direction = plan / max(dist, 1e-6)
        waypoint = direction * min(1.0, dist)
        return WaypointPrediction(
            waypoint=waypoint.reshape(1, 2),
            directions=direction.reshape(1, 2),
        )

    # ---- 统一的 predict（黑盒接口）----
    def predict(self, observation: Observation) -> WaypointPrediction:
        self.step_count += 1

        # 每 N 步或在第一步时重新规划
        if self.current_plan is None or self.step_count % self.plan_interval == 0:
            self.current_plan = self._slow_think(observation)

        return self._fast_act(observation, self.current_plan)
```

要点：
- 评估器**永远看不到** `_slow_think` 或 `_fast_act`——只看到 `predict()`。
- 规划频率、模型选择和内存管理完全是内部事务。
- 使用 `--provide-occ-map` 和/或 `--provide-history` 为慢系统提供更丰富的输入。

---

## 8. 指标说明

指标由 `metrics.py` 计算，按难度分组报告。室外模式下，任务根据 `shortest_path_length` 分为 **短**、**中**、**长** 三档。

### 核心指标

| 指标 | 公式 | 说明 |
|------|------|------|
| **SR**（成功率） | `mean(success)` | 终止时 `distance_to_goal ≤ arrive_threshold` 的任务比例。 |
| **SR_NEW** | `mean(success AND path_collision_count < collision_threshold)` | 碰撞感知成功率。仅在智能体到达**且**路径碰撞步数低于阈值（室外 3，室内 1，即零碰撞）时才算成功。 |
| **SPL_NEW** | `SR_NEW × shortest_path / max(travel_length, shortest_path)` | 以 SR_NEW 作为成功判定的路径效率加权成功率。碰撞过多则为零。 |
| **TCR_NEW**（时间合规率） | 单任务（仅 SR_NEW 成功）：`(steps − path_collision_count) / steps`；否则 0 | 以 SR_NEW 成功标准为前提的无路径（线段）碰撞时间步占比。衡量时间维度的安全性。 |
| **DCR_NEW**（距离合规率） | 单任务（仅 SR_NEW 成功）：`(total_distance − collision_path_length) / total_distance`；否则 0 | 以 SR_NEW 成功标准为前提的无碰撞行驶距离占比。 |

所有报告指标均为各任务均值。基于点碰撞的指标（SR_POINT、基于点碰撞的 TCR）、普通 SPL 以及各类全局比率变体已不再产出。

### 难度分类

**室外模式**（`--mode outdoor`）：
- 按 `shortest_path_length` 在 `[min_distance, max_distance]` 范围内三等分。
- 默认：短 ∈ [5, 20)，中 ∈ [20, 35)，长 ∈ [35, 50×1.5]。

**室内模式**（`--mode indoor`）：
- 场景根据 `metrics.py` 中定义的场景 ID 列表预分类为 `easy`（简单）或 `hard`（困难）。

---

## 9. 结果文件格式

### 单任务：`result.json`

位于 `{output_dir}/{episode_id}/{task_id}/result.json`。

```json
{
  "episode_id": "scene_001",
  "task_id": "task_0042",
  "status": "stop",              // "stop"（到达）| "max_steps" | "error"
  "target_label": "point_goal",
  "gt_taget_instance": "point_goal_target",
  "steps": 37,                   // 执行的步数
  "travel_length": 18.42,        // 实际路径长度（米）
  "shortest_path_length": 12.5,  // 真值最短路径（米）
  "distance_to_goal": 0.31,      // 最终距目标距离（米）
  "success": true,               // distance_to_goal <= arrive_threshold
  "oracle_success": true,        // 同 success（预留字段）
  "metrics": {
    "initial_distance_to_goal": 14.2,
    "min_distance_to_goal": 0.31,
    "final_distance_to_goal": 0.31,
    "pointgoal_arrive_threshold": 0.5,
    "success_new": true,          // success AND path_collision_count < threshold
    "spl_new": 0.6786,
    "tcr_new": 0.9730,           // (steps - path_collision_count) / steps，以 success_new 为前提
    "path_collision_count": 1,
    "collision_path_length": 0.35,
    "total_distance": 18.62,
    "is_path_collided": true,
    "collision_path_ratio": 0.0188,
    "collision_steps": [12]       // 发生路径碰撞的步骤索引
  }
}
```

### 汇总：`eval_summary.json`

位于 `{output_dir}/eval_summary.json`。一次运行的唯一统一输出。包含 `overall` 总体聚合及分难度 `groups`（室外为 short/medium/long，室内为 easy/hard），以及 `task_type`、`mode`、`collision_threshold`、`total_count`、`status_distribution` 和（室外）`distance_config`。每个指标块包含 `success_rate`、`oracle_success_rate`、`sr_new`、`spl_new`、`tcr_new`、`dcr_new` 及平均距离/步数/碰撞统计。

---

## 10. 智能体接口参考

```python
class BasePointGoalAgent(ABC):
    @abstractmethod
    def reset(self) -> None:
        """在新 episode 开始时重置内部状态。"""

    @abstractmethod
    def predict(self, observation: Observation) -> WaypointPrediction:
        """根据当前观测预测下一个路点。"""
```

评估器仅调用这两个方法。智能体可使用任何内部架构、外部服务或缓存策略。
