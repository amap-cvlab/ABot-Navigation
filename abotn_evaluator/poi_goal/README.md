# POI Goal Task

Navigate to a named Point of Interest (e.g. "Starbucks") identified by
visual recognition rather than raw coordinates. The agent receives a
`poi_name` string and must locate the corresponding shop, landmark, or
facade within a 3DGS-rendered scene.

## Quick Start

```bash
# 1. Start the render server
bash scripts/start_gs_server.sh --scenes-root /path/to/3dgs_scenes

# 2. Evaluate the built-in example agent
python -m abot_n1.tasks.poi_goal.runner \
    --agent-module abot_n1.tasks.poi_goal.example_agent:RandomPoiGoalAgent \
    --data-dir /path/to/poi_goal_benchmark \
    --render-url http://localhost:7007/render_gs \
    --output-dir ./results_poigoal \
    --arrive-threshold 1.0 \
    --collision-mode hard

# 3. Analyze results
python -m abot_n1.tasks.poi_goal.metrics \
    --result-dir ./results_poigoal --mode indoor
```

## Agent Interface

Implement `BasePoiGoalAgent` from `agent_interface.py`. The evaluator
treats the agent as a **black box**: it sends a `PoiGoalObservation` at
every step and expects a `WaypointPrediction` back. Any internal
architecture — single model, dual-system fast/slow reasoning, ensembles,
caching — is entirely up to the agent's `predict` implementation.

| Method | When called | Must return |
|--------|-------------|-------------|
| `reset()` | Start of each episode | — |
| `predict(obs)` | Every step | `WaypointPrediction` |

### PoiGoalObservation Fields

`PoiGoalObservation` inherits every field from the point-goal
`Observation` and adds one new field:

| Field | Type | Always? | Description |
|-------|------|---------|-------------|
| `images` | `Dict[str, ndarray(640,720,3)]` | Yes | `"left"`, `"front"`, `"right"` RGB views |
| `target_position` | `ndarray(2,)` | Yes | `[front, left]` metres in local frame (rough direction hint) |
| `position` | `ndarray(3,)` | Yes | World `[x, y, z]` coordinates |
| `rotation` | `ndarray(4,4)` | Yes | 4×4 camera-to-world pose matrix |
| `heading` | `float` | Yes | Yaw angle in radians |
| `step_count` | `int` | Yes | Steps taken so far in this episode |
| `distance_to_goal` | `float` | Yes | Euclidean distance to goal in XY plane (metres) |
| **`poi_name`** | **`str`** | **Yes** | **Target POI name, e.g. `"Starbucks"`, `"H&M"`** |
| `history_images` | `List[Dict]` | `--provide-history` | Past per-step image dicts |
| `history_poses` | `List[ndarray]` | `--provide-history` | Past 4×4 pose matrices |
| `occ_map` | `ndarray` | `--provide-occ-map` | Occupancy grid for current scene |
| `height_map` | `ndarray` | `--provide-height-map` | Height map for current scene |
| `meta_data` | `dict` | Optional | Scene metadata (coordinate transforms, etc.) |
| `extra` | `dict` | Always | Catch-all for task-specific or experimental data |

> **Key difference from Point Goal:** `target_position` provides only a
> rough coordinate direction. The agent **must visually recognise** the
> POI sign or facade to succeed — pure dead-reckoning will not work.

### WaypointPrediction Fields

Identical to Point Goal:

| Field | Type | Description |
|-------|------|-------------|
| `waypoint` | `ndarray(2,)` | `[front, left]` metres in local frame |
| `arrive` | `bool` | `True` → agent declares arrival; episode ends |
| `directions` | `ndarray(N,2)` (optional) | Unit direction vector(s) |
| `confidence` | `float` (optional) | Scalar confidence score in `[0, 1]` |

## Collision Modes

The runner supports three collision handling strategies selected via
`--collision-mode`:

| Mode | Behaviour | Use case |
|------|-----------|----------|
| `off` | Collisions are ignored entirely. The agent passes through obstacles and no penalty is applied. | Debugging path-planning logic without safety constraints. |
| `soft` | Collisions are recorded in metrics (`path_collision`) but the episode continues. Success metrics (SR_NEW, SPL_NEW) penalise collisions after evaluation. | Default balanced mode — measures safety without aborting runs. |
| `hard` | The episode **terminates immediately** on the first collision. The result is marked `collision_terminated: true` and counts as a failure. | Strict safety evaluation — any contact is unacceptable. |

Collision detection uses the robot footprint defined by `--robot-radius`
(default varies by scenario). A collision occurs when the robot's
circular footprint intersects an occupied cell in the occupancy map.

## Arrive Threshold: POI Goal vs Point Goal

| Parameter | Point Goal default | POI Goal default | Rationale |
|-----------|-------------------|------------------|-----------|
| `--arrive-threshold` | 0.5 m | **1.0 m** | POIs are physical objects with spatial extent; standing within 2 m of the storefront/facade constitutes successful arrival. |

The larger threshold affects both success determination and SPL
calculation:

- **Success:** An episode is successful if `distance_to_goal ≤ arrive_threshold` when the agent sets `arrive=True` (or reaches max steps within range).
- **SPL adjustment:** `effective_shortest = max(shortest_path − arrive_threshold, 1e-3)` accounts for the acceptance radius so that agents are not penalised for stopping slightly before the exact centre point.

## Runner Parameters

```
--agent-module       package.module:ClassName  (or --agent-url for HTTP mode)
--data-dir           benchmark data root directory
--render-url         3DGS render server URL, e.g. http://localhost:7007/render_gs
--output-dir         directory to save evaluation results
--max-steps          maximum steps per episode (default 100)
--arrive-threshold   2.0 metres (default for POI Goal; cf. 0.5 m for Point Goal)
--collision-mode     off | soft | hard  (default: soft)
--robot-radius       robot footprint radius in metres for collision checking
--provide-history    include history_images and history_poses in observations
--provide-occ-map    include occupancy map in observations
--occ-dilation-meters  dilate free space to tolerate annotation error (0.5 typical)
--provide-height-map include height map in observations
--mode               outdoor | indoor  (for metrics grouping)
```

## Metrics

Computed by `metrics.py` and reported by difficulty group (short / medium
/ long for outdoor scenarios):

| Metric | Formula | Description |
|--------|---------|-------------|
| **SR** | successes / total | Raw success rate |
| **SPL** | Σ(success_i × shortest_i / max(shortest_i, path_i)) / N | Path-length-weighted success |
| **SR_NEW** | SR with collision penalty | Success only if `path_collision < threshold` |
| **SPL_NEW** | SPL with collision penalty | SPL gated by collision-free requirement |
| **TCR** | collision-free episodes / total | Time Compliance Rate |
| **DCR** | collision-free episodes / total | Distance Compliance Rate |

### Per-POI Metrics Breakdown

In addition to aggregate metrics, `eval_summary.json` contains a
**`poi_stats`** section that breaks down performance by individual POI
name:

```json
{
  "poi_stats": {
    "Starbucks": {
      "total": 20,
      "success": 15,
      "sr": 0.75,
      "spl": 0.62,
      "avg_distance_to_goal": 1.8
    },
    "H&M": {
      "total": 18,
      "success": 10,
      "sr": 0.56,
      "spl": 0.41,
      "avg_distance_to_goal": 3.2
    }
  }
}
```

This breakdown enables analysis such as:

- Which POI categories (food, retail, services) are harder to navigate to?
- Are failures correlated with visual similarity between POIs?
- Does signage size or placement affect recognition accuracy?

### Additional Fields

- **`collision_terminated`** — number of episodes ended early due to
  collision in `hard` mode. Useful for quantifying safety violations.

### Output Files

| File | Contents |
|------|----------|
| `result.json` | Per-task detailed results |
| `eval_summary.json` | Aggregate + per-POI metrics |
| `difficulty_analysis.json` | Metrics grouped by difficulty level |

---

# POI Goal 任务（中文）

导航至指定的兴趣点（Point of Interest），例如"星巴克"。目标通过名称
标识，需要依靠视觉识别而非坐标定位。

## 快速开始

```bash
# 1. 启动渲染服务器
bash scripts/start_gs_server.sh --scenes-root /path/to/3dgs_scenes

# 2. 运行评测
python -m abot_n1.tasks.poi_goal.runner \
    --agent-module abot_n1.tasks.poi_goal.example_agent:RandomPoiGoalAgent \
    --data-dir /path/to/poi_goal_benchmark \
    --render-url http://localhost:7007/render_gs \
    --output-dir ./results_poigoal \
    --arrive-threshold 1.0 \
    --collision-mode hard

# 3. 分析结果
python -m abot_n1.tasks.poi_goal.metrics \
    --result-dir ./results_poigoal --mode indoor
```

## Agent 接口

实现 `agent_interface.py` 中的 `BasePoiGoalAgent`。评测器将 Agent 视为
**黑盒**：每步发送一个 `PoiGoalObservation`，期望返回一个
`WaypointPrediction`。内部架构（单模型、双系统快慢推理、集成等）完全由
Agent 自行决定。

| 方法 | 调用时机 | 返回值 |
|------|---------|--------|
| `reset()` | 每个 episode 开始时 | 无 |
| `predict(obs)` | 每一步 | `WaypointPrediction` |

### PoiGoalObservation 字段说明

`PoiGoalObservation` 继承 Point Goal 的所有字段，并新增一个字段：

| 字段 | 类型 | 是否必有 | 说明 |
|------|------|---------|------|
| `images` | `Dict[str, ndarray(640,720,3)]` | 是 | `"left"`、`"front"`、`"right"` RGB 图像 |
| `target_position` | `ndarray(2,)` | 是 | 局部坐标系下 `[前方, 左方]` 米（仅为粗略方向提示） |
| `position` | `ndarray(3,)` | 是 | 世界坐标 `[x, y, z]` |
| `rotation` | `ndarray(4,4)` | 是 | 4×4 相机到世界的位姿矩阵 |
| `heading` | `float` | 是 | 偏航角（弧度） |
| `step_count` | `int` | 是 | 当前 episode 已执行步数 |
| `distance_to_goal` | `float` | 是 | XY 平面内到目标的欧氏距离（米） |
| **`poi_name`** | **`str`** | **是** | **目标 POI 名称，如 `"星巴克"`、`"H&M"`** |
| `history_images` | `List[Dict]` | 需 `--provide-history` | 历史帧图像 |
| `history_poses` | `List[ndarray]` | 需 `--provide-history` | 历史帧位姿矩阵 |
| `occ_map` | `ndarray` | 需 `--provide-occ-map` | 占用栅格地图 |
| `height_map` | `ndarray` | 需 `--provide-height-map` | 高度图 |
| `meta_data` | `dict` | 可选 | 场景元数据（坐标变换参数等） |
| `extra` | `dict` | 是 | 预留扩展字段 |

> **与 Point Goal 的关键区别：** `target_position` 仅提供粗略的方向参考，
> Agent **必须通过视觉识别** POI 招牌或门面才能成功到达，纯航迹推算无法
> 完成任务。

### WaypointPrediction 字段说明

与 Point Goal 相同：

| 字段 | 类型 | 说明 |
|------|------|------|
| `waypoint` | `ndarray(2,)` | 局部坐标系下 `[前方, 左方]` 米 |
| `arrive` | `bool` | `True` 表示 Agent 判定已到达，episode 结束 |
| `directions` | `ndarray(N,2)`（可选） | 单位方向向量 |
| `confidence` | `float`（可选） | 置信度分数，范围 `[0, 1]` |

## 碰撞模式

通过 `--collision-mode` 选择碰撞处理策略：

| 模式 | 行为 | 适用场景 |
|------|------|---------|
| `off` | 完全忽略碰撞，Agent 可穿越障碍物，无任何惩罚。 | 调试路径规划逻辑时使用。 |
| `soft` | 碰撞记录到指标中（`path_collision`），但 episode 继续执行。评测后在 SR_NEW/SPL_NEW 中对碰撞进行惩罚。 | 默认均衡模式——衡量安全性但不中断运行。 |
| `hard` | 发生碰撞时 **立即终止** episode。结果标记为 `collision_terminated: true`，计为失败。 | 严格安全评测——任何接触均不可接受。 |

碰撞检测使用 `--robot-radius` 定义的机器人足迹半径。当机器人圆形足迹
与占用栅格中的障碍单元格相交时，判定为碰撞。

## 到达阈值：POI Goal 与 Point Goal 对比

| 参数 | Point Goal 默认值 | POI Goal 默认值 | 原因 |
|------|------------------|-----------------|------|
| `--arrive-threshold` | 0.5 m | **1.0 m** | POI 是具有空间范围的物理实体；距店面/门面 2 米以内即视为成功到达。 |

更大的阈值同时影响成功判定和 SPL 计算：

- **成功判定：** 当 Agent 设置 `arrive=True` 且 `distance_to_goal ≤ arrive_threshold` 时，判定为成功。
- **SPL 调整：** `effective_shortest = max(shortest_path − arrive_threshold, 1e-3)` 考虑了接受半径，避免对停在精确中心点之前的 Agent 产生不公平惩罚。

## 运行参数

```
--agent-module       包.模块:类名（或使用 --agent-url 进入 HTTP 模式）
--data-dir           基准数据根目录
--render-url         3DGS 渲染服务器地址，如 http://localhost:7007/render_gs
--output-dir         评测结果保存目录
--max-steps          每个 episode 最大步数（默认 100）
--arrive-threshold   到达阈值，默认 1.0 米（Point Goal 为 0.5 米）
--collision-mode     off | soft | hard（默认 soft）
--robot-radius       碰撞检测用的机器人足迹半径（米）
--provide-history    在观测中包含历史图像和位姿
--provide-occ-map    在观测中包含占用栅格地图
--occ-dilation-meters  膨胀自由空间以容忍标注误差（典型值 0.5）
--provide-height-map 在观测中包含高度图
--mode               outdoor | indoor（用于指标分组）
```

## 评测指标

由 `metrics.py` 计算，按难度分组报告（室外场景分为短/中/长距离）：

| 指标 | 公式 | 说明 |
|------|------|------|
| **SR** | 成功数 / 总数 | 原始成功率 |
| **SPL** | Σ(success_i × shortest_i / max(shortest_i, path_i)) / N | 路径长度加权成功率 |
| **SR_NEW** | 带碰撞惩罚的 SR | 仅在 `path_collision < threshold` 时算成功 |
| **SPL_NEW** | 带碰撞惩罚的 SPL | 要求无碰撞才计入 SPL |
| **TCR** | 无碰撞 episode 数 / 总数 | 时间合规率 |
| **DCR** | 无碰撞 episode 数 / 总数 | 距离合规率 |

### 按 POI 分类的指标明细

除汇总指标外，`eval_summary.json` 还包含 **`poi_stats`** 部分，按单个
POI 名称拆分性能数据：

```json
{
  "poi_stats": {
    "星巴克": {
      "total": 20,
      "success": 15,
      "sr": 0.75,
      "spl": 0.62,
      "avg_distance_to_goal": 1.8
    },
    "H&M": {
      "total": 18,
      "success": 10,
      "sr": 0.56,
      "spl": 0.41,
      "avg_distance_to_goal": 3.2
    }
  }
}
```

此明细支持以下分析：

- 哪些 POI 类别（餐饮、零售、服务）更难导航？
- 失败是否与 POI 之间的视觉相似性相关？
- 招牌大小或摆放位置是否影响识别准确率？

### 其他字段

- **`collision_terminated`** — 在 `hard` 模式下因碰撞提前终止的 episode
  数量，用于量化安全违规情况。

### 输出文件

| 文件 | 内容 |
|------|------|
| `result.json` | 每个任务的详细结果 |
| `eval_summary.json` | 汇总指标 + 按 POI 分类的指标 |
| `difficulty_analysis.json` | 按难度级别分组的指标 |
