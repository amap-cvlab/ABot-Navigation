# POI-Goal 评测

[English](../poi-goal.md) | [中文](poi-goal.md)

POI-Goal 导航任务中，agent 接收 POI 名称（如"星巴克"），需通过视觉识别导航至其入口 2.0 m 以内。

## 与 Point-Goal 的区别

| | Point-Goal | POI-Goal |
|:-|:-----------|:---------|
| 目标指定 | (x, y) 坐标 | POI 名称（`poi_name` 字段） |
| 到达阈值 | 0.5 m | 2.0 m |
| 碰撞处理 | 次数阈值（3 或 1） | Hard 模式（首次碰撞即终止） |
| 协议模式 | outdoor / indoor | 单一协议 |
| 渲染倍率 | 1.0 | 1.5（超采样，提升招牌清晰度） |
| 指标分组 | 按距离分组 | 全局 + 各 POI |

## 协议参数

| 参数 | 值 |
|:-----|:---|
| `arrive_threshold` | 2.0 m |
| `collision_mode` | `"hard"`（首次非豁免碰撞即终止） |
| `occ_dilation_meters` | 0.5 |
| `max_steps` | 100 |

目标圈（距目标 2.0 m 以内）内的碰撞豁免，因 POI 入口位于建筑外立面。

协议参数为**固定基准标准**，修改后产出的结果不可比。

## 前提条件

- 已安装评测框架（`pip install abotn-bench`）
- 渲染服务已用 `scripts/start_POIGoal_render_server.sh` 启动（`RENDER_SCALE=1.5`）
- 已下载 ABotN-POIBench 数据

## 通过 Python API 评测

接口字段与坐标系详见 [API 参考](api-reference.md)。如需适配不同 I/O 约定的模型，见[自定义 Agent](custom-agents.md)。

```python
import os
from datetime import datetime
from abotn_evaluator.poi_goal.evaluator import PoiGoalEvaluator, PoiGoalEvalConfig
from abotn_evaluator.scene import GaussianScene
from abotn_evaluator.render_client import GaussianRenderer
from abotn_evaluator.poi_goal.metrics import analyze_and_report
from your_agent_module import YourPoiAgent

RENDER_URL = "http://localhost:7036/render_gs"

scene = GaussianScene(
    local_data_path="/path/to/ABotN-POIBench/annotations",
    local_map_path="/path/to/ABotN-POIBench/occmaps",
)
renderer = GaussianRenderer(render_url=RENDER_URL)
config = PoiGoalEvalConfig(render_url=RENDER_URL, max_steps=100)

run_dir = os.path.join("./results/poi", datetime.now().strftime("%Y%m%d_%H%M%S"))
evaluator = PoiGoalEvaluator(scene=scene, renderer=renderer, config=config, output_dir=run_dir)

results = evaluator.evaluate(YourPoiAgent())
report = analyze_and_report(result_dir=run_dir)
```

`PoiGoalEvalConfig` 默认值即标准协议（`arrive_threshold=2.0`、`collision_mode="hard"`、`occ_dilation_meters=0.5`），无需手动覆盖。


## 通过 CLI 评测

```bash
python -m abotn_evaluator.poi_goal.runner \
    --agent-module your_agent_module:YourPoiAgent \
    --data-dir /path/to/ABotN-POIBench/annotations \
    --map-dir /path/to/ABotN-POIBench/occmaps \
    --render-url http://localhost:7036/render_gs \
    --output-dir ./results/poi \
    --max-steps 100
```

POI-Goal 无需指定 `--mode`，协议参数已内置默认值。断点续跑与多 GPU 机制同 Point-Goal（见 [Point-Goal 评测](point-goal.md#断点续跑与多-gpu)）。

### 完整参数参考

参数按类别分组。除标注 **（必填）** 外，所有参数均为可选。

#### 数据 / 输出

| 参数 | 类型 | 默认值 | 说明 |
|:-----|:-----|:-------|:-----|
| `--data-dir` | str | **（必填）** | 包含各场景子目录的根目录 |
| `--map-dir` | str | `None` | 地图数据的单独目录（occ_map、height_map 等）。省略时从 `--data-dir` 加载 |
| `--output-dir` | str | `./eval_output` | 评测输出目录。每次运行会创建带时间戳的子目录 |
| `--resume-dir` | str | `None` | 上次运行的输出目录，用于断点续跑。扫描已完成的 `result.json` 并跳过 |

#### Agent

| 参数 | 类型 | 默认值 | 说明 |
|:-----|:-----|:-------|:-----|
| `--agent-module` | str | **（必填）** | Agent 类路径，格式为 `package.module:ClassName` |
| `--agent-config` | str | `None` | 可选的 Agent YAML 配置文件。YAML 键值作为 `**kwargs` 传入 `__init__` |
| `--evaluator-module` | str | `None` | 自定义评测器类路径。构造函数须接受 `(scene, renderer, config, output_dir)` |

#### 渲染器

| 参数 | 类型 | 默认值 | 说明 |
|:-----|:-----|:-------|:-----|
| `--render-url` | str | `http://127.0.0.1:7001/render_gs` | Gaussian splatting 渲染服务的 URL |

#### POI 专属参数

以下参数为 POI-Goal 评测专属，在 Point-Goal runner 中不存在。

| 参数 | 类型 | 默认值 | 说明 |
|:-----|:-----|:-------|:-----|
| `--arrive-threshold` | float | `2.0` | 目标到达距离阈值（米）。比 Point-Goal（0.5 m）更大，因为 POI 目标通常在建筑外立面 |
| `--collision-mode` | str | `hard` | 碰撞处理模式：`off`、`soft` 或 `hard`。详见[碰撞模式详解](#碰撞模式详解) |
| `--robot-radius` | float | `0.0` | 机器人半径（米），用于圆形足迹碰撞检测。`0.0` 表示仅检查中心像素 |
| `--occ-obstacle-polarity` | str | `dark` | 占据图像素解读方式：`dark` = 暗色像素为障碍物；`light` = 亮色像素为障碍物 |
| `--occ-dark-threshold` | int | `64` | 障碍物检测的灰度阈值。在 `dark` 极性下，像素值 <= 此阈值视为障碍物 |
| `--per-poi` / `--no-per-poi` | 标志 | `True` | 是否在分析输出中打印各 POI 的指标分解 |

#### 评测参数

| 参数 | 类型 | 默认值 | 说明 |
|:-----|:-----|:-------|:-----|
| `--max-steps` | int | `100` | 每个任务的最大步数 |
| `--collision-threshold` | int | `3` | 碰撞次数阈值，用于旧版 `SR_NEW` 指标（保留向后兼容） |
| `--occ-dilation-meters` | float | `0.5` | 在占据图中将自由空间膨胀的米数 |

#### 相机

| 参数 | 类型 | 默认值 | 说明 |
|:-----|:-----|:-------|:-----|
| `--camera-width` | int | `720` | 相机图像宽度（像素） |
| `--camera-height` | int | `640` | 相机图像高度（像素） |
| `--camera-fx` | float | `252.075` | 相机 x 方向焦距 |
| `--camera-fy` | float | `252.075` | 相机 y 方向焦距 |
| `--extrinsic-height` | float | `0.65` | 相机距地面高度（米） |

#### 观测选项

| 参数 | 类型 | 默认值 | 说明 |
|:-----|:-----|:-------|:-----|
| `--provide-history` | 标志 | `False` | 在观测中包含历史图像和位姿 |
| `--provide-occ-map` | 标志 | `False` | 在观测中包含占据图 |
| `--provide-height-map` | 标志 | `False` | 在观测中包含高度图 |
| `--max-history-frames` | int | `20` | ShortMemory 中保留的最大历史帧数 |
| `--history-resize-ratio` | float | `0.25` | 历史帧图像的缩放比例 |

#### 输出选项

| 参数 | 类型 | 默认值 | 说明 |
|:-----|:-----|:-------|:-----|
| `--save-render-images` / `--no-save-render-images` | 标志 | `True` | 是否将渲染图像保存到磁盘 |
| `--enable-visualization` | 标志 | `False` | 启用 agent extra 字段的可视化（如可通行性像素叠加） |

#### 指标分析

| 参数 | 类型 | 默认值 | 说明 |
|:-----|:-----|:-------|:-----|
| `--mode` | str | `indoor` | 指标分组的难度分类模式：`outdoor` 或 `indoor` |
| `--min-distance` | float | `5.0` | Outdoor：路径长度分组的最小值 |
| `--max-distance` | float | `50.0` | Outdoor：路径长度分组的最大值 |
| `--exclude-scenes` | str 列表 | `None` | 从指标中排除的场景 ID |
| `--per-scene` | 标志 | `False` | 打印各场景的指标分解 |
| `--skip-metrics` | 标志 | `False` | 跳过评测后的指标分析（仅生成各任务的 `result.json`） |

## 碰撞模式详解

POI-Goal 评测支持三种碰撞检测模式，通过 `--collision-mode` 控制：

### `off` -- 关闭碰撞检测

完全不进行碰撞检测。不查询占据图，不记录碰撞统计。适用于仅测量纯导航精度而不考虑碰撞的场景。

### `soft` -- 仅记录

检测并记录碰撞到 `result.json` 的指标中，但碰撞**不影响**成功判定，也不导致提前终止。即使撞到障碍物，agent 仍可继续导航。此模式适合在开发阶段收集碰撞统计数据，同时保留完成任务的能力。

记录的统计量包括 `collision_count`、`collision_rate`、`max_consecutive_collision` 和 `collision_step_indices`。

### `hard` -- 首次碰撞即终止（默认）

首次非豁免碰撞时立即终止 episode。任务标记为失败，`status="collision"` 且 `success=false`，无论 agent 距目标多近。这是**标准基准模式**。

### 终端区豁免

在 `soft` 和 `hard` 模式下，发生在**到达阈值圈**（距目标位置 2.0 m 以内）内的碰撞**被豁免** -- 不计入碰撞次数，不触发终止。此豁免的原因是 POI 入口位于建筑外立面：agent 必须靠近建筑墙体才能到达目标，而占据图将建筑标记为障碍物。

被豁免的碰撞单独记录在结果指标的 `collision_skipped_near_goal` 字段中，用于诊断分析。

### `collision_terminated` 标志

每个任务结果的 metrics 中包含 `collision_terminated` 布尔值：

- `true`：任务被 hard 模式碰撞提前终止。`status` 字段为 `"collision"`。
- `false`：任务正常结束 -- agent 声明到达（`status="stop"`）、达到最大步数（`status="max_steps"`）或碰撞模式非 `hard`。

## 评测中途查看指标

在评测运行过程中，随时可以运行以下命令计算已完成任务的汇总指标：

```bash
python -m abotn_evaluator.poi_goal.metrics --result-dir <dir>
```

该命令扫描指定目录下所有已完成的 `result.json` 文件，打印与完整评测结束时相同的汇总表。可选参数：

```bash
python -m abotn_evaluator.poi_goal.metrics \
    --result-dir ./results/poi/20260713_143022 \
    --arrive-threshold 2.0 \
    --per-poi                   # 各 POI 分解（默认开启）
    --per-scene                 # 各场景分解
    --mode indoor               # 难度分组
    --exclude-scenes scene_042  # 跳过指定场景
    --output-path ./custom_analysis.json
```

命令将 `poi_goal_analysis.json` 写入结果目录（或 `--output-path` 指定的路径），并在标准输出打印格式化的汇总信息。

## Agent 接口

与 Point-Goal 唯一区别：`predict()` 接收 `PoiGoalObservation`，额外包含 `poi_name: str` 字段：

```python
from abotn_evaluator.interface.poi_goal import BasePoiGoalAgent, PoiGoalObservation
from abotn_evaluator.interface.point_goal import WaypointPrediction

class YourPoiAgent(BasePoiGoalAgent):
    def reset(self): ...
    def predict(self, obs: PoiGoalObservation) -> WaypointPrediction:
        name = obs.poi_name  # 如"星巴克"
        return WaypointPrediction(waypoint=..., arrive=obs.distance_to_goal < 2.0)
```

其余字段（`images`、`target_position`、`waypoint` 等）与 Point-Goal 一致。详见 [API 参考](api-reference.md)。

## 评测指标

`analyze_and_report` 生成 `poi_goal_analysis.json`，包含全局、分组和各 POI 的统计数据。

### 全局指标

| 字段 | 说明 |
|:-----|:-----|
| `success_rate` | agent 在 `arrive_threshold` 以内完成任务的比例（被碰撞终止的不计） |
| `spl` | 平均 Success weighted by Path Length（见下方公式） |
| `avg_steps` | 每个任务的平均步数 |
| `avg_travel_length` | 平均实际行走距离 |
| `avg_shortest_path_length` | 平均真值最短路径长度 |
| `avg_initial_distance` | 平均初始目标距离 |
| `avg_min_distance` | episode 中达到的平均最小目标距离 |
| `avg_final_distance` | 平均最终目标距离 |

### POI-SPL 公式

标准 SPL 为 `success * shortest / max(travel, shortest)`。但 POI-Goal 使用 2.0 m 的到达阈值，相对于许多路径长度来说较大。如果最短路径为 5.0 m，一个完全高效的 agent 只需走 3.0 m。在分母中使用原始最短路径会导致高效 agent 的 SPL 远低于 1.0。

因此 POI-Goal 从最短路径中减去到达阈值：

```
effective_shortest = max(shortest_path_length - arrive_threshold, 1e-3)
SPL = success * effective_shortest / max(travel_length, effective_shortest)
```

`1e-3` 下限防止最短路径短于阈值时除以零。

**为何需要此修正。** 若不修正，一个走近最优路径到达近距离 POI 的 agent（如 shortest = 3.0 m，travel = 1.5 m）将获得人为偏低的 SPL，因为 `shortest / max(travel, shortest)` = `3.0 / 3.0` = 1.0，掩盖了 agent 的高效性。修正后的公式给出 `max(3.0 - 2.0, 0.001) / max(1.5, 1.0)` = `1.0 / 1.5` = 0.667，正确反映了相对于最小所需距离的行走效率。

### 每任务结果字段

每个 `result.json` 包含：

| 字段 | 说明 |
|:-----|:-----|
| `success` | 终止时 `distance_to_goal <= arrive_threshold`（被碰撞终止时强制为 `false`） |
| `spl` | 使用修正公式计算的任务级 SPL |
| `status` | 终止原因：`"stop"`（agent 声明到达）、`"max_steps"`、`"collision"` 或 `"render_error"` |
| `steps` | 执行步数 |
| `travel_length` | 实际行走距离 |
| `shortest_path_length` | 真值最短路径长度 |
| `distance_to_goal` | 最终目标距离 |
| `metrics.poi_name` | 此任务的 POI 名称 |
| `metrics.collision_mode` | 活跃的碰撞模式（`off`/`soft`/`hard`） |
| `metrics.collision_count` | episode 中非豁免碰撞总次数 |
| `metrics.collision_rate` | `collision_count / steps` |
| `metrics.max_consecutive_collision` | 最长连续碰撞次数 |
| `metrics.collision_step_indices` | 发生碰撞的步骤索引列表 |
| `metrics.collision_skipped_near_goal` | 终端区内的碰撞次数（不计入） |
| `metrics.has_collision` | 布尔值：是否发生过非豁免碰撞 |
| `metrics.collision_terminated` | 布尔值：任务是否被 hard 模式碰撞终止 |
| `metrics.start_in_obstacle` | 健全性标志：起始位姿位于障碍物内 |
| `metrics.goal_in_obstacle` | 健全性标志：目标位姿位于障碍物内 |

### 各 POI 分解

启用 `--per-poi`（默认开启）后，分析输出中包含一个以 POI 名称为键的 `poi_stats` 字典。每个条目包含：

| 字段 | 说明 |
|:-----|:-----|
| `total` | 此 POI 的任务数 |
| `success` | 成功任务数 |
| `success_rate` | `success / total` |
| `avg_spl` | 此 POI 所有任务的平均 SPL |
| `collision_count` | 发生过至少一次碰撞的任务数 |
| `collision_terminated_count` | 被碰撞终止的任务数 |

此分解有助于识别对 agent 特别困难的 POI。

### 碰撞统计（汇总级）

分析输出中的汇总 `collision` 块包含：

| 字段 | 说明 |
|:-----|:-----|
| `tasks_evaluated` | 启用碰撞检测的任务数 |
| `tasks_with_collision` | 至少发生过一次碰撞的任务数 |
| `collision_task_rate` | `tasks_with_collision / tasks_evaluated` |
| `total_collision_steps` | 所有任务的碰撞步数总和 |
| `avg_collision_per_task` | `total_collision_steps / tasks_evaluated` |
| `avg_collision_rate` | 所有任务的平均每步碰撞率 |
| `collision_terminated_count` | 被 hard 模式碰撞终止的任务数 |

## 输出目录结构

```
results/poi/
└── 20260713_143022/
    ├── region_001/traj_1/
    │   ├── result.json
    │   └── render_images/
    ├── region_001/traj_2/
    │   ├── result.json
    │   └── render_images/
    ├── eval_summary.json          # 评测器在运行结束时写入
    └── poi_goal_analysis.json     # 由 analyze_and_report 写入
```

- `result.json`：每个任务的结果，包含上述所有指标字段。
- `render_images/`：启用 `--save-render-images` 时保存的渲染帧（默认启用）。
- `eval_summary.json`：评测器生成的汇总，包含各 POI 统计和碰撞汇总。
- `poi_goal_analysis.json`：由 `analyze_and_report` 生成的完整指标分析，包含分组和各 POI 分解。

