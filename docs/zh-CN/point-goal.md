# Point-Goal 评测

[English](../point-goal.md) | [中文](point-goal.md)

Point-Goal 导航任务中，agent 接收局部坐标系下的目标坐标，需导航至距目标 0.5 m 以内，同时避免碰撞。

## 协议参数

| 参数 | Outdoor | Indoor |
|:-----|:--------|:-------|
| `arrive_threshold` | 0.5 m | 0.5 m |
| `collision_threshold` | 3（允许至多 2 次碰撞） | 1（不允许碰撞） |
| `occ_dilation_meters` | 0.5 | 0.2 |
| 难度分组 | short (5-20 m) / medium (20-35 m) / long (35-50 m) | easy / hard |

协议参数为**固定基准标准**，修改后产出的结果不可比。这些参数编码在评测器内部的 `POINT_GOAL_PROTOCOL` 中，通过 `--mode outdoor` 或 `--mode indoor` 自动选择。

## 前提条件

- 已安装评测框架（`abotn-bench`）
- 渲染服务已启动（见[快速上手](getting-started.md)）
- 已下载 ABotN-PointBench 数据
- Python 3.8+，并安装 NumPy、SciPy、Pillow、tqdm 和 PyYAML

## 通过 Python API 评测

接口字段与坐标系详见 [API 参考](api-reference.md)。如需适配不同 I/O 约定的模型，见[自定义 Agent](custom-agents.md)。

### Outdoor

```python
import os
from datetime import datetime
from abotn_evaluator.point_goal.evaluator import PointGoalEvaluator, make_eval_config
from abotn_evaluator.scene import GaussianScene
from abotn_evaluator.render_client import GaussianRenderer
from abotn_evaluator.point_goal.metrics import analyze_and_report
from your_agent_module import YourAgent

RENDER_URL = "http://localhost:7036/render_gs"

scene = GaussianScene(
    local_data_path="/path/to/ABotN-PointBench/Outdoor/annotations",
    local_map_path="/path/to/ABotN-PointBench/Outdoor/occmaps",
)
renderer = GaussianRenderer(render_url=RENDER_URL)
config = make_eval_config(mode="outdoor", render_url=RENDER_URL, max_steps=100)

run_dir = os.path.join("./results/outdoor", datetime.now().strftime("%Y%m%d_%H%M%S"))
evaluator = PointGoalEvaluator(scene=scene, renderer=renderer, config=config, output_dir=run_dir)

results = evaluator.evaluate(YourAgent())
report = analyze_and_report(result_dir=run_dir, mode="outdoor")
```

`make_eval_config(mode=...)` 自动按上述标准表填充协议参数。除消融研究外，不应手动覆盖。

Python API 要点：

- `GaussianScene` 从数据目录加载所有 episode 和 task。`local_data_path` 指向轨迹数据；`local_map_path` 指向占据栅格地图/高度图。
- `GaussianRenderer` 封装渲染服务的 HTTP API，每个 agent 步骤都会调用渲染服务。
- `PointGoalEvaluator` 管理评测循环：构建观测、调用 `agent.predict(obs)`、检测碰撞、写入每个 task 的 `result.json`。
- `analyze_and_report` 读取所有 `result.json`，计算聚合指标，打印汇总表格，并保存 `eval_summary.json`。

### Indoor

相对 outdoor 示例修改三处：

1. 数据路径：`ABotN-PointBench/Indoor/annotations` 和 `ABotN-PointBench/Indoor/occmaps`
2. 配置：`make_eval_config(mode="indoor", ...)`
3. 指标：`analyze_and_report(result_dir=..., mode="indoor")`

## 通过 CLI 评测

### 基本用法

```bash
# Outdoor
python -m abotn_evaluator.point_goal.runner \
    --agent-module your_agent_module:YourAgent \
    --data-dir /path/to/ABotN-PointBench/Outdoor/annotations \
    --map-dir /path/to/ABotN-PointBench/Outdoor/occmaps \
    --render-url http://localhost:7036/render_gs \
    --output-dir ./results/outdoor \
    --mode outdoor

# Indoor
python -m abotn_evaluator.point_goal.runner \
    --agent-module your_agent_module:YourAgent \
    --data-dir /path/to/ABotN-PointBench/Indoor/annotations \
    --map-dir /path/to/ABotN-PointBench/Indoor/occmaps \
    --render-url http://localhost:7036/render_gs \
    --output-dir ./results/indoor \
    --mode indoor
```

`--mode` 自动选择协议参数。Agent 构造参数可通过 `--agent-config your.yaml` 传递（YAML 键值作为 `**kwargs` 传入 `__init__`）。

### 完整参数说明

所有 CLI 参数按类别分组列出如下。

#### 基本参数

| 参数 | 类型 | 默认值 | 说明 |
|:-----|:-----|:-------|:-----|
| `--agent-module` | str | *必填* | Agent 类路径，格式为 `package.module:ClassName`。模块必须在当前 Python 环境中可导入，类需实现 `predict(obs)` 和 `reset()` 方法。示例：`my_agents.point_nav:MyAgent`。 |
| `--agent-config` | str | `None` | YAML 配置文件路径，其顶层键值对作为 `**kwargs` 传入 agent 类构造函数。支持 `_base_` 继承和基于 `mode` 的子字典解析。 |
| `--data-dir` | str | *必填* | 包含各场景子目录的根目录，存放轨迹数据（起止位姿、参考路径）。每个子目录对应一个 episode。 |
| `--map-dir` | str | `None` | 地图数据的独立目录（占据栅格地图、高度图、元数据）。若省略，评测器将在 `--data-dir` 中查找地图数据。 |
| `--render-url` | str | `http://127.0.0.1:7001/render_gs` | 高斯渲染服务的完整 URL。渲染服务必须在该地址运行且可访问。 |
| `--output-dir` | str | `./eval_output` | 评测输出的根目录。每次新运行会自动在此目录下创建带时间戳的子目录（如 `20260713_143022`）。 |
| `--resume-dir` | str | `None` | 上次运行的输出目录路径，用于断点续跑。运行器扫描已有的 `result.json` 文件并跳过对应的 (episode, task) 对。省略时将在 `--output-dir` 下创建新的时间戳子目录。 |
| `--evaluator-module` | str | `None` | 自定义评测器类路径，格式为 `package.module:ClassName`。类构造函数须接受 `(scene, renderer, config, output_dir)` 参数。省略时使用内置的 `PointGoalEvaluator`。 |
| `--max-steps` | int | `100` | 每个 task 的最大 agent 步数。若 agent 在该步数内未到达目标或未声明 `arrive=True`，task 以 `status: "max_steps"` 结束。 |

#### 协议参数

这些参数控制基准协议，由 `--mode` 自动设置。正式评测提交时不应修改。

| 参数 | 类型 | 默认值 | 说明 |
|:-----|:-----|:-------|:-----|
| `--mode` | str | `outdoor` | 评测模式：`outdoor` 或 `indoor`。自动设置 `arrive_threshold`、`collision_threshold` 和 `occ_dilation_meters` 为标准协议值，同时决定指标的难度分类方案。 |
| `--arrive-threshold` | float | 按模式自动设置 | 目标到达距离阈值（米）。agent 最终距目标在此范围内视为成功。两种模式均自动设为 0.5 m。显式指定将覆盖模式默认值。 |
| `--collision-threshold` | int | 按模式自动设置 | `SR_NEW` 判定中允许的最大路径碰撞次数。outdoor 自动设为 3（容许至多 2 次碰撞），indoor 自动设为 1（不容许碰撞）。显式指定将覆盖模式默认值。 |
| `--occ-dilation-meters` | float | 按模式自动设置 | 对占据栅格地图中的自由空间进行膨胀的距离（米），等效于将障碍物缩小对应距离。outdoor 自动设为 0.5，indoor 自动设为 0.2。显式指定将覆盖模式默认值。 |

#### 相机参数

配置用于渲染观测的虚拟相机。

| 参数 | 类型 | 默认值 | 说明 |
|:-----|:-----|:-------|:-----|
| `--camera-width` | int | `720` | 渲染图像宽度（像素）。 |
| `--camera-height` | int | `640` | 渲染图像高度（像素）。 |
| `--camera-fx` | float | `252.075` | 相机水平方向焦距（像素）。 |
| `--camera-fy` | float | `252.075` | 相机垂直方向焦距（像素）。 |
| `--extrinsic-height` | float | `0.65` | 相机离地高度（米）。此值加上地面高度（从高度图查询）得到相机的 Z 坐标。 |

#### 历史/观测参数

控制传递给 agent 的观测中包含哪些额外信息。

| 参数 | 类型 | 默认值 | 说明 |
|:-----|:-----|:-------|:-----|
| `--provide-history` | 标志 | `False` | 启用后，每个观测中包含 `history_images`（历史渲染帧）和 `history_poses`（历史 4x4 位姿矩阵）。适用于利用时序上下文的 agent。 |
| `--provide-occ-map` | 标志 | `False` | 启用后，每个观测中包含当前 episode 的占据栅格地图（NumPy 数组）。地图为 2D 网格，编码自由空间和障碍物。 |
| `--provide-height-map` | 标志 | `False` | 启用后，每个观测中包含当前 episode 的高度图（NumPy 数组）。高度图提供场景各处的地面高程值。 |
| `--max-history-frames` | int | `20` | `ShortMemory` 缓冲区保留的最大历史帧数。缓冲满时丢弃最旧的帧。仅在 `--provide-history` 启用时生效。 |
| `--history-resize-ratio` | float | `0.25` | 历史帧图像在存储前的缩放比例。0.25 表示缩小到原始分辨率的 25%。在保留大量历史帧时可减少内存占用。 |

#### 输出参数

| 参数 | 类型 | 默认值 | 说明 |
|:-----|:-----|:-------|:-----|
| `--save-render-images` | 标志 | `True` | 将渲染图像保存到每个 task 的 `render_images/` 子目录。默认启用。 |
| `--no-save-render-images` | 标志 | （设置 `save_render_images=False`） | 禁用渲染图像保存。在仅需指标而无需可视化检查 agent 轨迹时，可加速评测。 |
| `--enable-visualization` | 标志 | `False` | 启用后，处理 agent 预测结果中的 `prediction.extra` 字段以生成可视化输出（如在渲染图像上叠加可通行性像素标注）。要求 agent 在预测中填充 `extra` 字段。 |
| `--output-dir` | str | `./eval_output` | 所有评测输出的根目录（同时列于基本参数中）。 |

#### 指标参数

控制评测后的指标分析步骤。

| 参数 | 类型 | 默认值 | 说明 |
|:-----|:-----|:-------|:-----|
| `--mode` | str | `outdoor` | 决定难度分类方案。outdoor 使用基于距离的分桶（short/medium/long）；indoor 使用基于场景列表的分组（easy/hard）。（同时列于协议参数中。） |
| `--min-distance` | float | `5.0` | 仅 outdoor：难度分桶的最小最短路径长度（米）。`shortest_path_length < 5.0` 的 task 归入 `out_of_range`。 |
| `--max-distance` | float | `50.0` | 仅 outdoor：难度分桶的最大最短路径长度（米）。与 `--min-distance` 配合，定义三个等宽分桶：short [5, 20)、medium [20, 35)、long [35, 50]。 |
| `--exclude-scenes` | str 列表 | `None` | 空格分隔的场景（episode）ID 列表，从指标计算中排除。省略时 outdoor 模式自动排除 `park3`；indoor 模式不排除任何场景。 |
| `--per-scene` | 标志 | `False` | 启用后，打印按场景分组的指标明细表，展示每个场景的成功率、SR_NEW、SPL_NEW、TCR_NEW、DCR_NEW 和平均最短路径长度。 |
| `--skip-metrics` | 标志 | `False` | 完全跳过评测后的指标分析步骤。启用后仅写入各 task 的 `result.json`，不生成聚合的 `eval_summary.json`。适用于后续统一聚合的部分评测场景。 |

## 评测工作流

从数据准备到结果分析的完整评测流程如下：

### 第 1 步：确认数据目录结构

数据目录应具有以下布局：

```
ABotN-PointBench/Outdoor/
  annotations/
    cross1/
      traj_0.json            # 起止位姿、参考路径、task 元数据
      traj_1.json
      png/                   # 标注图像
    park1/
      ...
  occmaps/
    cross1/
      map/
        occ_map.png          # 占据栅格地图
        occ_map_height.tiff  # 高度图
        occ_map_meta.txt     # 坐标系元数据
    park1/
      ...
```

`annotations/` 下的每个场景目录包含轨迹 JSON 文件（`traj_*.json`），每个文件包含起始位姿、终止位姿（目标点）和参考轨迹。`occmaps/` 目录包含各场景对应的占据栅格地图和高度图。

### 第 2 步：确认渲染服务

启动评测前，确认渲染服务已运行且可访问：

```bash
curl http://localhost:7036/ping
```

收到成功响应（如 `"pong"` 或 HTTP 200）说明服务就绪。若命令卡住或返回连接错误，请先启动渲染服务（见[快速上手](getting-started.md)）。

### 第 3 步：运行评测

```bash
python -m abotn_evaluator.point_goal.runner \
    --agent-module your_agent_module:YourAgent \
    --data-dir /path/to/ABotN-PointBench/Outdoor/annotations \
    --map-dir /path/to/ABotN-PointBench/Outdoor/occmaps \
    --render-url http://localhost:7036/render_gs \
    --output-dir ./results/outdoor \
    --mode outdoor
```

运行器将：

1. 导入并实例化 agent 类。
2. 创建带时间戳的输出子目录（如 `./results/outdoor/20260713_143022/`）。
3. 从数据目录加载所有场景和 task。
4. 根据协议对占据栅格地图进行膨胀。
5. 遍历每个 episode 和 task，在每一步调用 `agent.predict(obs)`。
6. 使用占据栅格地图检测连续位姿之间的碰撞。
7. 为每个完成的 task 写入 `result.json`。
8. 运行指标分析并保存 `eval_summary.json`（除非设置了 `--skip-metrics`）。

### 第 4 步：查看结果

评测完成后，检查输出：

- **逐 task 结果**：`results/outdoor/20260713_143022/scene_001/traj_1/result.json`
- **聚合摘要**：`results/outdoor/20260713_143022/eval_summary.json`
- **渲染图像**：`results/outdoor/20260713_143022/scene_001/traj_1/render_images/`（启用 `--save-render-images` 时）

终端输出还会打印难度分组对比表和各组详细指标。

## 评测过程中查看指标

指标分析模块可以在任何时间独立运行——即使评测仍在进行中。它读取已写入的所有 `result.json` 文件，从已完成的 task 计算阶段性聚合指标。

```bash
python -m abotn_evaluator.point_goal.metrics \
    --result-dir ./results/outdoor/20260713_143022 \
    --mode outdoor
```

适用场景：

- **监控进度**：查看随着更多 task 完成，指标如何变化。
- **提前终止**：如果在大量 task 完成后指标表现不佳，可取消评测并迭代优化 agent。
- **事后重新分析**：无需重新评测，即可使用不同的 `--exclude-scenes` 或 `--per-scene` 选项重新分析。

### 独立指标命令参数

独立指标命令接受以下参数：

| 参数 | 类型 | 默认值 | 说明 |
|:-----|:-----|:-------|:-----|
| `--result-dir` | str | *必填* | 扫描 `result.json` 文件的评测输出目录。 |
| `--mode` | str | `outdoor` | 难度分类模式（`outdoor` 或 `indoor`）。 |
| `--collision-threshold` | int | `3` | SR_NEW 计算中的碰撞次数阈值。 |
| `--min-distance` | float | `5.0` | outdoor 分桶的最小最短路径长度。 |
| `--max-distance` | float | `50.0` | outdoor 分桶的最大最短路径长度。 |
| `--exclude-scenes` | str 列表 | `None` | 要排除的场景 ID。outdoor 默认排除 `["park3"]`。 |
| `--output-path` | str | `None` | 自定义输出 JSON 路径。默认为 `{result-dir}/eval_summary.json`。 |
| `--per-scene` | 标志 | `False` | 打印按场景分组的指标明细。 |

### 示例输出

```
[./results/outdoor/20260713_143022] Loaded 156 result.json files
Excluded scenes ['park3']: filtered 12 tasks, 144 remaining

Outdoor difficulty distribution (range [5.0, 50.0]m, step=15.0m):
     short: [5.0, 20.0) -> 48 tasks
    medium: [20.0, 35.0) -> 52 tasks
      long: [35.0, 50.0) -> 44 tasks

====================================================================
Difficulty group comparison  collision_threshold=3
====================================================================
metric                                  short        medium          long       overall
--------------------------------------------------------------------
success_rate                           0.8750        0.7500        0.6136        0.7431
oracle_success_rate                    0.8750        0.7500        0.6136        0.7431
sr_new                                 0.8333        0.6923        0.5227        0.6736
spl_new                                0.7215        0.5680        0.4012        0.5587
tcr_new                                0.8102        0.6650        0.5010        0.6504
dcr_new                                0.8230        0.6788        0.5123        0.6628
avg_steps                             28.5000       42.3000       55.8000       42.2000
avg_travel_length                     12.4500       25.6700       40.1200       26.0800
avg_shortest_path_length              12.1000       27.2000       41.5000       26.9333
avg_final_distance                     1.2300        2.5600        4.8900        2.8933
--------------------------------------------------------------------
count                                      48            52            44           144
====================================================================

Saved eval summary JSON: ./results/outdoor/20260713_143022/eval_summary.json
```

## 断点续跑与多 GPU

### 从上次运行续跑

重复执行相同命令会自动跳过已完成的任务（已存在 `result.json` 的 task）。显式从上次运行续跑：

```bash
python -m abotn_evaluator.point_goal.runner \
    --agent-module your_agent_module:YourAgent \
    --data-dir /path/to/ABotN-PointBench/Outdoor/annotations \
    --map-dir /path/to/ABotN-PointBench/Outdoor/occmaps \
    --render-url http://localhost:7036/render_gs \
    --output-dir ./results/outdoor \
    --resume-dir ./results/outdoor/20260713_143022 \
    --mode outdoor
```

指定 `--resume-dir` 后：

1. 运行器扫描 `{resume-dir}/**/result.json`，获取已完成的 (episode_id, task_id) 对。
2. 评测过程中跳过这些 task。
3. 新结果写入同一目录（`--resume-dir` 路径成为实际输出目录）。
4. 终端打印 `[resume] Found N completed tasks in <dir>`。

### 多 GPU 评测

多 GPU 评测：启动多个进程，各自指定不同的 `CUDA_VISIBLE_DEVICES`，共享同一个 `--output-dir`。基于文件的续跑机制提供天然的去重能力：

```bash
# 终端 1
CUDA_VISIBLE_DEVICES=0 python -m abotn_evaluator.point_goal.runner \
    --agent-module your_agent_module:YourAgent \
    --data-dir /path/to/data --map-dir /path/to/maps \
    --render-url http://localhost:7036/render_gs \
    --output-dir ./results/outdoor \
    --resume-dir ./results/outdoor/shared_run \
    --mode outdoor

# 终端 2
CUDA_VISIBLE_DEVICES=1 python -m abotn_evaluator.point_goal.runner \
    --agent-module your_agent_module:YourAgent \
    --data-dir /path/to/data --map-dir /path/to/maps \
    --render-url http://localhost:7037/render_gs \
    --output-dir ./results/outdoor \
    --resume-dir ./results/outdoor/shared_run \
    --mode outdoor
```

每个进程在开始 task 前检查是否已存在 `result.json`。在边界情况下可能有少量重复工作（两个进程几乎同时启动同一 task），但最终结果一致，因为每个 task 的 `result.json` 是原子写入的。

注意：每个 GPU 进程应指向独立的渲染服务实例（不同端口），或共享一个支持并发请求的渲染服务。

## 指标公式

所有指标在 `result.json` 中逐 task 计算，然后跨 task 聚合。关键的碰撞感知指标如下：

### SR_NEW（碰撞感知成功率）

```
SR_NEW = success AND (path_collision_count < collision_threshold)
```

- `success`：agent 最终距目标在 `arrive_threshold`（0.5 m）以内。
- `path_collision_count`：agent 路径段与占据栅格地图中障碍物相交的总步数。
- `collision_threshold`：outdoor 为 3（容许至多 2 次碰撞），indoor 为 1（不容许碰撞）。

### SPL_NEW（碰撞感知的路径长度加权成功率）

```
SPL_NEW = SR_NEW * shortest_path_length / max(travel_length, shortest_path_length)
```

- 仅当 `SR_NEW = true` 时非零。
- 惩罚即使成功但路径更长的 agent。
- `shortest_path_length`：参考（真值）路径长度。
- `travel_length`：agent 实际行走的路径长度。

### TCR_NEW（时间碰撞比）

```
TCR_NEW = (total_steps - collision_steps) / total_steps
```

- **仅当 SR_NEW = true 时计算**；否则 TCR_NEW = 0。
- 衡量无碰撞步骤占总步骤的比例。
- `collision_steps` = 检测到碰撞的步数（即 `path_collision_count`）。
- 越接近 1.0 表示碰撞事件越少。

### DCR_NEW（距离碰撞比）

```
DCR_NEW = (total_distance - collision_path_length) / total_distance
```

- **仅当 SR_NEW = true 时计算**；否则 DCR_NEW = 0。
- 衡量无碰撞行走距离占总行走距离的比例。
- `collision_path_length`：与障碍物重叠的路径段总长度（以 0.05 m 间隔采样）。
- `total_distance`：所有路径段的累计距离。
- 越接近 1.0 表示在障碍物内部行走的距离越少。

### 难度分类

**Outdoor** 任务按 `shortest_path_length` 分为三个等宽分桶，范围 [5, 50] m：

| 难度 | 范围 |
|:-----|:-----|
| short | [5, 20) m |
| medium | [20, 35) m |
| long | [35, 50] m（上界扩展至 75 m 以捕获边界 task） |

`shortest_path_length` 不在 [5, 75] m 范围内的 task 归类为 `out_of_range`。

**Indoor** 任务按固定场景列表分类：

| 难度 | 场景 |
|:-----|:-----|
| easy | 0802_840243, 0803_840265, 0821_841631, 0822_841630, 0827_841619, 0841_841759, 0845_841765, 0854_841775 |
| hard | 0813_841249, 0814_841252, 0832_840249, 0833_840508, 0837_841153, 0843_841761, 0847_841768, 0861_841783 |

不在任一列表中的场景归类为 `unknown`。

### 排除场景

默认情况下，outdoor 评测从指标计算中排除场景 `park3`。可通过 `--exclude-scenes` 覆盖：

```bash
# 包含所有场景（不排除）
python -m abotn_evaluator.point_goal.metrics --result-dir ./results --mode outdoor --exclude-scenes

# 排除指定场景
python -m abotn_evaluator.point_goal.metrics --result-dir ./results --mode outdoor --exclude-scenes park3 park7
```

### 聚合方式

聚合指标（在 `eval_summary.json` 中）的计算方式：

- **success_rate、sr_new、spl_new、tcr_new、dcr_new**：组内所有 task 的算术平均值。
- **global_collision_path_ratio**：所有 task 的碰撞路径长度之和除以总距离之和（全局比率，而非逐 task 比率的平均值）。
- **分组**：指标分别按难度组（short/medium/long 或 easy/hard）和整体（overall）独立计算。

## 输出目录结构

```
results/outdoor/
  20260713_143022/
    scene_001/
      traj_1/
        result.json              # 逐 task 评测结果
        render_images/           # save_render_images=True 时
          0_left.jpg
          0_front.jpg
          0_right.jpg
          1_left.jpg
          1_front.jpg
          1_right.jpg
          ...
      traj_2/
        result.json
        render_images/
    scene_002/
      ...
    eval_summary.json            # 聚合指标（由 analyze_and_report 生成）
```

### result.json 字段说明

每个 `result.json` 包含以下字段：

| 字段 | 类型 | 说明 |
|:-----|:-----|:-----|
| `episode_id` | str | 场景标识 |
| `task_id` | str | 场景内的 task 标识 |
| `status` | str | 终止原因：`"stop"`（agent 声明到达）、`"max_steps"`、`"error"` 或 `"render_error"` |
| `steps` | int | 执行步数 |
| `travel_length` | float | 实际行走路径长度（米） |
| `shortest_path_length` | float | 参考路径长度（米） |
| `distance_to_goal` | float | 最终距目标距离（米） |
| `success` | bool | `distance_to_goal <= arrive_threshold` 是否成立 |
| `oracle_success` | bool | 当前实现中与 `success` 相同 |
| `metrics.initial_distance_to_goal` | float | 起始位姿到目标的距离 |
| `metrics.min_distance_to_goal` | float | 任意步骤中达到的最小距目标距离 |
| `metrics.final_distance_to_goal` | float | 最后一步的距目标距离 |
| `metrics.success_new` | bool | 该 task 的 SR_NEW |
| `metrics.spl_new` | float | 该 task 的 SPL_NEW |
| `metrics.tcr_new` | float | 该 task 的 TCR_NEW |
| `metrics.dcr_new` | float | 该 task 的 DCR_NEW |
| `metrics.path_collision_count` | int | 路径上的碰撞事件总数 |
| `metrics.collision_path_length` | float | 路径段在障碍物内部的总长度（米） |
| `metrics.total_distance` | float | 所有路径段的累计距离（米） |
| `metrics.is_path_collided` | bool | 是否发生过碰撞 |
| `metrics.collision_path_ratio` | float | `collision_path_length / total_distance` |
| `metrics.collision_steps` | list[int] | 发生碰撞的步骤索引 |

