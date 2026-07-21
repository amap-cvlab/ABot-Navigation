# Point-Goal Evaluation

[English](point-goal.md) | [中文](zh-CN/point-goal.md)

In Point-Goal navigation, the agent receives target coordinates in its local frame and must navigate to within 0.5 m while avoiding collisions.

## Protocol

| Parameter | Outdoor | Indoor |
|:----------|:--------|:-------|
| `arrive_threshold` | 0.5 m | 0.5 m |
| `collision_threshold` | 3 (up to 2 collisions allowed) | 1 (zero collisions allowed) |
| `occ_dilation_meters` | 0.5 | 0.2 |
| Difficulty grouping | short (5-20 m) / medium (20-35 m) / long (35-50 m) | easy / hard |

These protocol parameters are **fixed benchmark standards**. Changing them produces non-comparable results. They are codified in `POINT_GOAL_PROTOCOL` inside the evaluator and are auto-selected when you pass `--mode outdoor` or `--mode indoor`.

## Prerequisites

- Evaluator installed (`abotn-bench`)
- Render server running (see [Getting Started](getting-started.md))
- ABotN-PointBench data downloaded
- Python 3.8+ with NumPy, SciPy, Pillow, tqdm, and PyYAML

## Evaluation via Python API

For complete interface field definitions and coordinate system details, see [API Reference](api-reference.md). For adapting models with different I/O conventions, see [Custom Agents](custom-agents.md).

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

`make_eval_config(mode=...)` automatically fills protocol parameters from the standard table above. Do not override them unless conducting ablation studies.

Key points about the Python API:

- `GaussianScene` loads all episodes and tasks from the data directory. `local_data_path` points to trajectory data; `local_map_path` points to occupancy/height maps.
- `GaussianRenderer` wraps the render server HTTP API. It calls the render service for each agent step.
- `PointGoalEvaluator` manages the evaluation loop: it builds observations, calls `agent.predict(obs)`, detects collisions, and writes per-task `result.json` files.
- `analyze_and_report` reads all `result.json` files, computes aggregated metrics, prints summary tables, and saves `eval_summary.json`.

### Indoor

Change three things from the outdoor example:

1. Data paths: `ABotN-PointBench/Indoor/annotations` and `ABotN-PointBench/Indoor/occmaps`
2. Config: `make_eval_config(mode="indoor", ...)`
3. Metrics: `analyze_and_report(result_dir=..., mode="indoor")`

## Evaluation via CLI

### Basic Usage

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

`--mode` selects protocol parameters automatically. Pass agent constructor arguments via `--agent-config your.yaml` (YAML keys become `**kwargs` to `__init__`).

### Full Parameter Reference

All CLI parameters are documented below, grouped by category.

#### Basic Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `--agent-module` | str | *required* | Agent class path in `package.module:ClassName` format. The module must be importable from the current Python environment. The class must implement `predict(obs)` and `reset()`. Example: `my_agents.point_nav:MyAgent`. |
| `--agent-config` | str | `None` | Path to a YAML file whose top-level keys are passed as `**kwargs` to the agent class constructor. Supports `_base_` inheritance and `mode`-based sub-dict resolution. |
| `--data-dir` | str | *required* | Root directory containing per-scene subdirectories with trajectory data (start/end poses, reference paths). Each subdirectory is treated as one episode. |
| `--map-dir` | str | `None` | Separate directory for map data (occupancy maps, height maps, metadata). If omitted, the evaluator looks for map data alongside trajectory data in `--data-dir`. |
| `--render-url` | str | `http://127.0.0.1:7001/render_gs` | Full URL of the Gaussian splatting render service endpoint. The render server must be running and accessible at this URL. |
| `--output-dir` | str | `./eval_output` | Base directory for evaluation outputs. A timestamped subdirectory (e.g., `20260713_143022`) is created automatically under this path for each new run. |
| `--resume-dir` | str | `None` | Path to a previous run's output directory to resume from. The runner scans for existing `result.json` files and skips those (episode, task) pairs. When omitted, a new timestamped subdirectory is created under `--output-dir`. |
| `--evaluator-module` | str | `None` | Custom evaluator class path in `package.module:ClassName` format. The class must accept `(scene, renderer, config, output_dir)` in its constructor. When omitted, the built-in `PointGoalEvaluator` is used. |
| `--max-steps` | int | `100` | Maximum number of agent steps per task before forced termination. If the agent has not reached the goal or called `arrive=True` within this many steps, the task ends with `status: "max_steps"`. |

#### Protocol Parameters

These parameters control the benchmark protocol. They are auto-selected by `--mode` and should not be changed for official benchmark submissions.

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `--mode` | str | `outdoor` | Evaluation mode: `outdoor` or `indoor`. Automatically sets `arrive_threshold`, `collision_threshold`, and `occ_dilation_meters` to the standard protocol values. Also determines the difficulty classification scheme used for metrics. |
| `--arrive-threshold` | float | Auto by mode | Goal arrival distance threshold in metres. The agent is considered successful if its final distance to the goal is within this value. Auto-selected: 0.5 m for both outdoor and indoor. Explicitly setting this overrides the mode default. |
| `--collision-threshold` | int | Auto by mode | Maximum number of path collisions allowed for `SR_NEW` to be true. Auto-selected: 3 for outdoor (up to 2 collisions tolerated), 1 for indoor (zero collisions tolerated). Explicitly setting this overrides the mode default. |
| `--occ-dilation-meters` | float | Auto by mode | Dilates free space in the occupancy map by this distance in metres. Effectively shrinks obstacles by the dilation amount. Auto-selected: 0.5 for outdoor, 0.2 for indoor. Explicitly setting this overrides the mode default. |

#### Camera Parameters

These configure the virtual camera used for rendering observations.

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `--camera-width` | int | `720` | Width of the rendered camera image in pixels. |
| `--camera-height` | int | `640` | Height of the rendered camera image in pixels. |
| `--camera-fx` | float | `252.075` | Camera focal length along the x-axis (horizontal) in pixels. |
| `--camera-fy` | float | `252.075` | Camera focal length along the y-axis (vertical) in pixels. |
| `--extrinsic-height` | float | `0.65` | Height of the camera above the ground plane in metres. This value is added to the ground height (looked up from the height map) to compute the camera's Z position. |

#### History / Observation Parameters

These control what additional information is included in the observation passed to the agent.

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `--provide-history` | flag | `False` | When set, includes `history_images` (past rendered frames) and `history_poses` (past 4x4 pose matrices) in each observation. Useful for agents that leverage temporal context. |
| `--provide-occ-map` | flag | `False` | When set, includes the episode's occupancy map as a NumPy array in each observation. The map is a 2D grid where free space and obstacles are encoded. |
| `--provide-height-map` | flag | `False` | When set, includes the episode's height map as a NumPy array in each observation. The height map provides ground elevation values across the scene. |
| `--max-history-frames` | int | `20` | Maximum number of past frames retained in the `ShortMemory` buffer. When the buffer is full, the oldest frame is discarded. Only relevant when `--provide-history` is set. |
| `--history-resize-ratio` | float | `0.25` | Downscale ratio applied to history frame images before storing them. A value of 0.25 means history images are reduced to 25% of the original resolution. Reduces memory usage when keeping many history frames. |

#### Output Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `--save-render-images` | flag | `True` | Save rendered images to disk in a `render_images/` subdirectory for each task. This is the default behavior. |
| `--no-save-render-images` | flag | (sets `save_render_images=False`) | Disable saving rendered images. Useful for faster evaluation when you only need metrics and do not need to visually inspect the agent's trajectory. |
| `--enable-visualization` | flag | `False` | When set, processes the `prediction.extra` field from the agent for visualization outputs (e.g., affordance pixel overlay on rendered images). Requires the agent to populate the `extra` field in its prediction. |
| `--output-dir` | str | `./eval_output` | Base directory for all evaluation outputs (also listed under Basic Parameters above). |

#### Metrics Parameters

These control the post-evaluation metrics analysis step.

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `--mode` | str | `outdoor` | Determines the difficulty classification scheme. Outdoor uses distance-based buckets (short/medium/long). Indoor uses scene-list-based groups (easy/hard). (Also listed under Protocol Parameters.) |
| `--min-distance` | float | `5.0` | Outdoor only: minimum shortest-path length (in metres) for difficulty bucketing. Tasks with `shortest_path_length < 5.0` are placed in `out_of_range`. |
| `--max-distance` | float | `50.0` | Outdoor only: maximum shortest-path length (in metres) for difficulty bucketing. Combined with `--min-distance`, defines three equal-width buckets: short [5, 20), medium [20, 35), long [35, 50]. |
| `--exclude-scenes` | str list | `None` | Space-separated list of scene (episode) IDs to exclude from metrics computation. When omitted, outdoor mode automatically excludes `park3`; indoor mode excludes nothing. |
| `--per-scene` | flag | `False` | When set, prints a per-scene metrics breakdown table showing success rate, SR_NEW, SPL_NEW, TCR_NEW, DCR_NEW, and average shortest path length for each scene. |
| `--skip-metrics` | flag | `False` | Skip the post-evaluation metrics analysis step entirely. When set, only per-task `result.json` files are written; no aggregated `eval_summary.json` is generated. Useful when running partial evaluations that will be aggregated later. |

## Evaluation Workflow

A complete evaluation from data preparation to result analysis follows these steps:

### Step 1: Verify Data Directory Structure

The data directory should have the following layout:

```
ABotN-PointBench/Outdoor/
  annotations/
    cross1/
      traj_0.json            # start/end poses, reference path, task metadata
      traj_1.json
      png/                   # annotation images
    park1/
      ...
  occmaps/
    cross1/
      map/
        occ_map.png          # occupancy map
        occ_map_height.tiff  # height map
        occ_map_meta.txt     # coordinate system metadata
    park1/
      ...
```

Each scene directory under `annotations/` contains trajectory JSON files (`traj_*.json`) with start pose, end pose (goal), and the reference trajectory. The `occmaps/` directory contains the corresponding occupancy and height maps for each scene.

### Step 2: Verify the Render Server

Before starting evaluation, confirm the render server is running and reachable:

```bash
curl http://localhost:7036/ping
```

A successful response (e.g., `"pong"` or HTTP 200) confirms the server is ready. If the command hangs or returns a connection error, start the render server first (see [Getting Started](getting-started.md)).

### Step 3: Run the Evaluation

```bash
python -m abotn_evaluator.point_goal.runner \
    --agent-module your_agent_module:YourAgent \
    --data-dir /path/to/ABotN-PointBench/Outdoor/annotations \
    --map-dir /path/to/ABotN-PointBench/Outdoor/occmaps \
    --render-url http://localhost:7036/render_gs \
    --output-dir ./results/outdoor \
    --mode outdoor
```

The runner will:

1. Import and instantiate your agent class.
2. Create a timestamped output subdirectory (e.g., `./results/outdoor/20260713_143022/`).
3. Load all scenes and tasks from the data directory.
4. Apply occupancy map dilation based on the protocol.
5. Iterate over every episode and task, calling `agent.predict(obs)` at each step.
6. Detect collisions between consecutive poses using the occupancy map.
7. Write a `result.json` for each completed task.
8. Run metrics analysis and save `eval_summary.json` (unless `--skip-metrics` is set).

### Step 4: Review Results

After evaluation completes, examine the output:

- **Per-task results**: `results/outdoor/20260713_143022/scene_001/traj_1/result.json`
- **Aggregated summary**: `results/outdoor/20260713_143022/eval_summary.json`
- **Rendered images**: `results/outdoor/20260713_143022/scene_001/traj_1/render_images/` (if `--save-render-images` is enabled)

The terminal output also prints a difficulty-group comparison table and per-group detailed metrics.

## Check Metrics During Evaluation

The metrics analysis module can be run independently at any time -- even while evaluation is still in progress. It reads all `result.json` files that have been written so far and computes interim aggregate metrics from completed tasks.

```bash
python -m abotn_evaluator.point_goal.metrics \
    --result-dir ./results/outdoor/20260713_143022 \
    --mode outdoor
```

This is useful for:

- **Monitoring progress**: check how metrics evolve as more tasks complete.
- **Early stopping**: if metrics look poor after a significant number of tasks, you can cancel and iterate on the agent.
- **Post-hoc re-analysis**: re-run with different `--exclude-scenes` or `--per-scene` options without re-evaluating.

### Standalone Metrics Parameters

The standalone metrics command accepts the following parameters:

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `--result-dir` | str | *required* | Evaluation output directory to scan for `result.json` files. |
| `--mode` | str | `outdoor` | Difficulty classification mode (`outdoor` or `indoor`). |
| `--collision-threshold` | int | `3` | Collision-count threshold for SR_NEW computation. |
| `--min-distance` | float | `5.0` | Minimum shortest-path length for outdoor bucketing. |
| `--max-distance` | float | `50.0` | Maximum shortest-path length for outdoor bucketing. |
| `--exclude-scenes` | str list | `None` | Scene IDs to exclude. Defaults to `["park3"]` for outdoor. |
| `--output-path` | str | `None` | Custom path for the output JSON. Defaults to `{result-dir}/eval_summary.json`. |
| `--per-scene` | flag | `False` | Print per-scene metrics breakdown. |

### Example Output

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

## Resume and Multi-GPU

### Resume from a Previous Run

Re-running the same command automatically skips completed tasks (those with an existing `result.json`). To explicitly resume from a previous run:

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

When `--resume-dir` is specified:

1. The runner scans `{resume-dir}/**/result.json` for completed (episode_id, task_id) pairs.
2. Those pairs are skipped during evaluation.
3. New results are written to the same directory (the `--resume-dir` path becomes the effective output directory).
4. The terminal prints `[resume] Found N completed tasks in <dir>`.

### Multi-GPU Evaluation

For multi-GPU evaluation, launch multiple processes with different `CUDA_VISIBLE_DEVICES`, all pointing to the same `--output-dir`. The file-based resume mechanism provides natural deduplication:

```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 python -m abotn_evaluator.point_goal.runner \
    --agent-module your_agent_module:YourAgent \
    --data-dir /path/to/data --map-dir /path/to/maps \
    --render-url http://localhost:7036/render_gs \
    --output-dir ./results/outdoor \
    --resume-dir ./results/outdoor/shared_run \
    --mode outdoor

# Terminal 2
CUDA_VISIBLE_DEVICES=1 python -m abotn_evaluator.point_goal.runner \
    --agent-module your_agent_module:YourAgent \
    --data-dir /path/to/data --map-dir /path/to/maps \
    --render-url http://localhost:7037/render_gs \
    --output-dir ./results/outdoor \
    --resume-dir ./results/outdoor/shared_run \
    --mode outdoor
```

Each process checks for existing `result.json` files before starting a task. There may be a small amount of duplicate work at the boundary (if two processes start the same task nearly simultaneously), but the final results are consistent because each task's `result.json` is written atomically.

Note: each GPU process should point to its own render server instance (different ports), or share a render server that supports concurrent requests.

## Metrics Formulas

All metrics are computed per-task in `result.json` and then aggregated across tasks. The key collision-aware metrics are:

### SR_NEW (Success Rate, collision-aware)

```
SR_NEW = success AND (path_collision_count < collision_threshold)
```

- `success`: the agent's final distance to the goal is within `arrive_threshold` (0.5 m).
- `path_collision_count`: total number of steps where the agent's path segment intersected an obstacle in the occupancy map.
- `collision_threshold`: 3 for outdoor (tolerates up to 2 collisions), 1 for indoor (tolerates zero collisions).

### SPL_NEW (Success weighted by Path Length, collision-aware)

```
SPL_NEW = SR_NEW * shortest_path_length / max(travel_length, shortest_path_length)
```

- Only nonzero when `SR_NEW = true`.
- Penalises agents that take longer paths even when they succeed.
- `shortest_path_length`: the reference (ground truth) path length.
- `travel_length`: the agent's actual traversed path length.

### TCR_NEW (Temporal Collision Ratio)

```
TCR_NEW = (total_steps - collision_steps) / total_steps
```

- Computed **only when SR_NEW = true**; otherwise TCR_NEW = 0.
- Measures the fraction of steps that were collision-free.
- `collision_steps` = number of steps where a collision was detected (i.e., `path_collision_count`).
- Values closer to 1.0 indicate fewer collision events.

### DCR_NEW (Distance Collision Ratio)

```
DCR_NEW = (total_distance - collision_path_length) / total_distance
```

- Computed **only when SR_NEW = true**; otherwise DCR_NEW = 0.
- Measures the fraction of traversed distance that was collision-free.
- `collision_path_length`: total length of path segments that overlap with obstacles (sampled at 0.05 m intervals).
- `total_distance`: cumulative distance of all path segments.
- Values closer to 1.0 indicate less distance spent inside obstacles.

### Difficulty Classification

**Outdoor** tasks are classified by `shortest_path_length` into three equal-width buckets over [5, 50] m:

| Difficulty | Range |
|:-----------|:------|
| short | [5, 20) m |
| medium | [20, 35) m |
| long | [35, 50] m (upper bound extended to 75 m to capture boundary tasks) |

Tasks with `shortest_path_length` outside [5, 75] m are classified as `out_of_range`.

**Indoor** tasks are classified by a fixed scene list:

| Difficulty | Scenes |
|:-----------|:-------|
| easy | 0802_840243, 0803_840265, 0821_841631, 0822_841630, 0827_841619, 0841_841759, 0845_841765, 0854_841775 |
| hard | 0813_841249, 0814_841252, 0832_840249, 0833_840508, 0837_841153, 0843_841761, 0847_841768, 0861_841783 |

Scenes not in either list are classified as `unknown`.

### Excluded Scenes

By default, the outdoor evaluation excludes scene `park3` from metrics computation. This can be overridden with `--exclude-scenes`:

```bash
# Include all scenes (no exclusion)
python -m abotn_evaluator.point_goal.metrics --result-dir ./results --mode outdoor --exclude-scenes

# Exclude specific scenes
python -m abotn_evaluator.point_goal.metrics --result-dir ./results --mode outdoor --exclude-scenes park3 park7
```

### Aggregation

Aggregated metrics (in `eval_summary.json`) are computed as follows:

- **success_rate, sr_new, spl_new, tcr_new, dcr_new**: arithmetic mean across all tasks in the group.
- **global_collision_path_ratio**: total collision path length divided by total distance across all tasks (a global ratio, not a mean of per-task ratios).
- **Groups**: metrics are computed independently per difficulty group (short/medium/long or easy/hard) and also for the overall set.

## Output Directory Structure

```
results/outdoor/
  20260713_143022/
    scene_001/
      traj_1/
        result.json              # per-task evaluation result
        render_images/           # when save_render_images=True
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
    eval_summary.json            # aggregated metrics (generated by analyze_and_report)
```

### result.json Fields

Each `result.json` contains:

| Field | Type | Description |
|:------|:-----|:------------|
| `episode_id` | str | Scene identifier |
| `task_id` | str | Task identifier within the scene |
| `status` | str | Termination reason: `"stop"` (agent declared arrival), `"max_steps"`, `"error"`, or `"render_error"` |
| `steps` | int | Number of steps taken |
| `travel_length` | float | Actual path length traversed (metres) |
| `shortest_path_length` | float | Reference path length (metres) |
| `distance_to_goal` | float | Final distance to goal (metres) |
| `success` | bool | Whether `distance_to_goal <= arrive_threshold` |
| `oracle_success` | bool | Same as `success` in the current implementation |
| `metrics.initial_distance_to_goal` | float | Distance from start pose to goal |
| `metrics.min_distance_to_goal` | float | Minimum distance achieved at any step |
| `metrics.final_distance_to_goal` | float | Distance at the last step |
| `metrics.success_new` | bool | SR_NEW for this task |
| `metrics.spl_new` | float | SPL_NEW for this task |
| `metrics.tcr_new` | float | TCR_NEW for this task |
| `metrics.dcr_new` | float | DCR_NEW for this task |
| `metrics.path_collision_count` | int | Total collision events along the path |
| `metrics.collision_path_length` | float | Total length of path segments inside obstacles (metres) |
| `metrics.total_distance` | float | Cumulative distance of all path segments (metres) |
| `metrics.is_path_collided` | bool | Whether any collision occurred |
| `metrics.collision_path_ratio` | float | `collision_path_length / total_distance` |
| `metrics.collision_steps` | list[int] | Step indices where collisions occurred |

