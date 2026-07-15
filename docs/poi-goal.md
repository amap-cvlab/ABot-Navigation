# POI-Goal Evaluation

[English](poi-goal.md) | [中文](zh-CN/poi-goal.md)

In POI-Goal navigation, the agent receives a POI name (e.g., "Starbucks") and must navigate to within 2.0 m of its entrance using visual recognition.

## Differences from Point-Goal

| | Point-Goal | POI-Goal |
|:-|:-----------|:---------|
| Goal specification | (x, y) coordinates | POI name (`poi_name` field) |
| Arrival threshold | 0.5 m | 2.0 m |
| Collision handling | Count threshold (3 or 1) | Hard mode (first collision terminates) |
| Protocol modes | outdoor / indoor | Single protocol |
| Render scale | 1.0 | 1.5 (supersampling for signage) |
| Metrics grouping | By distance | Global + per-POI |

## Protocol

| Parameter | Value |
|:----------|:------|
| `arrive_threshold` | 2.0 m |
| `collision_mode` | `"hard"` (terminate on first non-exempt collision) |
| `occ_dilation_meters` | 0.5 |
| `max_steps` | 100 |

Collisions inside the arrival circle (within 2.0 m of the goal) are exempt, since POI entrances are on building facades.

These protocol parameters are **fixed benchmark standards**. Changing them produces non-comparable results.

## Prerequisites

- Evaluator installed (`pip install abotn-bench`)
- Render server running with `scripts/start_POIGoal_render_server.sh` (`RENDER_SCALE=1.5`)
- ABotN-POIBench data downloaded

## Evaluation via Python API

For complete interface field definitions and coordinate system details, see [API Reference](api-reference.md). For adapting models with different I/O conventions, see [Custom Agents](custom-agents.md).

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

`PoiGoalEvalConfig` defaults match the standard protocol (`arrive_threshold=2.0`, `collision_mode="hard"`, `occ_dilation_meters=0.5`). No manual overrides needed.

You can also customise the collision mode or robot radius via the config object:

```python
config = PoiGoalEvalConfig(
    render_url=RENDER_URL,
    max_steps=100,
    collision_mode="soft",       # count collisions without terminating
    robot_radius=0.15,           # 15 cm circular footprint
    occ_obstacle_polarity="dark",
    occ_dark_threshold=64,
)
```

## Evaluation via CLI

```bash
python -m abotn_evaluator.poi_goal.runner \
    --agent-module your_agent_module:YourPoiAgent \
    --data-dir /path/to/ABotN-POIBench/annotations \
    --map-dir /path/to/ABotN-POIBench/occmaps \
    --render-url http://localhost:7036/render_gs \
    --output-dir ./results/poi \
    --max-steps 100
```

POI-Goal does not require `--mode` for protocol selection -- protocol parameters are built-in defaults. Resume and multi-GPU work the same as Point-Goal (see [Point-Goal Evaluation](point-goal.md#resume-and-multi-gpu)).

### Full Parameter Reference

Parameters are grouped by category. All parameters listed below are optional unless marked **(required)**.

#### Data / Output

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `--data-dir` | str | **(required)** | Root directory containing per-scene subdirectories |
| `--map-dir` | str | `None` | Separate directory for map data (occ_map, height_map, etc.). When omitted, maps are loaded from `--data-dir` |
| `--output-dir` | str | `./eval_output` | Directory for evaluation outputs. A timestamped subdirectory is created for each run |
| `--resume-dir` | str | `None` | Directory of a previous run to resume. Scans for completed `result.json` files and skips them |

#### Agent

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `--agent-module` | str | **(required)** | Agent class path as `package.module:ClassName` |
| `--agent-config` | str | `None` | Optional YAML config file for the agent. YAML keys become `**kwargs` to `__init__` |
| `--evaluator-module` | str | `None` | Custom evaluator class path. Must accept `(scene, renderer, config, output_dir)` in its constructor |

#### Renderer

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `--render-url` | str | `http://127.0.0.1:7001/render_gs` | URL of the Gaussian splatting render service |

#### POI-Specific Parameters

These parameters are unique to POI-Goal evaluation and do not exist in the Point-Goal runner.

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `--arrive-threshold` | float | `2.0` | Goal arrival distance threshold in metres. Larger than Point-Goal (0.5 m) because POI targets are typically at building facades |
| `--collision-mode` | str | `hard` | Collision handling mode: `off`, `soft`, or `hard`. See [Collision Mode Details](#collision-mode-details) |
| `--robot-radius` | float | `0.0` | Robot radius in metres for circular footprint collision detection. `0.0` means only the centre pixel is checked |
| `--occ-obstacle-polarity` | str | `dark` | How to interpret occ_map pixel values: `dark` = dark pixels are obstacles; `light` = light pixels are obstacles |
| `--occ-dark-threshold` | int | `64` | Greyscale threshold for obstacle detection. Pixels at or below this value (in `dark` polarity) are treated as obstacles |
| `--per-poi` / `--no-per-poi` | flag | `True` | Print per-POI metrics breakdown in the analysis output |

#### Evaluation Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `--max-steps` | int | `100` | Maximum steps per task |
| `--collision-threshold` | int | `3` | Collision count threshold for legacy `SR_NEW` metrics (kept for backward compatibility) |
| `--occ-dilation-meters` | float | `0.5` | Dilate free space in the occupancy map by this many metres |

#### Camera

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `--camera-width` | int | `720` | Camera image width in pixels |
| `--camera-height` | int | `640` | Camera image height in pixels |
| `--camera-fx` | float | `252.075` | Camera focal length x |
| `--camera-fy` | float | `252.075` | Camera focal length y |
| `--extrinsic-height` | float | `0.65` | Camera height above ground in metres |

#### Observation Options

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `--provide-history` | flag | `False` | Include history images and poses in observations |
| `--provide-occ-map` | flag | `False` | Include occupancy map in observations |
| `--provide-height-map` | flag | `False` | Include height map in observations |
| `--max-history-frames` | int | `20` | Maximum history frames kept in ShortMemory |
| `--history-resize-ratio` | float | `0.25` | Resize ratio for history frame images |

#### Output Options

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `--save-render-images` / `--no-save-render-images` | flag | `True` | Save rendered images to disk |
| `--enable-visualization` | flag | `False` | Enable visualization of agent's extra field (e.g., affordance pixel overlay) |

#### Metrics Analysis

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `--mode` | str | `indoor` | Difficulty classification mode for metrics grouping: `outdoor` or `indoor` |
| `--min-distance` | float | `5.0` | Outdoor: minimum path length for bucketing |
| `--max-distance` | float | `50.0` | Outdoor: maximum path length for bucketing |
| `--exclude-scenes` | str list | `None` | Scene IDs to exclude from metrics |
| `--per-scene` | flag | `False` | Print per-scene metrics breakdown |
| `--skip-metrics` | flag | `False` | Skip post-evaluation metrics analysis (only per-task `result.json` files are produced) |

## Collision Mode Details

POI-Goal evaluation supports three collision detection modes, controlled by `--collision-mode`:

### `off` -- No Collision Detection

No collision checking is performed at all. The occupancy map is not queried, and no collision statistics are recorded. Use this mode when you want to measure pure navigation accuracy without collision considerations.

### `soft` -- Count Only

Collisions are detected and recorded in `result.json` metrics, but they **do not** affect success determination or cause early termination. The agent continues navigating even after hitting obstacles. This mode is useful for gathering collision statistics during development while preserving the ability to complete tasks.

Recorded statistics include `collision_count`, `collision_rate`, `max_consecutive_collision`, and `collision_step_indices`.

### `hard` -- Terminate on First Collision (Default)

The episode terminates immediately upon the first non-exempt collision. The task is marked as failed with `status="collision"` and `success=false`, regardless of how close the agent was to the goal. This is the **standard benchmark mode**.

### Terminal-Zone Exemption

In all modes (`soft` and `hard`), collisions that occur inside the **arrive_threshold circle** (within 2.0 m of the goal position) are **exempt** -- they are not counted and do not trigger termination. This exemption exists because POI entrances are on building facades: the agent must approach the building wall to reach the goal, and the occupancy map marks the building as an obstacle.

These exempt collisions are tracked separately in the `collision_skipped_near_goal` field of the result metrics for diagnostic purposes.

### The `collision_terminated` Flag

Each task result contains a `collision_terminated` boolean in its metrics:

- `true`: the task was stopped early by a hard-mode collision. The `status` field will be `"collision"`.
- `false`: the task ended normally -- either the agent declared arrival (`status="stop"`), reached `max_steps` (`status="max_steps"`), or collision mode was not `hard`.

## Check Metrics During Evaluation

You can compute aggregate metrics at any time -- even while evaluation is still in progress -- by running:

```bash
python -m abotn_evaluator.poi_goal.metrics --result-dir <dir>
```

This scans all completed `result.json` files under the given directory and prints the same summary tables that are produced at the end of a full evaluation run. Optional flags:

```bash
python -m abotn_evaluator.poi_goal.metrics \
    --result-dir ./results/poi/20260713_143022 \
    --arrive-threshold 2.0 \
    --per-poi                   # per-POI breakdown (default: on)
    --per-scene                 # per-scene breakdown
    --mode indoor               # difficulty grouping
    --exclude-scenes scene_042  # skip specific scenes
    --output-path ./custom_analysis.json
```

The command writes `poi_goal_analysis.json` to the result directory (or to `--output-path` if specified) and prints a formatted summary to stdout.

## Agent Interface

The only difference from Point-Goal is that `predict()` receives a `PoiGoalObservation` with an additional `poi_name: str` field:

```python
from abotn_evaluator.interface.poi_goal import BasePoiGoalAgent, PoiGoalObservation
from abotn_evaluator.interface.point_goal import WaypointPrediction

class YourPoiAgent(BasePoiGoalAgent):
    def reset(self): ...
    def predict(self, obs: PoiGoalObservation) -> WaypointPrediction:
        name = obs.poi_name  # e.g., "Starbucks"
        return WaypointPrediction(waypoint=..., arrive=obs.distance_to_goal < 2.0)
```

All other fields (`images`, `target_position`, `waypoint`, etc.) are identical to Point-Goal. See [API Reference](api-reference.md).

## Metrics

`analyze_and_report` produces `poi_goal_analysis.json` with global, per-group, and per-POI statistics.

### Global Metrics

| Field | Description |
|:------|:------------|
| `success_rate` | Fraction of tasks where the agent finished within `arrive_threshold` of the goal (and was not collision-terminated) |
| `spl` | Average Success weighted by Path Length (see formula below) |
| `avg_steps` | Average number of steps taken per task |
| `avg_travel_length` | Average actual travel distance |
| `avg_shortest_path_length` | Average ground-truth shortest path length |
| `avg_initial_distance` | Average initial distance to goal |
| `avg_min_distance` | Average minimum distance to goal achieved during the episode |
| `avg_final_distance` | Average final distance to goal |

### POI-SPL Formula

Standard SPL is `success * shortest / max(travel, shortest)`. However, POI-Goal uses a 2.0 m arrival threshold, which is large relative to many path lengths. If the shortest path is, say, 5.0 m, a perfectly efficient agent only needs to travel 3.0 m. Using the raw shortest path in the denominator would cap SPL well below 1.0 for efficient agents.

POI-Goal therefore subtracts the arrival threshold from the shortest path:

```
effective_shortest = max(shortest_path_length - arrive_threshold, 1e-3)
SPL = success * effective_shortest / max(travel_length, effective_shortest)
```

The `1e-3` floor prevents division by zero when the shortest path is shorter than the threshold.

**Why this adjustment matters.** Without it, an agent that walks a near-optimal path to a nearby POI (e.g., shortest = 3.0 m, travel = 1.5 m) would get an artificially low SPL because `shortest / max(travel, shortest)` = `3.0 / 3.0` = 1.0, which hides the fact that the agent was efficient. The adjusted formula gives `max(3.0 - 2.0, 0.001) / max(1.5, 1.0)` = `1.0 / 1.5` = 0.667, properly reflecting the travel efficiency relative to the minimum required distance.

### Per-Task Result Fields

Each `result.json` contains:

| Field | Description |
|:------|:------------|
| `success` | Whether `distance_to_goal <= arrive_threshold` at termination (forced `false` if collision-terminated) |
| `spl` | Task-level SPL using the adjusted formula |
| `status` | Termination reason: `"stop"` (agent declared arrival), `"max_steps"`, `"collision"`, or `"render_error"` |
| `steps` | Number of steps taken |
| `travel_length` | Actual distance travelled |
| `shortest_path_length` | Ground-truth shortest path length |
| `distance_to_goal` | Final distance to goal |
| `metrics.poi_name` | The POI name for this task |
| `metrics.collision_mode` | Active collision mode (`off`/`soft`/`hard`) |
| `metrics.collision_count` | Total non-exempt collisions during the episode |
| `metrics.collision_rate` | `collision_count / steps` |
| `metrics.max_consecutive_collision` | Longest consecutive collision streak |
| `metrics.collision_step_indices` | List of step indices where collisions occurred |
| `metrics.collision_skipped_near_goal` | Collisions inside the terminal zone (not counted) |
| `metrics.has_collision` | Boolean: any non-exempt collision occurred |
| `metrics.collision_terminated` | Boolean: task was ended by a hard-mode collision |
| `metrics.start_in_obstacle` | Sanity flag: start pose is inside an obstacle |
| `metrics.goal_in_obstacle` | Sanity flag: goal pose is inside an obstacle |

### Per-POI Breakdown

When `--per-poi` is enabled (default), the analysis output includes a `poi_stats` dictionary keyed by POI name. Each entry contains:

| Field | Description |
|:------|:------------|
| `total` | Number of tasks for this POI |
| `success` | Number of successful tasks |
| `success_rate` | `success / total` |
| `avg_spl` | Average SPL across tasks for this POI |
| `collision_count` | Number of tasks with at least one collision |
| `collision_terminated_count` | Number of tasks terminated by collision |

This breakdown helps identify which POIs are particularly difficult for the agent.

### Collision Statistics (Aggregate)

The aggregate `collision` block in the analysis output contains:

| Field | Description |
|:------|:------------|
| `tasks_evaluated` | Number of tasks with collision detection enabled |
| `tasks_with_collision` | Number of tasks where at least one collision occurred |
| `collision_task_rate` | `tasks_with_collision / tasks_evaluated` |
| `total_collision_steps` | Total collision steps across all tasks |
| `avg_collision_per_task` | `total_collision_steps / tasks_evaluated` |
| `avg_collision_rate` | Average per-step collision rate across tasks |
| `collision_terminated_count` | Number of tasks terminated by hard-mode collision |

## Output Directory Structure

```
results/poi/
└── 20260713_143022/
    ├── region_001/traj_1/
    │   ├── result.json
    │   └── render_images/
    ├── region_001/traj_2/
    │   ├── result.json
    │   └── render_images/
    ├── eval_summary.json          # written by the evaluator at the end of a run
    └── poi_goal_analysis.json     # written by analyze_and_report
```

- `result.json`: Per-task result with all metrics fields described above.
- `render_images/`: Rendered frames saved when `--save-render-images` is enabled (default).
- `eval_summary.json`: Summary produced by the evaluator, including per-POI stats and collision summary.
- `poi_goal_analysis.json`: Full metrics analysis produced by `analyze_and_report`, including per-group and per-POI breakdowns.

