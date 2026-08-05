# API Reference

[English](api-reference.md) | [中文](zh-CN/api-reference.md)

Complete interface specification for the `abotn_evaluator` package.

---

## Evaluation Loop

The evaluator drives each episode as a black-box loop:

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

- The evaluator only calls `reset()` and `predict()`
- Internal agent architecture (memory, planning, etc.) is invisible to the evaluator
- When `arrive=True`, the evaluator checks whether the actual distance ≤ `arrive_threshold`

---

## Agent Interface

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

`PoiGoalObservation` inherits all `Observation` fields and adds `poi_name: str`.

---

## Observation Fields

| Field | Type | Required | Format |
|:------|:-----|:--------:|:-------|
| `images` | `Dict[str, ndarray]` | Yes | Three-view RGB. Keys: `"left"`, `"front"`, `"right"`. Shape: `(640, 720, 3)`, dtype `uint8`, RGB. Views at -90°, 0°, +90° from heading. |
| `target_position` | `ndarray` | Yes | Target in agent-local coordinates `[front, left]` in metres |
| `distance_to_goal` | `float` | Yes | Euclidean XY distance to target (metres) |
| `position` | `ndarray` | Yes | Agent world position `[x, y, z]` (metres) |
| `rotation` | `ndarray` | Yes | Camera-to-world 4x4 homogeneous transform |
| `heading` | `float` | Yes | Agent yaw angle (radians) |
| `step_count` | `int` | Yes | Steps taken in current episode (0-indexed) |
| `goal_world` | `ndarray` | No | Goal position in world coordinates `[x, y]` (metres). Provided in Point-Goal only; `None` in POI-Goal. |
| `history_images` | `List[Dict[str, ndarray]]` | No | Past frames (requires `provide_history=True`) |
| `history_poses` | `List[ndarray]` | No | Past 4x4 poses (requires `provide_history=True`) |
| `occ_map` | `ndarray` | No | Occupancy grid (requires `provide_occ_map=True`) |
| `height_map` | `ndarray` | No | Height map (requires `provide_height_map=True`) |
| `meta_data` | `dict` | No | Scene metadata (coordinate conversion parameters, etc.) |
| `extra` | `dict` | Yes | Extension field, empty by default |

**POI-Goal additional field:**

| Field | Type | Description |
|:------|:-----|:------------|
| `poi_name` | `str` | Target POI name (e.g., `"Starbucks"`) |

## WaypointPrediction Fields

| Field | Type | Required | Format |
|:------|:-----|:--------:|:-------|
| `waypoint` | `ndarray` | Yes | Next waypoint in agent-local `[front, left]` (metres). Accepts shape `(2,)`, `(1,2)`, or `(N,2)`. |
| `arrive` | `bool` | Yes | Whether the agent declares arrival. Set `True` when `distance_to_goal < arrive_threshold`. |
| `directions` | `ndarray` | No | Direction vectors, shape `(N,2)`. Auto-computed from waypoint if omitted. |
| `confidence` | `float` | No | Confidence score `[0,1]`. Logged only, does not affect evaluation. |
| `extra` | `dict` | No | Debug/visualization data (see [Visualization](#visualization-extension)). |

---

## Coordinate System

### Agent-Local (used by `target_position` and `waypoint`)

- Origin: agent's current position
- Axis 0 (`right`): positive right
- Axis 1 (`front`): positive front

```
     front (+)
        |
        |
        o----->
            right (+) agent
```

Right-handed, units in metres.

### World (used by `position` and `rotation`)

Fixed scene-global frame. The `rotation` matrix transforms local vectors to world:

```
world_point = rotation[:3, :3] @ local_point + rotation[:3, 3]
```

You only need to ensure `waypoint` is in the local frame in metres -- the evaluator handles the conversion internally.

> If your model outputs a different coordinate system (e.g., `[forward, left]` or polar `(r, theta)`), convert in `predict()`. See [Custom Agent Integration](custom-agents.md).

---

## Configuration

### Point-Goal: `make_eval_config`

```python
from abotn_evaluator.point_goal.evaluator import make_eval_config

config = make_eval_config(
    mode="outdoor",  # or "indoor"
    render_url="http://localhost:7036/render_gs",
    max_steps=100,
)
```

Protocol parameters are set automatically by `mode`:

| Mode | `arrive_threshold` | `collision_threshold` | `occ_dilation_meters` |
|:-----|:-------------------|:----------------------|:----------------------|
| `outdoor` | 0.5 | 3 | 0.5 |
| `indoor` | 0.5 | 1 | 0.2 |

Runtime parameters (passed as `**kwargs`):

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `max_steps` | 100 | Maximum steps per episode |
| `save_render_images` | `True` | Save rendered images to disk |
| `provide_history` | `False` | Include `history_images` / `history_poses` in observations |
| `provide_occ_map` | `False` | Include `occ_map` in observations |
| `provide_height_map` | `False` | Include `height_map` in observations |
| `enable_visualization` | `False` | Process `WaypointPrediction.extra` for visualization |

Protocol parameters can be explicitly overridden for ablation studies: `make_eval_config(mode="outdoor", collision_threshold=5)`.

### POI-Goal: `PoiGoalEvalConfig`

```python
from abotn_evaluator.poi_goal.evaluator import PoiGoalEvalConfig

config = PoiGoalEvalConfig(
    render_url="http://localhost:7036/render_gs",
    max_steps=100,
)
```

Single protocol -- defaults match the standard:

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `arrive_threshold` | 2.0 | Arrival distance (metres) |
| `collision_mode` | `"hard"` | `"off"` / `"soft"` / `"hard"` |
| `occ_dilation_meters` | 0.5 | Free-space dilation (metres) |
| `robot_radius` | 0.0 | Robot footprint radius (metres) |
| `occ_obstacle_polarity` | `"dark"` | `"dark"` = dark pixels are obstacles |
| `occ_dark_threshold` | 64 | Greyscale obstacle threshold |

`collision_mode` values:
- `"off"`: no collision detection
- `"soft"`: count collisions, no effect on success or termination
- `"hard"`: terminate on first non-exempt collision (collisions inside the arrival circle are exempt)

---

## CLI Parameters

| Parameter | Point-Goal | POI-Goal | Description |
|:----------|:-----------|:---------|:------------|
| `--agent-module` | Yes | Yes | Agent class as `package.module:ClassName` |
| `--agent-config` | Yes | Yes | YAML config (keys become `**kwargs` to `__init__`) |
| `--data-dir` | Yes | Yes | Trajectory data directory |
| `--map-dir` | Yes | Yes | Occupancy map directory |
| `--render-url` | Yes | Yes | Render server URL |
| `--output-dir` | Yes | Yes | Output directory |
| `--max-steps` | 100 | 100 | Maximum steps per episode |
| `--mode` | `outdoor` / `indoor` | `indoor` | Selects protocol parameters (Point-Goal) or metrics grouping (POI-Goal) |
| `--arrive-threshold` | Auto | 2.0 | Override arrival distance |
| `--collision-threshold` | Auto | 3 | Collision count threshold |
| `--collision-mode` | -- | `hard` | POI-Goal collision mode |
| `--occ-dilation-meters` | Auto | 0.5 | Override free-space dilation |
| `--provide-history` | Flag | Flag | Enable history observations |
| `--save-render-images` | Flag | Flag | Save images (default: on) |
| `--enable-visualization` | Flag | Flag | Enable extra-field visualization |
| `--resume-dir` | Yes | Yes | Resume from a previous run directory |
| `--skip-metrics` | Flag | Flag | Skip post-evaluation metrics analysis |

Point-Goal protocol parameters are auto-selected by `--mode` when not explicitly provided.

---

## GaussianScene

Scene manager that loads episodes from a local directory tree and provides coordinate conversion, collision detection, and iteration utilities.

### Import and Construction

```python
from abotn_evaluator.scene import GaussianScene

scene = GaussianScene(
    local_data_path="/path/to/ABotN-PointBench/Outdoor/annotations",
    local_map_path="/path/to/ABotN-PointBench/Outdoor/occmaps",   # optional
)
```

### Constructor Parameters

| Parameter | Type | Required | Description |
|:----------|:-----|:--------:|:------------|
| `local_data_path` | `str` | Yes | Root directory containing per-scene subdirectories. Each subdirectory holds `traj_*.json` files and optionally a `map/` folder. |
| `local_map_path` | `str` | No | Separate directory for map data. When provided, occupancy maps are loaded from `{local_map_path}/{episode_id}/` instead of the episode's own `scene_path`. Useful when map data is stored separately from trajectory data. |

On construction, `GaussianScene` automatically scans `local_data_path` for subdirectories, creates an `Episode` for each one, and loads all trajectory tasks and map data.

### Episode Dataclass

Each scene directory becomes an `Episode` instance.

```python
from abotn_evaluator.scene import Episode

episode = Episode.from_scene_dir("/path/to/scene_001")
```

| Field | Type | Description |
|:------|:-----|:------------|
| `episode_id` | `str` | Unique scene identifier (the directory name). |
| `scene_path` | `str` | Absolute path to the local scene directory. |
| `tasks` | `List[Task]` | Navigation tasks loaded from `traj_*.json` files. |
| `occ_map` | `ndarray` or `None` | Occupancy grid -- grayscale `uint8`. Pixel values `>=128` are **free space**; values `<128` are **obstacles**. Loaded from `map/occ_map.png`. |
| `height_map` | `ndarray` or `None` | Per-pixel ground height in metres. Loaded from `map/occ_map_height.tiff`. |
| `meta_data` | `dict` or `None` | Coordinate-system metadata parsed from `map/occ_map_meta.txt`. Keys include `TOP_LEFT_X`, `TOP_LEFT_Y`, `IMAGE_WIDTH`, `IMAGE_HEIGHT`, `COORDINATE_RANGE_X`, `COORDINATE_RANGE_Y`, etc. |

**Expected directory layout:**

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

**Convenience methods:**

| Method / Property | Return Type | Description |
|:------------------|:------------|:------------|
| `has_map_data()` | `bool` | Returns `True` if `occ_map`, `meta_data`, and `height_map` are all loaded. |
| `num_tasks` | `int` | Number of tasks in this episode. |
| `get_meter_per_pixel()` | `float` or `None` | Average map resolution (metres per pixel), computed from metadata. |

### Task Dataclass

Each `traj_*.json` file becomes a `Task` instance.

```python
from abotn_evaluator.scene import Task

task = Task.from_json("/path/to/scene_001/traj_0.json")
```

| Field / Property | Type | Description |
|:-----------------|:-----|:------------|
| `task_id` | `str` | Unique identifier -- the JSON filename stem (e.g. `"traj_0"`). |
| `trajectory` | `List[Dict[str, float]]` | Sequence of waypoints. Each dict contains at least `x`, `y`, `z`, `roll`, `pitch`, `yaw` (Euler angles in radians). |
| `label` | `dict` | Metadata label from the original trajectory JSON. |
| `start_pose` | `dict` or `None` | Property. First element of `trajectory`, or `None` if empty. |
| `end_pose` | `dict` or `None` | Property. Last element of `trajectory`, or `None` if empty. |
| `goal_label` | `str` | Property. Semantic label of the navigation target, extracted from `label["extend"]["goal_label"]`. Returns `""` if not present. |

### Collision Detection Methods

All collision methods are classmethods on `GaussianScene`.

#### `check_point_collision_status`

```python
is_colliding = GaussianScene.check_point_collision_status(
    pose,                   # ndarray (4x4) or dict with "x", "y"
    episode,                # Episode with occ_map and meta_data
    robot_radius_pixel=2,   # half-size of square footprint (pixels)
)
```

Checks whether a single pose falls on an obstacle in the occupancy map. When `robot_radius_pixel > 0`, a square patch around the point is checked; any obstacle pixel within the patch triggers a collision. Points outside the map boundary are treated as collisions.

**Returns:** `bool` -- `True` if the point is in collision.

#### `check_line_collision_fast`

```python
has_collision = GaussianScene.check_line_collision_fast(
    pose_start,             # ndarray (4x4) or dict with "x", "y"
    pose_end,               # ndarray (4x4) or dict with "x", "y"
    episode,                # Episode with occ_map and meta_data
    step_size_pixel=2.0,    # sampling interval in pixels
)
```

Checks whether the straight-line path between two poses crosses an obstacle. Uses uniform sampling along the line in pixel space. Points outside the map boundary are treated as collisions.

**Returns:** `bool` -- `True` if any sample point along the line is in collision.

#### `compute_line_collision_length`

```python
collision_meters = GaussianScene.compute_line_collision_length(
    pose_start,               # ndarray (4x4) or dict with "x", "y"
    pose_end,                 # ndarray (4x4) or dict with "x", "y"
    episode,                  # Episode with occ_map and meta_data
    sample_step_meter=0.05,   # sampling interval in metres
)
```

Computes the physical length (in metres) of the path segment that lies inside obstacles. Uses dense sampling in world coordinates, then maps each sample to the occupancy grid. Out-of-bounds points are treated as obstacles. Contiguous obstacle segments are accumulated into the total collision length.

**Returns:** `float` -- total collision length in metres.

### Coordinate Conversion

Both methods are static methods on `GaussianScene`.

#### `convert_actual_to_pixel`

```python
x_pixel, y_pixel = GaussianScene.convert_actual_to_pixel(
    x_actual,   # float, world X coordinate
    y_actual,   # float, world Y coordinate
    episode,    # Episode with meta_data
)
```

Converts world coordinates to occupancy-map pixel coordinates. Uses `meta_data` fields (`TOP_LEFT_X`, `TOP_LEFT_Y`, `IMAGE_WIDTH`, `IMAGE_HEIGHT`, `COORDINATE_RANGE_X`, `COORDINATE_RANGE_Y`) and automatically handles axis direction based on corner coordinates.

**Returns:** `(int, int)` -- pixel coordinates `(x_pixel, y_pixel)`.

#### `convert_pixel_to_actual`

```python
x_actual, y_actual = GaussianScene.convert_pixel_to_actual(
    x_pixel,    # int, pixel X coordinate
    y_pixel,    # int, pixel Y coordinate
    episode,    # Episode with meta_data
)
```

Inverse of `convert_actual_to_pixel`. Converts occupancy-map pixel coordinates back to world coordinates.

**Returns:** `(float, float)` -- world coordinates `(x_actual, y_actual)`.

### Iteration

```python
# Iterate over episodes
for episode in scene:
    print(episode.episode_id, episode.num_tasks)

# Iterate over all (episode, task) pairs
for episode, task in scene.iter_tasks():
    print(episode.episode_id, task.task_id)

# Total task count
print(scene.total_tasks)
```

| Method / Property | Return Type | Description |
|:------------------|:------------|:------------|
| `iter_tasks()` | `Generator[(Episode, Task)]` | Yields all `(episode, task)` pairs across all episodes. |
| `total_tasks` | `int` | Total number of tasks across all episodes. |
| `__len__()` | `int` | Number of episodes. |
| `__iter__()` | `Iterator[Episode]` | Iterate over episodes. |
| `get_episode(idx)` | `Episode` | Get episode by integer index. |
| `get_episode_by_id(episode_id)` | `Episode` or `None` | Look up episode by ID string. |
| `summary()` | `dict` | Returns `{"num_episodes": ..., "total_tasks": ..., "episode_ids": [...]}`. |

### Map Manipulation

#### `dilate_free_space_by_meter`

```python
episode.dilate_free_space_by_meter(dilation_meters=0.5)
```

Dilates the free-space region of the occupancy map by a physical distance. This effectively shrinks obstacles, providing tolerance for annotation inaccuracies. Internally converts metres to pixels using scene metadata (`get_meter_per_pixel()`) and applies morphological dilation with an elliptical kernel.

If metadata is unavailable or `dilation_meters <= 0`, the operation is skipped with a warning.

There is also a lower-level `dilate_free_space(dilation_pixels)` method that operates directly in pixel units.

### Pose Conversion

```python
pose_matrix = GaussianScene.get_gaussian_pose(cur_info)
```

Converts a pose dictionary (with keys `x`, `y`, `z`, `roll`, `pitch`, `yaw` in radians) to a 4x4 camera-to-world homogeneous transformation matrix using XYZ Euler angle convention.

---

## GaussianRenderer

Client for a remote Gaussian splatting render service. Sends render requests via HTTP and returns RGB images.

### Import and Construction

```python
from abotn_evaluator.render_client import GaussianRenderer, CameraConfig

renderer = GaussianRenderer(
    render_url="http://localhost:7036/render_gs",
    camera_config=CameraConfig(),   # optional, uses defaults
    num_views=3,                    # 3 = left/right/front, 1 = front only
    timeout=30,                     # HTTP timeout in seconds
    max_retries=3,                  # retry count on failure
    retry_backoff=1.0,              # base sleep time between retries
)
```

### CameraConfig Dataclass

Pinhole camera intrinsics and height offset.

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

| Field | Type | Default | Description |
|:------|:-----|:--------|:------------|
| `width` | `int` | `720` | Image width in pixels. |
| `height` | `int` | `640` | Image height in pixels. |
| `fx` | `float` | `252.075` | Focal length along the x-axis (pixels). |
| `fy` | `float` | `252.075` | Focal length along the y-axis (pixels). |
| `cx` | `float` | `360.0` | Principal point x coordinate (pixels). |
| `cy` | `float` | `320.0` | Principal point y coordinate (pixels). |
| `extrinsic_height` | `float` | `0.65` | Camera height offset relative to the ground plane (metres). Positive means above ground. |

**Property:** `intrinsics_colmap` -- returns the intrinsics formatted as a COLMAP camera line: `"1 PINHOLE <width> <height> <fx> <fy> <cx> <cy>"`.

### Constructor Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `render_url` | `str` | (required) | Full URL of the render endpoint (e.g. `"http://host:7036/render_gs"`). |
| `camera_config` | `CameraConfig` | `CameraConfig()` | Camera intrinsics/extrinsics configuration. |
| `num_views` | `int` | `3` | Number of views to render per pose. Use `3` for left/right/front, or `1` for front only. |
| `timeout` | `int` | `30` | HTTP request timeout in seconds. |
| `max_retries` | `int` | `3` | Maximum number of retries on render failure. |
| `retry_backoff` | `float` | `1.0` | Base sleep time (seconds) between retries. Doubled after each attempt (exponential backoff). |

### `render_at_pose`

```python
images = renderer.render_at_pose(
    pose,           # ndarray, 4x4 camera-to-world transform
    scene_id,       # str, identifier of the scene to render
    save_dir=None,  # optional str, directory to save images
    image_id=0,     # int, numeric prefix for saved filenames
)
```

Renders images at a given 4x4 pose matrix. For multi-view setups (`num_views=3`), the renderer produces three views by adding fixed yaw offsets to the base pose.

**View order and yaw offsets (for `num_views=3`):**

| Index | View Name | Yaw Offset |
|:------|:----------|:-----------|
| 0 | `left` | +90 degrees |
| 1 | `right` | -90 degrees |
| 2 | `front` | 0 degrees |

When `save_dir` is provided, images are saved as `{image_id}_{view_name}.jpg` (e.g. `0_left.jpg`, `0_right.jpg`, `0_front.jpg`).

**Parameters:**

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `pose` | `ndarray` | (required) | 4x4 camera-to-world transformation matrix. |
| `scene_id` | `str` | (required) | Scene identifier passed to the render server. |
| `save_dir` | `str` | `None` | If provided, rendered images are saved to this directory. |
| `image_id` | `int` | `0` | Numeric prefix for saved filenames. |
| `save_img_idx` | `List[int]` | `None` | If set, only images whose index appears in this list are saved to disk. |

**Returns:** `List[PIL.Image.Image]` -- list of rendered PIL images.

### Retry Mechanism

When a render request fails (HTTP error or exception), the renderer retries with exponential backoff:

1. First retry: sleeps `retry_backoff` seconds (default 1.0s)
2. Second retry: sleeps `retry_backoff * 2` seconds (default 2.0s)
3. Third retry: sleeps `retry_backoff * 4` seconds (default 4.0s)

After all retries are exhausted, `render_at_pose` raises `RenderFailureError`.

```python
from abotn_evaluator.render_client import RenderFailureError

try:
    images = renderer.render_at_pose(pose, scene_id)
except RenderFailureError as e:
    print(f"Render failed after all retries: {e}")
```

---

## ShortMemory

Sliding-window memory buffer for multi-view RGB observations and agent poses. Manages a fixed-capacity history of observation frames, automatically downscaling historical images to save memory.

### Import and Construction

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

### Constructor Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `max_history_frames` | `int` | `20` | Maximum number of **history** frames to retain (excluding the current frame). |
| `num_current_views` | `int` | `3` | Number of camera views per frame (e.g. 3 for left/front/right). |
| `input_img_size` | `tuple(int, int)` | `(476, 420)` | `(width, height)` of the input images. Used as the reference size when downscaling history frames. |
| `resize_ratio` | `float` | `0.25` | Scale factor applied to history frame images. `0.25` means images are shrunk to 25% of their original dimensions. |
| `reorder_views` | `bool` | `True` | If `True`, reorder incoming views from internal order `[left, right, front]` to external order `[left, front, right]`. |
| `resize_on_add` | `bool` | `True` | If `True`, historical frames are downscaled when a new frame is added. Set to `False` when external code handles resizing. |

### `add_frame`

```python
frame_index = memory.add_frame(
    rgbs,                    # List of ndarray or PIL.Image
    pose,                    # ndarray, camera-to-world transform
    have_memory=False,       # eviction strategy
    num_current_image=None,  # optional override for num_current_views
)
```

Appends a new multi-view observation frame to the buffer. On each call, the method:

1. Converts input images to PIL format and optionally reorders views.
2. **Collapses the previous current frame** to front-view only -- side views (left, right) are discarded so that history retains only front views.
3. Appends the new frame's images and pose.
4. **Downscales** the frame that just became history (if `resize_on_add=True`).
5. **Evicts** one frame if the buffer exceeds `max_history_frames`.

**Parameters:**

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `rgbs` | `List` | (required) | Camera-view images. Each element is a NumPy `uint8` HWC array or a PIL Image. Expected length equals `num_current_views`. |
| `pose` | `ndarray` | (required) | Camera-to-world transform associated with this frame. |
| `have_memory` | `bool` | `False` | Eviction strategy when the window is full (see below). |
| `num_current_image` | `int` | `None` | Override `num_current_views` for this call. |

**Returns:** `int` -- the zero-based frame index assigned to this observation.

### Eviction Strategies

When the buffer exceeds capacity, one historical frame must be removed:

- **FIFO** (`have_memory=True`): Drops the oldest frame. Simple and predictable.
- **Adaptive** (`have_memory=False`): Drops the history frame whose temporal gap to its predecessor is smallest. This preserves temporal spread, keeping frames that are more evenly spaced in time.

### View Reorder

When `reorder_views=True` (default), the internal input order `[left, right, front]` (indices 0, 1, 2) is reordered to the external order `[left, front, right]` (indices 0, 2, 1). This ensures a consistent left-front-right ordering for downstream consumers.

### Accessors

| Method | Return Type | Description |
|:-------|:------------|:------------|
| `get_all_images()` | `List[PIL.Image]` | Full list of stored images (history front views + current multi-view). |
| `get_current_images()` | `List[PIL.Image]` | Only the current frame's multi-view images (typically 3: left, front, right). |
| `get_history_images()` | `List[PIL.Image]` | Only the historical (downscaled, front-only) images, excluding the current frame. |
| `get_all_poses()` | `List[ndarray]` | Full list of stored poses (one per stored image entry). |
| `get_history_poses()` | `List[ndarray]` | Poses corresponding to historical frames only. |
| `get_last_pose()` | `ndarray` or `None` | The most recently added pose, or `None` if the buffer is empty. |
| `get_image_indices()` | `List[int]` | Frame-index label for every stored image entry. Useful for tracking temporal spacing. |
| `reset()` | `None` | Clears all stored images, poses, and counters. |
| `reset_history()` | `None` | Clears history, keeping only the current frame's views. |

**Properties:**

| Property | Type | Description |
|:---------|:-----|:------------|
| `frame_count` | `int` | Total number of frames that have been added (including evicted ones). |
| `__len__()` | `int` | Number of individual image entries currently stored. |

---

## analyze_and_report

Full-pipeline metrics analysis function. **Can be called at any time during evaluation** -- it aggregates metrics from all completed `result.json` files found so far, making it useful for monitoring interim results.

### Point-Goal Usage

```python
from abotn_evaluator.point_goal.metrics import analyze_and_report

report = analyze_and_report(
    result_dir="./results/outdoor/run1",
    mode="outdoor",
)
```

### POI-Goal Usage

```python
from abotn_evaluator.poi_goal.metrics import analyze_and_report

report = analyze_and_report(
    result_dir="./results/poi/run1",
    mode="indoor",
    arrive_threshold=2.0,
)
```

### Point-Goal Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `result_dir` | `str` | (required) | Root evaluation output directory. All `result.json` files under this directory are loaded recursively. |
| `mode` | `str` | `"outdoor"` | `"outdoor"` or `"indoor"`. Controls difficulty classification (short/medium/long for outdoor; easy/hard for indoor). |
| `collision_threshold` | `int` | `3` | Collision count threshold for SR_NEW. A task's SR_NEW is `True` only if it succeeds AND has `path_collision_count < collision_threshold`. |
| `min_distance` | `float` | `5.0` | Minimum shortest-path distance for outdoor difficulty bucketing. |
| `max_distance` | `float` | `50.0` | Maximum shortest-path distance for outdoor difficulty bucketing. |
| `exclude_scenes` | `List[str]` | `None` | Scene IDs to exclude from analysis. Defaults to `["park3"]` for outdoor mode, empty for indoor. |
| `output_path` | `str` | `None` | Path for the output JSON file. Defaults to `{result_dir}/eval_summary.json`. |
| `per_scene` | `bool` | `False` | If `True`, prints a per-scene metrics breakdown table. |

### POI-Goal Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `result_dir` | `str` | (required) | Root evaluation output directory. |
| `mode` | `str` | `"indoor"` | `"outdoor"` or `"indoor"`. |
| `collision_threshold` | `int` | `3` | Collision count threshold (kept for backward compatibility). |
| `arrive_threshold` | `float` | `2.0` | Arrival distance threshold used during evaluation. Used to compute the adjusted SPL formula: `effective_shortest = max(shortest - arrive_threshold, 1e-3)`. |
| `min_distance` | `float` | `5.0` | Minimum distance for outdoor difficulty bucketing. |
| `max_distance` | `float` | `50.0` | Maximum distance for outdoor difficulty bucketing. |
| `exclude_scenes` | `List[str]` | `None` | Scene IDs to exclude. |
| `output_path` | `str` | `None` | Path for output JSON. Defaults to `{result_dir}/poi_goal_analysis.json`. |
| `per_scene` | `bool` | `False` | Print per-scene breakdown. |
| `per_poi` | `bool` | `True` | Print per-POI breakdown table. |

### CLI Usage

```bash
# Point-Goal metrics (can be run while evaluation is still in progress)
python -m abotn_evaluator.point_goal.metrics \
    --result-dir ./results/outdoor/run1 \
    --mode outdoor \
    --collision-threshold 3

# POI-Goal metrics
python -m abotn_evaluator.poi_goal.metrics \
    --result-dir ./results/poi/run1 \
    --mode indoor \
    --arrive-threshold 2.0 \
    --per-poi
```

### Return Value Structure

The function returns a dictionary (also saved as JSON) with the following structure:

**Point-Goal (`eval_summary.json`):**

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

**POI-Goal (`poi_goal_analysis.json`):**

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

### Difficulty Classification

**Outdoor mode** classifies tasks into `short`, `medium`, and `long` buckets based on `shortest_path_length`:

| Bucket | Range |
|:-------|:------|
| `short` | `[min_distance, min_distance + step)` |
| `medium` | `[min_distance + step, min_distance + 2*step)` |
| `long` | `[min_distance + 2*step, max_distance * 1.5]` |

where `step = (max_distance - min_distance) / 3`.

**Indoor mode** classifies tasks into `easy` and `hard` buckets based on predefined scene ID lists.

---

## Agent Loading

Dynamic class loading utility for importing agent classes at runtime.

### Import and Usage

```python
from abotn_evaluator.agent_loader import load_class

AgentClass = load_class("your_package.module:YourAgent")
agent = AgentClass(**config)
```

### Format

The `module_path` string must use the format `"package.module:ClassName"` with exactly one colon (`:`) separating the importable module path from the class name.

**Examples:**

```python
# Load from an installed package
Agent = load_class("agent_examples.point_goal_random:RandomPointGoalAgent")

# Load from a nested package
Agent = load_class("my_agents.vlm.qwen_agent:QwenPointGoalAgent")
```

### Error Handling

| Exception | Condition | Example |
|:----------|:----------|:--------|
| `ValueError` | Module path does not contain exactly one `:`, or either the module or class name is empty. | `load_class("no_colon_here")`, `load_class("a:b:c")`, `load_class(":MyClass")` |
| `ImportError` | The module cannot be imported. The error message includes the original import error for debugging. | `load_class("nonexistent_package.module:Agent")` |
| `AttributeError` | The class name is not found in the module. The error message lists all available classes in the module. | `load_class("agent_examples.point_goal_random:NonexistentClass")` |
| `TypeError` | The resolved attribute exists but is not a class (e.g. it is a function or variable). | `load_class("os.path:join")` |

---

## Configuration

YAML-based configuration loading with inheritance and mode resolution.

### Import and Usage

```python
from abotn_evaluator.config import load_agent_config

config = load_agent_config("agent_config.yaml")
```

### `load_agent_config`

Loads an agent configuration YAML file with two special features:

#### 1. `_base_` Inheritance

A child YAML can reference a base YAML file via the `_base_` key. The base file is loaded first, then the child's values are deep-merged on top (child values override base values).

**base_config.yaml:**
```yaml
model_name: "qwen-vl"
temperature: 0.7
max_tokens: 2048
```

**agent_config.yaml:**
```yaml
_base_: base_config.yaml
temperature: 0.5
custom_param: true
```

**Result:** `{"model_name": "qwen-vl", "temperature": 0.5, "max_tokens": 2048, "custom_param": true}`

The `_base_` path is resolved relative to the child YAML file's directory.

#### 2. Mode Resolution

If the config contains a `mode` key with a string value, and the config also contains sub-dictionaries, the matching mode's sub-dictionary is flattened into the top level.

**config_with_modes.yaml:**
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

**Result:** `{"mode": "outdoor", "model_name": "qwen-vl", "step_size": 2.0, "max_distance": 50.0}`

The non-matching mode sub-dicts (`indoor` in this case) are removed from the output.

### `load_yaml`

Lower-level helper that loads a YAML file and returns a dictionary without any inheritance or mode processing.

```python
from abotn_evaluator.config import load_yaml

raw_config = load_yaml("path/to/config.yaml")
```

Raises `FileNotFoundError` if the file does not exist, or `ValueError` if the content is not a top-level mapping.

### `merge_cli_args`

Merges command-line arguments into an existing config dictionary. CLI values override config values; only non-`None` arguments are merged.

```python
from abotn_evaluator.config import merge_cli_args

merged = merge_cli_args(
    config,                 # base config dict
    args,                   # argparse.Namespace or dict
    override_keys=None,     # optional: list of keys to consider
)
```

The original `config` dictionary is not modified; a new dictionary is returned.

---

## Visualization Extension

Agents can return debug data in `WaypointPrediction.extra`. When `enable_visualization=True`, the evaluator dispatches entries by key to registered handlers.

Built-in handler:

- **`affordance_pixel`**: Overlays pixel markers on rendered images

```python
return WaypointPrediction(
    waypoint=wp,
    arrive=False,
    extra={"affordance_pixel": {"front": [360, 320], "left": [200, 400]}},
)
```

Pixel coordinates accept three formats: normalized `[0, 1]`, Qwen-style `[0, 1000]`, or absolute pixels. If `extra` is empty, visualization is skipped even when enabled.

To add custom visualization types, register a handler in `abotn_evaluator/visualization.py`'s `_HANDLERS` dict.

---

## Output Format

### `result.json` (per task)

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

`status` values: `"stop"` (agent declared arrival), `"max_steps"` (step limit reached), `"collision"` (hard-mode termination), `"error"` (runtime error).

### Aggregated Summary

Generated by `analyze_and_report()`:

- **Point-Goal** -> `eval_summary.json`: overall + difficulty-group breakdowns of `sr_new` / `spl_new`
- **POI-Goal** -> `poi_goal_analysis.json`: overall + per-POI statistics

The CLI runner generates these automatically after evaluation. For API usage, call `analyze_and_report(result_dir=...)` explicitly.
