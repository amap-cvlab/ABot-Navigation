# Custom Agent Integration

[English](custom-agents.md) | [中文](zh-CN/custom-agents.md)

This guide explains how to integrate your own navigation model with ABotN-Bench, from the very first smoke-test to a full evaluation run.

## Integration Approaches

| Approach | When to Use |
|:---------|:------------|
| Direct | Your model already implements `reset()` / `predict()` with matching I/O |
| Wrapper | Your model uses a different coordinate system, image format, or output structure |
| CLI + Wrapper | Same as wrapper, but loaded via `--agent-module` for batch / multi-GPU runs |

All approaches require your agent to satisfy the interface defined in [API Reference](api-reference.md).

## Step-by-Step Workflow

This section walks you through the full path from "I have a trained model" to "I have evaluation metrics".

### Step 1: Verify the Pipeline with RandomAgent

Before touching your own model, confirm that the render server, data directories, and evaluation pipeline all work end-to-end by running the built-in `RandomPointGoalAgent`:

```bash
# Make sure the render server is running first
curl http://localhost:7036/ping   # should return a 200

# Run a quick smoke test with the random agent
python -m abotn_evaluator.point_goal.runner \
    --agent-module agent_examples.point_goal_random:RandomPointGoalAgent \
    --data-dir /path/to/annotations \
    --map-dir /path/to/occmaps \
    --render-url http://localhost:7036/render_gs \
    --mode outdoor \
    --max-steps 10
```

If this produces per-task `result.json` files and a final `eval_summary.json` under `./eval_output/`, the pipeline is healthy. If it fails, fix the infrastructure problems before proceeding.

### Step 2: Write Your Wrapper Class

Create a Python file that wraps your model behind the `BasePointGoalAgent` (or `BasePoiGoalAgent`) interface. See [Wrapper Pattern](#wrapper-pattern-interface-mismatch) below for detailed examples.

A minimal skeleton:

```python
from abotn_evaluator.interface.point_goal import BasePointGoalAgent, Observation, WaypointPrediction
import numpy as np

class MyAgent(BasePointGoalAgent):
    def __init__(self, model_path: str = "", device: str = "cuda:0"):
        # Load your model here
        pass

    def reset(self):
        pass

    def predict(self, obs: Observation) -> WaypointPrediction:
        # Your inference logic here
        waypoint = np.array([0.5, 0.0])
        return WaypointPrediction(waypoint=waypoint, arrive=False)
```

### Step 3: Fast Iteration with `--skip-metrics` and `--max-steps`

While debugging your wrapper, avoid waiting for full metrics computation. Use `--skip-metrics` and a small `--max-steps`:

```bash
python -m abotn_evaluator.point_goal.runner \
    --agent-module my_project.wrapper:MyAgent \
    --agent-config my_config.yaml \
    --data-dir /path/to/annotations \
    --map-dir /path/to/occmaps \
    --render-url http://localhost:7036/render_gs \
    --mode outdoor \
    --max-steps 5 \
    --skip-metrics
```

This runs only 5 steps per episode and skips aggregated analysis, so you get feedback in seconds. Check the per-task `result.json` files to verify your waypoints are reasonable.

### Step 4: Full Evaluation Run

Once your wrapper produces sensible results, run the full evaluation:

```bash
python -m abotn_evaluator.point_goal.runner \
    --agent-module my_project.wrapper:MyAgent \
    --agent-config my_config.yaml \
    --data-dir /path/to/annotations \
    --map-dir /path/to/occmaps \
    --render-url http://localhost:7036/render_gs \
    --mode outdoor \
    --max-steps 100 \
    --output-dir ./eval_output
```

### Step 5: Analyze Results

The runner produces `eval_summary.json` automatically. You can also run metrics analysis independently on a results directory -- useful for checking mid-run progress or re-analyzing with different parameters:

```bash
# Analyze results from a previous run (or a still-running one)
python -m abotn_evaluator.point_goal.metrics \
    --result-dir ./eval_output/20240101_120000 \
    --mode outdoor \
    --per-scene
```

## Direct Integration (Interface Already Aligned)

```python
from abotn_evaluator.point_goal.evaluator import PointGoalEvaluator, make_eval_config
from abotn_evaluator.scene import GaussianScene
from abotn_evaluator.render_client import GaussianRenderer
from your_model import YourAgent

scene = GaussianScene(local_data_path="/path/to/annotations", local_map_path="/path/to/occmaps")
renderer = GaussianRenderer(render_url="http://localhost:7036/render_gs")
config = make_eval_config(mode="outdoor", render_url=renderer.render_url, max_steps=100)
evaluator = PointGoalEvaluator(scene=scene, renderer=renderer, config=config)
results = evaluator.evaluate(YourAgent())
```

## Wrapper Pattern (Interface Mismatch)

When your model's inputs or outputs don't match the benchmark interface, write a wrapper class that converts between the two. The wrapper is a thin adapter -- all heavy lifting stays inside your model.

### Point-Goal Wrapper

A realistic wrapper that handles model loading on a specific GPU device and converts between coordinate systems:

```python
import numpy as np
import torch
from PIL import Image

from abotn_evaluator.interface.point_goal import BasePointGoalAgent, Observation, WaypointPrediction


class YourPointGoalWrapper(BasePointGoalAgent):
    """Wrap a PyTorch navigation model for ABotN-Bench evaluation.

    Constructor parameters come from the --agent-config YAML file.
    Every key in the YAML becomes a keyword argument here.
    """

    def __init__(
        self,
        model_path: str = "",
        device: str = "cuda:0",
        step_size: float = 0.5,
        arrive_distance: float = 0.5,
        image_size: tuple = (224, 224),
    ):
        self.device = torch.device(device)
        self.step_size = step_size
        self.arrive_distance = arrive_distance
        self.image_size = tuple(image_size)

        # Load your model checkpoint
        from your_model import NavigationNet
        self.model = NavigationNet.load_from_checkpoint(model_path)
        self.model = self.model.to(self.device).eval()

    def reset(self):
        """Reset episode-level state (hidden states, history buffers, etc.)."""
        self.model.reset_hidden_state()

    def predict(self, obs: Observation) -> WaypointPrediction:
        # ---- 1. Prepare images ----
        # obs.images values are (640, 720, 3) uint8 RGB numpy arrays.
        # Resize and normalize for your model.
        images = []
        for cam in ("front", "left", "right"):
            img = obs.images[cam]                          # (640, 720, 3) RGB
            pil = Image.fromarray(img)
            pil = pil.resize(self.image_size)              # resize to model input
            arr = np.array(pil, dtype=np.float32) / 255.0  # normalize to [0, 1]
            images.append(arr)

        # Stack to (B, 3, H, W) tensor
        batch = np.stack(images, axis=0)                   # (3, H, W, 3)
        batch = np.transpose(batch, (0, 3, 1, 2))         # (3, 3, H, W)
        batch_tensor = torch.from_numpy(batch).unsqueeze(0).to(self.device)

        # ---- 2. Prepare goal ----
        target = obs.target_position  # [front, left] in metres

        # Example: your model expects [x_right, y_forward]
        goal_for_model = np.array([-target[1], target[0]], dtype=np.float32)
        goal_tensor = torch.from_numpy(goal_for_model).unsqueeze(0).to(self.device)

        # ---- 3. Run inference ----
        with torch.no_grad():
            output = self.model(batch_tensor, goal_tensor)

        # output.waypoint is [x_right, y_forward] -- convert back
        wp = output.waypoint.cpu().numpy().flatten()       # [x_right, y_forward]
        waypoint = np.array([wp[1], -wp[0]])               # -> [front, left]

        # ---- 4. Arrival decision ----
        arrive = obs.distance_to_goal < self.arrive_distance

        return WaypointPrediction(
            waypoint=waypoint.reshape(1, 2),
            arrive=arrive,
            confidence=float(output.confidence) if hasattr(output, 'confidence') else None,
        )
```

### POI-Goal Wrapper

The only difference is that `predict()` receives a `PoiGoalObservation` with a `poi_name` field. Your model can use this name (e.g. "Starbucks", "City Library") for visual grounding or language-guided navigation:

```python
import numpy as np
import torch

from abotn_evaluator.interface.poi_goal import BasePoiGoalAgent, PoiGoalObservation
from abotn_evaluator.interface.point_goal import WaypointPrediction


class YourPoiGoalWrapper(BasePoiGoalAgent):
    def __init__(
        self,
        model_path: str = "",
        device: str = "cuda:0",
        arrive_distance: float = 2.0,
    ):
        self.device = torch.device(device)
        self.arrive_distance = arrive_distance

        from your_vlm import VisionLanguageNavigator
        self.model = VisionLanguageNavigator.load(model_path)
        self.model = self.model.to(self.device).eval()

    def reset(self):
        self.model.reset()

    def predict(self, obs: PoiGoalObservation) -> WaypointPrediction:
        # The POI name is the key additional field
        poi_name = obs.poi_name  # e.g. "Starbucks", "City Library"

        # Build a text prompt for your VLM
        prompt = f"Navigate to {poi_name}."

        # Prepare image (using front camera only for this example)
        front_img = obs.images["front"]  # (640, 720, 3) uint8 RGB

        # Run your vision-language model
        with torch.no_grad():
            output = self.model.predict(
                image=front_img,
                text=prompt,
                goal_direction=obs.target_position,  # [front, left] metres
            )

        waypoint = np.array(output["waypoint"], dtype=np.float32)
        arrive = obs.distance_to_goal < self.arrive_distance

        return WaypointPrediction(
            waypoint=waypoint.reshape(1, 2),
            arrive=arrive,
        )
```

### How the Constructor Receives Configuration

When you pass `--agent-config agent.yaml`, the runner calls `load_agent_config(path)` and unpacks the resulting dictionary as `**kwargs` to your `__init__`. This means every top-level key in the YAML must match a parameter name in your constructor:

```yaml
# agent.yaml
model_path: /data/checkpoints/nav_model_v3.pth
device: cuda:0
step_size: 0.5
arrive_distance: 0.5
image_size: [224, 224]
```

```python
# These YAML keys map directly to __init__ parameters:
class MyWrapper(BasePointGoalAgent):
    def __init__(self, model_path, device, step_size, arrive_distance, image_size):
        ...
```

If you use `**kwargs` in the constructor instead, all keys will be available in the `kwargs` dict.

### Handling Different Image Formats

`obs.images` values are `(640, 720, 3)` uint8 RGB numpy arrays. Common conversions:

```python
import numpy as np
from PIL import Image

img = obs.images["front"]  # (640, 720, 3) uint8 RGB

# To BGR (OpenCV convention)
bgr = img[:, :, ::-1]

# To PIL Image
pil = Image.fromarray(img)

# To a different resolution
pil_resized = pil.resize((224, 224))
resized_array = np.array(pil_resized)

# To float32 normalized [0, 1]
float_img = img.astype(np.float32) / 255.0

# To PyTorch tensor (B, C, H, W)
import torch
tensor = torch.from_numpy(float_img).permute(2, 0, 1).unsqueeze(0)
```

### Handling Different Coordinate Systems

The benchmark uses `[front, left]` in metres for both `target_position` (input) and `waypoint` (output). Here is how to convert between common conventions:

```python
import numpy as np

target = obs.target_position  # [front, left] in metres

# ---- Input conversion: benchmark -> your model ----

# If your model uses [x_right, y_forward]:
model_input = np.array([-target[1], target[0]])

# If your model uses [forward, right]:
model_input = np.array([target[0], -target[1]])

# If your model uses polar (distance, angle), angle clockwise from front:
distance = np.linalg.norm(target)
angle = -np.arctan2(target[1], target[0])  # negative because left is positive
model_input = np.array([distance, angle])


# ---- Output conversion: your model -> benchmark ----

# If your model outputs [x_right, y_forward]:
model_out = model.predict(...)  # returns [x_right, y_forward]
waypoint = np.array([model_out[1], -model_out[0]])  # -> [front, left]

# If your model outputs [forward, right]:
model_out = model.predict(...)  # returns [forward, right]
waypoint = np.array([model_out[0], -model_out[1]])  # -> [front, left]

# If your model outputs polar (distance, angle), angle clockwise from front:
d, theta = model.predict(...)
waypoint = np.array([d * np.cos(theta), -d * np.sin(theta)])  # -> [front, left]
```

## YAML Configuration

The `--agent-config` flag loads a YAML file whose keys are passed as `**kwargs` to your agent's `__init__`. This keeps model paths, hyperparameters, and device settings out of your code.

### Basic Example

```yaml
# agent_config.yaml
model_path: /data/checkpoints/nav_model_v3.pth
device: cuda:0
step_size: 0.5
arrive_distance: 0.5
image_size: [224, 224]
```

All keys become keyword arguments:

```python
# The runner effectively does:
config = load_agent_config("agent_config.yaml")
# config == {"model_path": "/data/checkpoints/nav_model_v3.pth", "device": "cuda:0", ...}
agent = YourWrapper(**config)
```

### `_base_` Inheritance

A child YAML can extend a base YAML using the `_base_` key. The child's values override the base:

```yaml
# base_config.yaml
model_path: /data/checkpoints/default.pth
device: cuda:0
step_size: 0.5
arrive_distance: 0.5
image_size: [224, 224]
```

```yaml
# experiment_v2.yaml
_base_: base_config.yaml
model_path: /data/checkpoints/v2_finetuned.pth
step_size: 0.3
```

The loader first reads `base_config.yaml`, then overlays the child's keys. The effective config is:

```python
{
    "model_path": "/data/checkpoints/v2_finetuned.pth",  # overridden
    "device": "cuda:0",                                   # from base
    "step_size": 0.3,                                     # overridden
    "arrive_distance": 0.5,                                # from base
    "image_size": [224, 224],                              # from base
}
```

The `_base_` path is resolved relative to the child YAML's directory.

### Mode Resolution

If the config contains a `mode` key and sub-dicts matching mode names, the loader flattens the selected mode's parameters into the top level:

```yaml
# multi_mode_config.yaml
model_path: /data/checkpoints/universal.pth
device: cuda:0
mode: outdoor

outdoor:
  step_size: 0.5
  arrive_distance: 0.5

indoor:
  step_size: 0.3
  arrive_distance: 0.3
```

When `mode` is `"outdoor"`, the effective config becomes:

```python
{
    "model_path": "/data/checkpoints/universal.pth",
    "device": "cuda:0",
    "mode": "outdoor",
    "step_size": 0.5,          # from outdoor sub-dict
    "arrive_distance": 0.5,    # from outdoor sub-dict
}
```

The `indoor` sub-dict is discarded. This lets you maintain a single YAML for both environments.

## Coordinate Conversion

The benchmark uses agent-local `[front, left]` in metres for both `target_position` and `waypoint`. Common conversions:

| Your Model's Coordinate System | Conversion to `[front, left]` |
|:-------------------------------|:------------------------------|
| `[x_right, y_forward]` | `[y_forward, -x_right]` |
| `[forward, right]` | `[forward, -right]` |
| Polar `(r, theta)`, theta clockwise from front | `[r*cos(theta), -r*sin(theta)]` |

**Image format**: `obs.images` values are `(640, 720, 3)` uint8 RGB. Convert as needed:
- BGR: `img[:, :, ::-1]`
- PIL: `Image.fromarray(img)`
- Different resolution: resize in your wrapper

## CLI Loading

### Module Path Format

The `--agent-module` argument uses the format `package.module:ClassName`:

```
--agent-module my_project.agents.wrapper:MyNavigationAgent
              ├── module path ──────────┤ ├── class name ──┤
```

- **Module path** (left of `:`): A standard Python dotted module path, exactly as you would use in an `import` statement. For example, `my_project.agents.wrapper` corresponds to the file `my_project/agents/wrapper.py`.
- **Class name** (right of `:`): The name of the class to instantiate from that module.

The colon must appear exactly once. Both parts must be non-empty.

### Python Import Resolution

The runner uses `importlib.import_module()` to load your module. Python finds modules through these mechanisms:

1. **Current working directory** -- Python automatically adds the current directory to `sys.path`, so running the command from your project root is often sufficient:
   ```bash
   cd /path/to/your_project
   python -m abotn_evaluator.point_goal.runner \
       --agent-module my_agents.wrapper:MyAgent ...
   ```

2. **Installed packages** -- If your model is a pip-installable package, install it first:
   ```bash
   pip install -e /path/to/your_model_repo
   python -m abotn_evaluator.point_goal.runner \
       --agent-module your_model_package.agent:YourAgent ...
   ```

3. **PYTHONPATH** -- Prepend your project directory to `PYTHONPATH`:
   ```bash
   PYTHONPATH=/path/to/your_project:$PYTHONPATH \
   python -m abotn_evaluator.point_goal.runner \
       --agent-module my_agents.wrapper:MyAgent ...
   ```

### Full CLI Example

```bash
python -m abotn_evaluator.point_goal.runner \
    --agent-module your_project.wrapper:YourWrapper \
    --agent-config your_agent.yaml \
    --data-dir /path/to/annotations \
    --map-dir /path/to/occmaps \
    --render-url http://localhost:7036/render_gs \
    --mode outdoor \
    --max-steps 100 \
    --output-dir ./eval_output \
    --provide-history \
    --enable-visualization
```

## Debugging Tips

### 1. Start with RandomAgent

Always verify the pipeline works before plugging in your model. See [Step 1](#step-1-verify-the-pipeline-with-randomagent) above.

### 2. Use `--skip-metrics` for Fast Iteration

Metrics analysis adds overhead. Skip it while debugging your wrapper:

```bash
python -m abotn_evaluator.point_goal.runner \
    --agent-module my_project.wrapper:MyAgent \
    --data-dir /path/to/annotations --map-dir /path/to/occmaps \
    --render-url http://localhost:7036/render_gs \
    --mode outdoor --max-steps 5 --skip-metrics
```

### 3. Limit Steps with `--max-steps`

Use `--max-steps 5` or `--max-steps 10` to test just a few steps per episode. This reveals basic issues (wrong coordinate system, image format problems, crash on first step) without waiting for a full run.

### 4. Enable Visualization

Use `--enable-visualization` to overlay debug data on rendered images. This requires your agent to return visualization data in `WaypointPrediction.extra`:

```python
return WaypointPrediction(
    waypoint=wp,
    arrive=False,
    extra={"affordance_pixel": {"front": [360, 320]}},
)
```

The evaluator saves annotated images to the output directory, making it easy to verify that your agent is looking at the right things.

### 5. Check Render Server Health

The render server must be running before you start evaluation. Verify it:

```bash
curl http://localhost:7036/ping
```

If this does not return a 200 response, the server is down or unreachable. Common issues:
- The server is not started yet.
- The server is on a different port. Check the `--render-url` you pass.
- The server is on a different machine. Make sure the URL is correct and the port is open.

### 6. Resume Failed Runs

If a run crashes partway through (network error, OOM, etc.), resume from where it left off instead of starting over. The `--resume-dir` flag scans for completed `result.json` files and skips those tasks:

```bash
python -m abotn_evaluator.point_goal.runner \
    --agent-module my_project.wrapper:MyAgent \
    --agent-config my_config.yaml \
    --data-dir /path/to/annotations --map-dir /path/to/occmaps \
    --render-url http://localhost:7036/render_gs \
    --mode outdoor \
    --resume-dir ./eval_output/20240101_120000
```

### 7. Analyze Results Mid-Run

You do not have to wait for a run to finish. The metrics module can analyze whatever results exist so far:

```bash
python -m abotn_evaluator.point_goal.metrics \
    --result-dir ./eval_output/20240101_120000 \
    --mode outdoor
```

## Visualization Debugging

Return debug data in `WaypointPrediction.extra` to overlay visualizations on rendered images when `enable_visualization=True` (CLI: `--enable-visualization`).

```python
return WaypointPrediction(
    waypoint=wp,
    arrive=False,
    extra={"affordance_pixel": {"front": [360, 320]}},
)
```

The built-in `affordance_pixel` handler marks pixels on the rendered view. See [API Reference](api-reference.md#visualization-extension) for details and how to register custom handlers.

## Reference Templates

The repository includes example agents in `agent_examples/`:

- `point_goal_wrapper.py` -- Point-Goal wrapper template with extensive inline comments
- `poi_goal_wrapper.py` -- POI-Goal wrapper template
- `point_goal_random.py` -- Heuristic agent for smoke-testing the evaluation pipeline

These templates are designed to be copied into your own project and modified. They contain `TODO` markers at every point where you need to insert your model-specific logic.
