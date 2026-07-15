# 自定义 Agent 接入

[English](../custom-agents.md) | [中文](custom-agents.md)

本指南说明如何将自有导航模型接入 ABotN-Bench 评测，涵盖从首次冒烟测试到完整评测运行的全流程。

## 接入方式

| 方式 | 适用场景 |
|:-----|:---------|
| 直接使用 | 模型已实现 `reset()` / `predict()`，输入输出与接口一致 |
| Wrapper 模式 | 模型使用不同的坐标系、图像格式或输出结构 |
| CLI + Wrapper | 同 wrapper，通过 `--agent-module` 加载，适合批量 / 多 GPU 运行 |

所有方式均要求 agent 满足 [API 参考](api-reference.md) 中定义的接口。

## 分步工作流

本节指导你完成从"我有一个训练好的模型"到"我拿到了评测指标"的完整流程。

### 第 1 步：用 RandomAgent 验证流程

在接入你自己的模型之前，先用内置的 `RandomPointGoalAgent` 确认渲染服务器、数据目录和评测流程能端到端正常工作：

```bash
# 先确认渲染服务器已启动
curl http://localhost:7036/ping   # 应返回 200

# 用随机 agent 进行快速冒烟测试
python -m abotn_evaluator.point_goal.runner \
    --agent-module agent_examples.point_goal_random:RandomPointGoalAgent \
    --data-dir /path/to/annotations \
    --map-dir /path/to/occmaps \
    --render-url http://localhost:7036/render_gs \
    --mode outdoor \
    --max-steps 10
```

如果在 `./eval_output/` 下生成了 per-task 的 `result.json` 和最终的 `eval_summary.json`，说明流程正常。如果失败，请先解决基础设施问题，再继续后续步骤。

### 第 2 步：编写 Wrapper 类

创建一个 Python 文件，将你的模型封装到 `BasePointGoalAgent`（或 `BasePoiGoalAgent`）接口之后。详细示例见下方 [Wrapper 模式](#wrapper-模式接口不一致)。

最小骨架：

```python
from abotn_evaluator.interface.point_goal import BasePointGoalAgent, Observation, WaypointPrediction
import numpy as np

class MyAgent(BasePointGoalAgent):
    def __init__(self, model_path: str = "", device: str = "cuda:0"):
        # 在此加载你的模型
        pass

    def reset(self):
        pass

    def predict(self, obs: Observation) -> WaypointPrediction:
        # 你的推理逻辑
        waypoint = np.array([0.5, 0.0])
        return WaypointPrediction(waypoint=waypoint, arrive=False)
```

### 第 3 步：用 `--skip-metrics` 和 `--max-steps` 快速迭代

调试 wrapper 时，避免等待完整的指标计算。使用 `--skip-metrics` 和较小的 `--max-steps`：

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

这样每个 episode 只运行 5 步且跳过聚合分析，几秒内即可获得反馈。检查 per-task 的 `result.json` 文件，验证你的 waypoint 是否合理。

### 第 4 步：完整评测运行

当你的 wrapper 输出合理后，执行完整评测：

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

### 第 5 步：分析结果

runner 会自动生成 `eval_summary.json`。你也可以对结果目录单独运行指标分析——在运行中途检查进度或用不同参数重新分析时非常有用：

```bash
# 分析之前的运行结果（或仍在运行中的结果）
python -m abotn_evaluator.point_goal.metrics \
    --result-dir ./eval_output/20240101_120000 \
    --mode outdoor \
    --per-scene
```

## 直接使用（接口已对齐）

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

## Wrapper 模式（接口不一致）

模型输入输出与基准接口不一致时，编写 wrapper 类进行转换。Wrapper 是一个轻量适配层——所有实际推理在你的模型内部完成。

### Point-Goal Wrapper

一个更真实的 wrapper 示例，处理模型加载到指定 GPU 设备、图像预处理和坐标系转换：

```python
import numpy as np
import torch
from PIL import Image

from abotn_evaluator.interface.point_goal import BasePointGoalAgent, Observation, WaypointPrediction


class YourPointGoalWrapper(BasePointGoalAgent):
    """将 PyTorch 导航模型封装为 ABotN-Bench 评测所需的接口。

    构造函数参数来自 --agent-config YAML 文件。
    YAML 中的每个 key 都会成为此处的关键字参数。
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

        # 加载模型 checkpoint
        from your_model import NavigationNet
        self.model = NavigationNet.load_from_checkpoint(model_path)
        self.model = self.model.to(self.device).eval()

    def reset(self):
        """重置 episode 级别的状态（隐藏状态、历史缓冲区等）。"""
        self.model.reset_hidden_state()

    def predict(self, obs: Observation) -> WaypointPrediction:
        # ---- 1. 准备图像 ----
        # obs.images 值为 (640, 720, 3) uint8 RGB numpy 数组。
        # 缩放并归一化以适配你的模型。
        images = []
        for cam in ("front", "left", "right"):
            img = obs.images[cam]                          # (640, 720, 3) RGB
            pil = Image.fromarray(img)
            pil = pil.resize(self.image_size)              # 缩放到模型输入尺寸
            arr = np.array(pil, dtype=np.float32) / 255.0  # 归一化到 [0, 1]
            images.append(arr)

        # 堆叠为 (B, 3, H, W) tensor
        batch = np.stack(images, axis=0)                   # (3, H, W, 3)
        batch = np.transpose(batch, (0, 3, 1, 2))         # (3, 3, H, W)
        batch_tensor = torch.from_numpy(batch).unsqueeze(0).to(self.device)

        # ---- 2. 准备目标 ----
        target = obs.target_position  # [front, left] 米

        # 示例：你的模型使用 [x_right, y_forward]
        goal_for_model = np.array([-target[1], target[0]], dtype=np.float32)
        goal_tensor = torch.from_numpy(goal_for_model).unsqueeze(0).to(self.device)

        # ---- 3. 推理 ----
        with torch.no_grad():
            output = self.model(batch_tensor, goal_tensor)

        # output.waypoint 为 [x_right, y_forward]——转换回来
        wp = output.waypoint.cpu().numpy().flatten()       # [x_right, y_forward]
        waypoint = np.array([wp[1], -wp[0]])               # -> [front, left]

        # ---- 4. 到达判定 ----
        arrive = obs.distance_to_goal < self.arrive_distance

        return WaypointPrediction(
            waypoint=waypoint.reshape(1, 2),
            arrive=arrive,
            confidence=float(output.confidence) if hasattr(output, 'confidence') else None,
        )
```

### POI-Goal Wrapper

唯一区别是 `predict()` 接收 `PoiGoalObservation`，额外包含 `poi_name` 字段。你的模型可以利用这个名称（如 "Starbucks"、"城市图书馆"）进行视觉定位或语言引导导航：

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
        # POI 名称是最关键的额外字段
        poi_name = obs.poi_name  # 如 "Starbucks"、"城市图书馆"

        # 为你的 VLM 构建文本提示
        prompt = f"导航到 {poi_name}。"

        # 准备图像（此例仅使用前视相机）
        front_img = obs.images["front"]  # (640, 720, 3) uint8 RGB

        # 运行视觉语言模型
        with torch.no_grad():
            output = self.model.predict(
                image=front_img,
                text=prompt,
                goal_direction=obs.target_position,  # [front, left] 米
            )

        waypoint = np.array(output["waypoint"], dtype=np.float32)
        arrive = obs.distance_to_goal < self.arrive_distance

        return WaypointPrediction(
            waypoint=waypoint.reshape(1, 2),
            arrive=arrive,
        )
```

### 构造函数如何接收配置

当你传入 `--agent-config agent.yaml` 时，runner 会调用 `load_agent_config(path)` 并将结果字典解包为 `**kwargs` 传入你的 `__init__`。这意味着 YAML 中的每个顶层 key 必须匹配你构造函数的参数名：

```yaml
# agent.yaml
model_path: /data/checkpoints/nav_model_v3.pth
device: cuda:0
step_size: 0.5
arrive_distance: 0.5
image_size: [224, 224]
```

```python
# 这些 YAML key 直接映射到 __init__ 参数：
class MyWrapper(BasePointGoalAgent):
    def __init__(self, model_path, device, step_size, arrive_distance, image_size):
        ...
```

如果你在构造函数中使用 `**kwargs`，所有 key 都可通过 `kwargs` 字典获取。

### 处理不同的图像格式

`obs.images` 值为 `(640, 720, 3)` uint8 RGB numpy 数组。常见转换：

```python
import numpy as np
from PIL import Image

img = obs.images["front"]  # (640, 720, 3) uint8 RGB

# 转为 BGR（OpenCV 约定）
bgr = img[:, :, ::-1]

# 转为 PIL Image
pil = Image.fromarray(img)

# 缩放到其他分辨率
pil_resized = pil.resize((224, 224))
resized_array = np.array(pil_resized)

# 转为 float32 归一化 [0, 1]
float_img = img.astype(np.float32) / 255.0

# 转为 PyTorch tensor (B, C, H, W)
import torch
tensor = torch.from_numpy(float_img).permute(2, 0, 1).unsqueeze(0)
```

### 处理不同的坐标系

基准使用 `[front, left]`（单位米）作为 `target_position`（输入）和 `waypoint`（输出）的坐标系。以下是常见约定之间的转换方法：

```python
import numpy as np

target = obs.target_position  # [front, left] 米

# ---- 输入转换：基准 -> 你的模型 ----

# 如果你的模型使用 [x_right, y_forward]：
model_input = np.array([-target[1], target[0]])

# 如果你的模型使用 [forward, right]：
model_input = np.array([target[0], -target[1]])

# 如果你的模型使用极坐标 (distance, angle)，angle 从正前方顺时针：
distance = np.linalg.norm(target)
angle = -np.arctan2(target[1], target[0])  # 取负因为 left 为正
model_input = np.array([distance, angle])


# ---- 输出转换：你的模型 -> 基准 ----

# 如果你的模型输出 [x_right, y_forward]：
model_out = model.predict(...)  # 返回 [x_right, y_forward]
waypoint = np.array([model_out[1], -model_out[0]])  # -> [front, left]

# 如果你的模型输出 [forward, right]：
model_out = model.predict(...)  # 返回 [forward, right]
waypoint = np.array([model_out[0], -model_out[1]])  # -> [front, left]

# 如果你的模型输出极坐标 (distance, angle)，angle 从正前方顺时针：
d, theta = model.predict(...)
waypoint = np.array([d * np.cos(theta), -d * np.sin(theta)])  # -> [front, left]
```

## YAML 配置

`--agent-config` 标志加载一个 YAML 文件，其 key 作为 `**kwargs` 传入你的 agent 的 `__init__`。这样可以将模型路径、超参数和设备设置从代码中分离出来。

### 基本示例

```yaml
# agent_config.yaml
model_path: /data/checkpoints/nav_model_v3.pth
device: cuda:0
step_size: 0.5
arrive_distance: 0.5
image_size: [224, 224]
```

所有 key 成为关键字参数：

```python
# runner 实际执行的逻辑：
config = load_agent_config("agent_config.yaml")
# config == {"model_path": "/data/checkpoints/nav_model_v3.pth", "device": "cuda:0", ...}
agent = YourWrapper(**config)
```

### `_base_` 继承

子 YAML 可以通过 `_base_` 键继承基础 YAML。子配置的值会覆盖基础配置：

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

加载器先读取 `base_config.yaml`，然后覆盖子配置的 key。最终有效配置为：

```python
{
    "model_path": "/data/checkpoints/v2_finetuned.pth",  # 被覆盖
    "device": "cuda:0",                                   # 来自基础配置
    "step_size": 0.3,                                     # 被覆盖
    "arrive_distance": 0.5,                                # 来自基础配置
    "image_size": [224, 224],                              # 来自基础配置
}
```

`_base_` 路径相对于子 YAML 所在目录解析。

### 模式解析

如果配置中包含 `mode` 键和与模式名称匹配的子字典，加载器会将选中模式的参数展平到顶层：

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

当 `mode` 为 `"outdoor"` 时，最终有效配置为：

```python
{
    "model_path": "/data/checkpoints/universal.pth",
    "device": "cuda:0",
    "mode": "outdoor",
    "step_size": 0.5,          # 来自 outdoor 子字典
    "arrive_distance": 0.5,    # 来自 outdoor 子字典
}
```

`indoor` 子字典会被丢弃。这使你可以用一份 YAML 管理两种环境的配置。

## 坐标系转换

基准使用 agent 局部坐标系 `[front, left]`，单位米，同时适用于 `target_position` 和 `waypoint`。常见转换：

| 模型坐标系 | 转换为 `[front, left]` |
|:-----------|:----------------------|
| `[x_right, y_forward]` | `[y_forward, -x_right]` |
| `[forward, right]` | `[forward, -right]` |
| 极坐标 `(r, theta)`，theta 从正前方顺时针 | `[r*cos(theta), -r*sin(theta)]` |

**图像格式**：`obs.images` 值为 `(640, 720, 3)` uint8 RGB。按需转换：
- BGR：`img[:, :, ::-1]`
- PIL：`Image.fromarray(img)`
- 其他分辨率：在 wrapper 中自行 resize

## 通过 CLI 加载

### 模块路径格式

`--agent-module` 参数使用 `package.module:ClassName` 格式：

```
--agent-module my_project.agents.wrapper:MyNavigationAgent
              ├── 模块路径 ──────────────┤ ├── 类名 ────────┤
```

- **模块路径**（`:` 左侧）：标准 Python 点分模块路径，与 `import` 语句中使用的格式相同。例如 `my_project.agents.wrapper` 对应文件 `my_project/agents/wrapper.py`。
- **类名**（`:` 右侧）：要从该模块实例化的类名。

冒号必须恰好出现一次，两部分都不能为空。

### Python 导入解析

runner 使用 `importlib.import_module()` 加载你的模块。Python 通过以下机制查找模块：

1. **当前工作目录** -- Python 自动将当前目录添加到 `sys.path`，所以从项目根目录运行命令通常就够了：
   ```bash
   cd /path/to/your_project
   python -m abotn_evaluator.point_goal.runner \
       --agent-module my_agents.wrapper:MyAgent ...
   ```

2. **已安装的包** -- 如果你的模型是可 pip 安装的包，先安装：
   ```bash
   pip install -e /path/to/your_model_repo
   python -m abotn_evaluator.point_goal.runner \
       --agent-module your_model_package.agent:YourAgent ...
   ```

3. **PYTHONPATH** -- 将你的项目目录添加到 `PYTHONPATH`：
   ```bash
   PYTHONPATH=/path/to/your_project:$PYTHONPATH \
   python -m abotn_evaluator.point_goal.runner \
       --agent-module my_agents.wrapper:MyAgent ...
   ```

### 完整 CLI 示例

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

## 调试技巧

### 1. 从 RandomAgent 开始

在接入你自己的模型之前，务必先验证流程可用。参见上方[第 1 步](#第-1-步用-randomagent-验证流程)。

### 2. 用 `--skip-metrics` 快速迭代

指标分析会增加开销。调试 wrapper 时可跳过：

```bash
python -m abotn_evaluator.point_goal.runner \
    --agent-module my_project.wrapper:MyAgent \
    --data-dir /path/to/annotations --map-dir /path/to/occmaps \
    --render-url http://localhost:7036/render_gs \
    --mode outdoor --max-steps 5 --skip-metrics
```

### 3. 用 `--max-steps` 限制步数

使用 `--max-steps 5` 或 `--max-steps 10` 仅测试几步。这能快速发现基本问题（坐标系错误、图像格式问题、首步崩溃），而无需等待完整运行。

### 4. 启用可视化

使用 `--enable-visualization` 在渲染图像上叠加调试数据。需要你的 agent 在 `WaypointPrediction.extra` 中返回可视化数据：

```python
return WaypointPrediction(
    waypoint=wp,
    arrive=False,
    extra={"affordance_pixel": {"front": [360, 320]}},
)
```

evaluator 会将标注后的图像保存到输出目录，方便验证你的 agent 是否关注了正确的区域。

### 5. 检查渲染服务器状态

评测前必须确保渲染服务器已启动。验证方法：

```bash
curl http://localhost:7036/ping
```

如果没有返回 200 响应，说明服务器未启动或不可达。常见问题：
- 服务器尚未启动。
- 服务器在不同端口。检查你传入的 `--render-url`。
- 服务器在不同机器上。确认 URL 正确且端口已开放。

### 6. 恢复失败的运行

如果运行中途崩溃（网络错误、OOM 等），可以从中断处恢复而不是重新开始。`--resume-dir` 标志会扫描已完成的 `result.json` 文件并跳过这些任务：

```bash
python -m abotn_evaluator.point_goal.runner \
    --agent-module my_project.wrapper:MyAgent \
    --agent-config my_config.yaml \
    --data-dir /path/to/annotations --map-dir /path/to/occmaps \
    --render-url http://localhost:7036/render_gs \
    --mode outdoor \
    --resume-dir ./eval_output/20240101_120000
```

### 7. 运行中途分析结果

不需要等待运行结束。指标模块可以分析目前已有的任何结果：

```bash
python -m abotn_evaluator.point_goal.metrics \
    --result-dir ./eval_output/20240101_120000 \
    --mode outdoor
```

## 可视化调试

在 `WaypointPrediction.extra` 中返回调试数据，`enable_visualization=True`（CLI：`--enable-visualization`）时可在渲染图像上叠加可视化。

```python
return WaypointPrediction(
    waypoint=wp,
    arrive=False,
    extra={"affordance_pixel": {"front": [360, 320]}},
)
```

内置 `affordance_pixel` 处理器在渲染视图上标记像素点。详见 [API 参考](api-reference.md#可视化扩展)。

## 参考模板

仓库 `agent_examples/` 目录包含示例：

- `point_goal_wrapper.py` -- Point-Goal wrapper 模板，含详尽行内注释
- `poi_goal_wrapper.py` -- POI-Goal wrapper 模板
- `point_goal_random.py` -- 启发式 agent，用于冒烟测试评测流程

这些模板设计为复制到你自己的项目中进行修改。它们在每个需要插入模型特定逻辑的位置都标有 `TODO` 标记。
