"""Agent Wrapper 模板：将任意导航模型接入 ABotN-PointBench 评测

本文件是一个模板。你需要复制它到你自己的项目中，然后修改标记了
"TODO" 的位置，填入你自己的模型加载和推理逻辑。

== 你需要做的事情 ==

  1. 在 __init__() 中加载你的模型（方式由你决定）
  2. 在 predict() 中把 Observation 转换为你模型的输入
  3. 在 predict() 中把你模型的输出转换为 WaypointPrediction

== 你不需要关心的事情 ==

  - 渲染图像（evaluator 负责）
  - 碰撞检测（evaluator 负责）
  - 指标计算（evaluator 负责）
  - 评测循环（evaluator 负责）

== 使用方式 ==

  python -m abotn_evaluator.point_goal.runner \\
      --agent-module your_package.your_wrapper:YourAgentClass \\
      --data-dir /path/to/data \\
      --render-url http://localhost:7036/render_gs \\
      --output-dir ./results

  如果你的 agent 需要额外参数，通过 --agent-config 传入 YAML：
  python -m abotn_evaluator.point_goal.runner \\
      --agent-module your_package.your_wrapper:YourAgentClass \\
      --agent-config your_config.yaml \\
      ...

  YAML 中的所有 key-value 会作为 **kwargs 传入 __init__()。

== Observation 字段 ==

  images           Dict[str, ndarray]   三视角 RGB，keys: left/front/right
                                        shape: (640,720,3) uint8 RGB
  target_position  ndarray(2,)          目标 [front, left] 米
  distance_to_goal float                到目标距离（米）
  position         ndarray(3,)          世界坐标 [x,y,z]
  rotation         ndarray(4,4)         camera-to-world 矩阵
  heading          float                yaw（弧度）
  step_count       int                  当前步数
  history_images   List[Dict] | None    历史帧（需 --provide-history）
  history_poses    List[ndarray] | None 历史 pose
  occ_map          ndarray | None       占用图（需 --provide-occ-map）

== WaypointPrediction 字段 ==

  waypoint   ndarray   [front, left] 米，shape (2,)/(1,2)/(N,2)
  arrive     bool      是否到达（True 时 evaluator 检查距离）
  directions ndarray   可选，方向向量
  extra      dict      可选，可视化数据（如 affordance_pixel）
"""

import numpy as np

from abotn_evaluator.interface.point_goal import (
    BasePointGoalAgent,
    Observation,
    WaypointPrediction,
)


class PointGoalWrapperTemplate(BasePointGoalAgent):
    """Point-Goal agent wrapper 模板。

    __init__ 接收 **kwargs——如果通过 --agent-config 传入 YAML，
    YAML 中的所有字段会作为关键字参数传入。你也可以不用 YAML，
    在这里硬编码路径、用环境变量、或任何你喜欢的方式。
    """

    def __init__(self, **kwargs):
        # TODO: 加载你的模型。以下是几种常见方式，选一种即可：

        # 方式 1：从 --agent-config YAML 中读取参数
        # model_path = kwargs.get("model_path", "")

        # 方式 2：从环境变量读取
        # import os
        # model_path = os.environ.get("MY_MODEL_PATH", "")

        # 方式 3：直接硬编码
        # model_path = "/path/to/my/model"

        # 方式 4：从你自己的配置系统读取
        # from my_project.config import load_config
        # cfg = load_config("my_config.json")
        # model_path = cfg["model_path"]

        # TODO: 如果你的模型在外部仓库，确保它可以被 import：
        # 方式 A：运行时设置 PYTHONPATH
        #   PYTHONPATH=/path/to/my_repo:$PYTHONPATH python -m ...
        # 方式 B：在这里动态添加
        #   import sys; sys.path.insert(0, "/path/to/my_repo")
        # 方式 C：把 wrapper 文件放到你自己的仓库中

        # TODO: 加载模型
        # from my_model import MyNavigator
        # self.model = MyNavigator.load(model_path)

        self._step = 0
        print(f"[PointGoalWrapperTemplate] kwargs={kwargs}")

    def reset(self):
        self._step = 0
        # TODO: 重置你模型的内部状态（如果有的话）
        # self.model.reset()

    def predict(self, observation: Observation) -> WaypointPrediction:
        self._step += 1

        # ===========================================================
        # 1. 从 Observation 提取你需要的输入
        # ===========================================================

        # 图像
        front = observation.images["front"]    # (640, 720, 3) uint8 RGB
        left = observation.images["left"]
        right = observation.images["right"]

        # 目标位置（agent-local 坐标，[front, left]，单位米）
        target = observation.target_position   # ndarray(2,)

        # 其他可用信息
        dist = observation.distance_to_goal    # float，到目标距离
        pos = observation.position             # [x, y, z] 世界坐标
        rot = observation.rotation             # 4x4 矩阵
        yaw = observation.heading              # 弧度
        step = observation.step_count          # 当前步数

        # ===========================================================
        # 2. 调用你的模型（TODO：替换这里）
        # ===========================================================

        # my_output = self.model.predict(front, left, right, target, ...)
        # waypoint = my_output["waypoint"]   # 你模型输出的 waypoint
        # done = my_output["done"]           # 你模型判断的到达状态

        # 如果你的模型输出坐标系不同，在这里转换：
        # 例如你的模型用 [right, forward]：
        #   waypoint = np.array([my_out[1], -my_out[0]])
        # 例如你的模型输出极坐标 (distance, angle)：
        #   waypoint = np.array([d * np.cos(a), d * np.sin(a)])

        # 占位：简单启发式
        direction = target / (np.linalg.norm(target) + 1e-6)
        waypoint = direction * min(0.5, dist)
        done = dist < 0.5

        # ===========================================================
        # 3. 封装为 WaypointPrediction 返回
        # ===========================================================

        return WaypointPrediction(
            waypoint=waypoint.reshape(1, 2),
            arrive=done,
            # directions=...,    # 可选：方向向量
            # extra={            # 可选：可视化（配合 --enable-visualization）
            #     "affordance_pixel": {"front": [360, 320]},
            # },
        )
