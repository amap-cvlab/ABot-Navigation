"""Agent Wrapper 模板（POI-Goal）：将任意导航模型接入 ABotN-POIBench 评测

与 Point-Goal wrapper 的唯一区别：Observation 多一个 poi_name 字段。
你的模型需要利用 POI 名称（如 "Starbucks"）在视觉中定位目标。

其余字段（images, target_position 等）完全一致。
完整字段说明参见 point_goal_wrapper.py 或 docs/API.md。

== 使用方式 ==

  python -m abotn_evaluator.poi_goal.runner \\
      --agent-module your_package.your_wrapper:YourPoiAgentClass \\
      --data-dir /path/to/poibench/trajectory \\
      --render-url http://localhost:7036/render_gs \\
      --output-dir ./results \\
      --arrive-threshold 2.0 \\
      --collision-mode hard
"""

import numpy as np

from abotn_evaluator.interface.point_goal import WaypointPrediction
from abotn_evaluator.interface.poi_goal import (
    BasePoiGoalAgent,
    PoiGoalObservation,
)


class PoiGoalWrapperTemplate(BasePoiGoalAgent):
    """POI-Goal agent wrapper 模板。

    与 Point-Goal 版本结构相同，但 predict() 接收的 observation
    多一个 poi_name: str 字段。到达阈值通常为 2.0m。
    """

    def __init__(self, **kwargs):
        # TODO: 加载你的模型（方式同 point_goal_wrapper.py）
        self._step = 0
        print(f"[PoiGoalWrapperTemplate] kwargs={kwargs}")

    def reset(self):
        self._step = 0

    def predict(self, observation: PoiGoalObservation) -> WaypointPrediction:
        self._step += 1

        # ===========================================================
        # POI-Goal 特有：目标 POI 名称
        # ===========================================================
        poi_name = observation.poi_name    # str，如 "Starbucks"、"全家便利店"

        # 图像和其余字段与 Point-Goal 完全一致
        front = observation.images["front"]
        target = observation.target_position
        dist = observation.distance_to_goal

        # ===========================================================
        # TODO: 调用你的模型
        # ===========================================================

        # my_output = self.model.navigate_to_poi(
        #     images=[front, ...],
        #     poi_name=poi_name,
        #     target_direction=target,
        # )

        # 占位：简单启发式
        direction = target / (np.linalg.norm(target) + 1e-6)
        waypoint = direction * min(0.5, dist)

        # POI-Goal 到达阈值通常为 2.0m
        done = dist < 2.0

        return WaypointPrediction(
            waypoint=waypoint.reshape(1, 2),
            arrive=done,
        )
