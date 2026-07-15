"""Core evaluator for POI-goal navigation.

Extends the point-goal evaluator with POI-specific behaviour:

* ``arrive_threshold`` defaults to 2.0m (POI targets are typically at
  building facades where the agent cannot get as close).
* SPL uses ``effective_shortest = max(shortest - arrive_threshold, 1e-3)``
  to avoid the SPL distribution collapsing to {0, 1}.
* Collision detection supports off/soft/hard modes with a terminal-zone
  exemption inside the arrive_threshold circle around the goal.
* Observations carry a ``poi_name`` field via :class:`PoiGoalObservation`.
* The evaluation summary includes per-POI statistics.
"""

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from abotn_evaluator.interface.point_goal import WaypointPrediction
from abotn_evaluator.point_goal.evaluator import (
    EvalConfig as PointGoalEvalConfig,
    PointGoalEvaluator,
    _make_serialisable,
    get_z_value,
)
from abotn_evaluator.scene import Episode, GaussianScene, Task
from abotn_evaluator.render_client import RenderFailureError
from abotn_evaluator.interface.poi_goal import BasePoiGoalAgent, PoiGoalObservation


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PoiGoalEvalConfig(PointGoalEvalConfig):
    """Evaluation configuration for POI-goal navigation.

    Extends the point-goal config with collision detection modes and a
    larger default arrival threshold suitable for POI targets.

    Attributes
    ----------
    arrive_threshold : float
        Distance (metres) at which the agent is considered to have reached
        the POI goal.  Default 2.0m (larger than point-goal's 0.5m because
        POI targets are typically on building facades).
    collision_mode : str
        Collision handling mode:
        ``"off"`` -- no collision detection (default for point-goal);
        ``"soft"`` -- count collisions but do not affect success or
        termination;
        ``"hard"`` -- terminate on first non-exempt collision, marking the
        task as failed with ``status='collision'``.
    robot_radius : float
        Robot radius (metres) for circular footprint collision detection.
        The occupancy map is inflated by this radius.  ``0.0`` means only
        the centre pixel is checked.
    enable_collision_check : bool
        Deprecated compatibility flag.  If ``collision_mode`` is ``"off"``
        and this is ``True``, collision_mode is treated as ``"soft"``.
    occ_obstacle_polarity : str
        How to interpret occupancy map pixel values:
        ``"dark"`` -- dark pixels are obstacles;
        ``"light"`` -- light pixels are obstacles.
    occ_dark_threshold : int
        Greyscale threshold for obstacle detection.
    """

    arrive_threshold: float = 2.0
    occ_dilation_meters: float = 0.5
    collision_mode: str = "hard"
    robot_radius: float = 0.0
    enable_collision_check: bool = False
    occ_obstacle_polarity: str = "dark"
    occ_dark_threshold: int = 64

    def resolve_collision_mode(self) -> str:
        """Resolve the effective collision mode, handling legacy flags.

        If ``collision_mode`` is ``"off"`` but ``enable_collision_check``
        is ``True``, the effective mode is ``"soft"`` for backward
        compatibility.
        """
        mode = (self.collision_mode or "off").lower()
        if mode not in ("off", "soft", "hard"):
            raise ValueError(
                f"invalid collision_mode={self.collision_mode!r}, "
                f"must be one of off/soft/hard"
            )
        if mode == "off" and self.enable_collision_check:
            return "soft"
        return mode


# ============================================================================
# Evaluator
# ============================================================================

class PoiGoalEvaluator(PointGoalEvaluator):
    """POI-goal navigation evaluator for Gaussian splatting scenes.

    Extends :class:`PointGoalEvaluator` with:

    * POI name in observations via :class:`PoiGoalObservation`.
    * Collision detection with off/soft/hard modes and terminal-zone
      exemption.
    * Modified SPL that subtracts ``arrive_threshold`` from the shortest
      path to avoid degenerate {0, 1} distributions.
    * Per-POI statistics in the evaluation summary.

    Parameters
    ----------
    scene : GaussianScene
        The scene manager with loaded episodes.
    renderer : object
        A Gaussian splatting renderer instance.
    config : PoiGoalEvalConfig or None
        Evaluation configuration.  Defaults to :class:`PoiGoalEvalConfig`.
    output_dir : str
        Directory for saving results and optional images.
    """

    def __init__(
        self,
        scene: GaussianScene,
        renderer: Any,
        config: Optional[PoiGoalEvalConfig] = None,
        output_dir: str = "./eval_output",
    ) -> None:
        resolved_config = config or PoiGoalEvalConfig()
        super().__init__(
            scene=scene,
            renderer=renderer,
            config=resolved_config,
            output_dir=output_dir,
        )

    # ------------------------------------------------------------------
    # Per-task evaluation (overridden for POI-specific logic)
    # ------------------------------------------------------------------

    def _evaluate_task(
        self,
        agent: BasePoiGoalAgent,
        episode: Episode,
        task: Task,
        short_memory: Any,
        episode_dir: str,
    ) -> dict:
        """Evaluate a single POI-goal task.

        Differences from the point-goal version:
        * Reads ``task.goal_label`` as the POI name.
        * Uses off/soft/hard collision modes with terminal-zone exemption.
        * SPL formula subtracts ``arrive_threshold`` from shortest path.
        * Result dict includes POI-specific metrics.
        """
        poi_name = getattr(task, "goal_label", None) or "unknown"

        task_dir = os.path.join(episode_dir, task.task_id)
        os.makedirs(task_dir, exist_ok=True)
        render_image_path = os.path.join(task_dir, "render_images")
        os.makedirs(render_image_path, exist_ok=True)

        agent.reset()
        short_memory.reset()
        scene_id = episode.episode_id

        # Build the reference path
        try:
            point_path = self._build_pointgoal_path(task)
        except Exception as exc:
            print(f"[PoiGoal] build path failed for {task.task_id}: {exc}")
            result = self._make_error_result(episode, task)
            result["target_label"] = poi_name
            result["gt_taget_instance"] = f"poi_goal:{poi_name}"
            self._save_task_result(result, task_dir)
            return result

        start_pose = self._pose_dict_to_matrix(point_path[0])
        goal_pose = self._pose_dict_to_matrix(point_path[-1])
        self._adjust_pose_z(start_pose, episode)
        self._adjust_pose_z(goal_pose, episode)
        goal_xy = np.array(
            [point_path[-1]["x"], point_path[-1]["y"]], dtype=np.float32
        )

        # GT path length
        shortest_path_length = self.compute_path_length(point_path)

        # Render initial images
        try:
            start_images = self._render_images(
                start_pose, scene_id, render_image_path, short_memory.frame_count
            )
        except RenderFailureError as exc:
            print(f"[PoiGoal] initial render failed for {task.task_id}: {exc}")
            result = self._make_error_result(episode, task)
            result["target_label"] = poi_name
            result["gt_taget_instance"] = f"poi_goal:{poi_name}"
            self._save_task_result(result, task_dir)
            return result
        short_memory.add_frame(start_images, start_pose)

        trajectory = [start_pose]
        initial_distance = self._distance_to_goal(start_pose, goal_xy)
        min_distance = initial_distance

        # Collision tracking
        collision_count = 0
        collision_step_indices: List[int] = []
        max_consecutive_collision = 0
        cur_consecutive_collision = 0
        collision_skipped_near_goal = 0
        collision_terminated = False

        # Resolve effective collision mode
        collision_mode = self.config.resolve_collision_mode()

        # Sanity check: are start/goal inside obstacles?
        start_in_obstacle = False
        goal_in_obstacle = False
        if collision_mode != "off":
            start_in_obstacle = self._check_poi_collision(start_pose, episode)
            goal_in_obstacle = self._check_poi_collision(goal_pose, episode)
            if start_in_obstacle or goal_in_obstacle:
                print(
                    f"[PoiGoal][collision sanity] task={task.task_id} "
                    f"POI={poi_name}: start_in_obstacle={start_in_obstacle}, "
                    f"goal_in_obstacle={goal_in_obstacle} "
                    f"(robot_radius={self.config.robot_radius}m). "
                    f"Hint: bench annotations assume start/goal are in free "
                    f"space. If this occurs frequently, try a smaller "
                    f"--robot-radius or check occ_map/meta_data."
                )

        prev_pose = start_pose
        cur_step = 0

        while cur_step < self.config.max_steps:
            # --- Build observation ---
            observation = self._build_poi_observation(
                episode=episode,
                short_memory=short_memory,
                goal_xy=goal_xy,
                step_count=cur_step,
                poi_name=poi_name,
            )

            # --- Agent prediction ---
            prediction = agent.predict(observation)

            # --- Convert waypoint to world poses ---
            pred_poses = self.get_pred_poses(
                prediction.waypoint, prediction.directions, short_memory
            )
            self._adjust_poses_z(pred_poses, episode)

            # --- Select the best pose (check for early arrival) ---
            select_pose_idx = min(2, len(pred_poses) - 1)
            arrive = prediction.arrive
            for idx in range(len(pred_poses)):
                d2g = self._distance_to_goal(pred_poses[idx], goal_xy)
                if d2g < self.config.arrive_threshold:
                    select_pose_idx = idx
                    arrive = True
                    break

            select_pose_idx = min(select_pose_idx, len(pred_poses) - 1)
            select_pose = pred_poses[select_pose_idx]

            # --- Render at new pose ---
            try:
                cur_images = self._render_images(
                    select_pose, scene_id, render_image_path,
                    short_memory.frame_count,
                )
            except RenderFailureError as exc:
                print(
                    f"[PoiGoal] render failed at step {cur_step} for "
                    f"{task.task_id}: {exc}"
                )
                travel_length = self.compute_path_length(trajectory)
                result = {
                    "episode_id": episode.episode_id,
                    "task_id": task.task_id,
                    "status": "render_error",
                    "target_label": poi_name,
                    "gt_taget_instance": f"poi_goal:{poi_name}",
                    "steps": cur_step,
                    "travel_length": travel_length,
                    "shortest_path_length": shortest_path_length,
                    "distance_to_goal": min_distance,
                    "success": False,
                    "oracle_success": False,
                    "spl": 0.0,
                    "metrics": {
                        "poi_name": poi_name,
                        "initial_distance_to_goal": initial_distance,
                        "min_distance_to_goal": min_distance,
                        "final_distance_to_goal": min_distance,
                        "pointgoal_arrive_threshold": self.config.arrive_threshold,
                        "collision_mode": collision_mode,
                        "collision_count": int(collision_count),
                        "collision_terminated": False,
                    },
                }
                self._save_task_result(result, task_dir)
                return result

            cur_dis = self._distance_to_goal(select_pose, goal_xy)
            min_distance = min(min_distance, cur_dis)

            short_memory.add_frame(cur_images, select_pose)
            trajectory.append(select_pose)
            cur_step += 1

            # --- Collision detection (off / soft / hard) ---
            # Terminal-zone exemption: when inside the arrive_threshold
            # circle around the goal, collisions are not counted because
            # POI targets are typically on building walls and the robot
            # must get close.
            if collision_mode != "off":
                hit_obstacle = self._check_poi_collision(select_pose, episode)
                inside_goal_zone = cur_dis <= self.config.arrive_threshold

                if hit_obstacle and inside_goal_zone:
                    collision_skipped_near_goal += 1
                    cur_consecutive_collision = 0
                elif hit_obstacle:
                    collision_count += 1
                    cur_consecutive_collision += 1
                    max_consecutive_collision = max(
                        max_consecutive_collision, cur_consecutive_collision
                    )
                    collision_step_indices.append(cur_step - 1)
                    if collision_mode == "hard":
                        collision_terminated = True
                        print(
                            f"[PoiGoal][HARD collision] task={task.task_id} "
                            f"POI={poi_name}: step={cur_step - 1} hit "
                            f"obstacle, terminate."
                        )
                else:
                    cur_consecutive_collision = 0

            prev_pose = select_pose

            # --- Check termination ---
            if arrive or cur_step >= self.config.max_steps or collision_terminated:
                success = cur_dis <= self.config.arrive_threshold
                # If terminated by collision, force failure even if inside
                # the arrival zone (the robot stopped on a wall).
                if collision_terminated:
                    success = False

                travel_length = self.compute_path_length(trajectory)
                spl = self.compute_poi_spl(
                    success, travel_length, shortest_path_length,
                    self.config.arrive_threshold,
                )

                if collision_terminated:
                    status = "collision"
                else:
                    status = "stop" if success else "max_steps"

                result = {
                    "episode_id": episode.episode_id,
                    "task_id": task.task_id,
                    "status": status,
                    "target_label": poi_name,
                    "gt_taget_instance": f"poi_goal:{poi_name}",
                    "steps": cur_step,
                    "travel_length": travel_length,
                    "shortest_path_length": shortest_path_length,
                    "distance_to_goal": cur_dis,
                    "success": success,
                    "oracle_success": success,
                    "spl": spl,
                    "metrics": {
                        "poi_name": poi_name,
                        "initial_distance_to_goal": initial_distance,
                        "min_distance_to_goal": min_distance,
                        "final_distance_to_goal": cur_dis,
                        "pointgoal_arrive_threshold": self.config.arrive_threshold,
                        # Collision metrics
                        "collision_mode": collision_mode,
                        "collision_check_enabled": bool(collision_mode != "off"),
                        "robot_radius": float(self.config.robot_radius),
                        "collision_count": int(collision_count),
                        "collision_rate": (
                            float(collision_count) / max(cur_step, 1)
                        ),
                        "max_consecutive_collision": int(
                            max_consecutive_collision
                        ),
                        "collision_step_indices": collision_step_indices,
                        "collision_skipped_near_goal": int(
                            collision_skipped_near_goal
                        ),
                        "has_collision": bool(collision_count > 0),
                        "collision_terminated": bool(collision_terminated),
                        # Sanity check flags
                        "start_in_obstacle": bool(start_in_obstacle),
                        "goal_in_obstacle": bool(goal_in_obstacle),
                        "start_or_goal_in_obstacle": bool(
                            start_in_obstacle or goal_in_obstacle
                        ),
                    },
                }
                self._save_task_result(result, task_dir)
                return result

        # Should not reach here, but just in case
        result = self._make_error_result(episode, task)
        result["target_label"] = poi_name
        result["gt_taget_instance"] = f"poi_goal:{poi_name}"
        self._save_task_result(result, task_dir)
        return result

    # ------------------------------------------------------------------
    # POI-specific observation building
    # ------------------------------------------------------------------

    def _build_poi_observation(
        self,
        episode: Episode,
        short_memory: Any,
        goal_xy: np.ndarray,
        step_count: int,
        poi_name: str,
    ) -> PoiGoalObservation:
        """Construct a :class:`PoiGoalObservation` from current state.

        Builds the base point-goal observation and then wraps it into a
        :class:`PoiGoalObservation` with the ``poi_name`` field set.
        """
        base_obs = self._build_observation(
            episode=episode,
            short_memory=short_memory,
            goal_xy=goal_xy,
            step_count=step_count,
        )

        poi_obs = PoiGoalObservation(
            # Required fields from base
            images=base_obs.images,
            target_position=base_obs.target_position,
            position=base_obs.position,
            rotation=base_obs.rotation,
            heading=base_obs.heading,
            step_count=base_obs.step_count,
            distance_to_goal=base_obs.distance_to_goal,
            # Optional fields
            history_images=base_obs.history_images,
            history_poses=base_obs.history_poses,
            occ_map=base_obs.occ_map,
            height_map=base_obs.height_map,
            meta_data=base_obs.meta_data,
            extra=base_obs.extra,
            # POI-specific field
            poi_name=poi_name,
        )
        return poi_obs

    # ------------------------------------------------------------------
    # Collision detection
    # ------------------------------------------------------------------

    def _check_poi_collision(
        self, pose: np.ndarray, episode: Episode
    ) -> bool:
        """Check whether the given pose collides with obstacles.

        Uses the occupancy map with circular robot footprint inflation.
        Returns False if collision checking is disabled or if the
        occupancy map / metadata is unavailable.

        Parameters
        ----------
        pose : ndarray, shape (4, 4)
            World-frame pose matrix.
        episode : Episode
            Current episode (must have ``occ_map`` and ``meta_data``).

        Returns
        -------
        bool
            True if the pose is in an obstacle region.
        """
        if self.config.resolve_collision_mode() == "off":
            return False

        occ_map = getattr(episode, "occ_map", None)
        meta_data = getattr(episode, "meta_data", None)
        if occ_map is None or meta_data is None:
            return False

        # World coordinates
        x, y = float(pose[0, 3]), float(pose[1, 3])

        # Convert to pixel coordinates
        required_fields = [
            "TOP_LEFT_X", "TOP_LEFT_Y", "IMAGE_WIDTH", "IMAGE_HEIGHT",
            "COORDINATE_RANGE_X", "COORDINATE_RANGE_Y",
        ]
        for f in required_fields:
            if f not in meta_data:
                return False

        top_left_x = meta_data["TOP_LEFT_X"]
        top_left_y = meta_data["TOP_LEFT_Y"]
        range_x = meta_data["COORDINATE_RANGE_X"]
        range_y = meta_data["COORDINATE_RANGE_Y"]
        img_w = int(meta_data["IMAGE_WIDTH"])
        img_h = int(meta_data["IMAGE_HEIGHT"])

        px = int((top_left_x - x) * img_w / range_x)
        py = int((y - top_left_y) * img_h / range_y)

        # Compute inflation radius in pixels
        robot_radius = float(self.config.robot_radius)
        if robot_radius > 0:
            meters_per_pixel = range_x / img_w
            radius_px = int(math.ceil(robot_radius / meters_per_pixel))
        else:
            radius_px = 0

        # Determine obstacle threshold
        polarity = getattr(self.config, "occ_obstacle_polarity", "dark")
        dark_thr = getattr(self.config, "occ_dark_threshold", 64)

        # Handle multi-channel occ_map (convert to greyscale)
        if occ_map.ndim == 3:
            grey = np.mean(occ_map, axis=2)
        else:
            grey = occ_map

        def _is_obstacle(pixel_x: int, pixel_y: int) -> bool:
            if pixel_x < 0 or pixel_x >= img_w:
                return False
            if pixel_y < 0 or pixel_y >= img_h:
                return False
            val = grey[pixel_y, pixel_x]
            if polarity == "dark":
                return val <= dark_thr
            else:
                return val >= dark_thr

        # Check the circular footprint
        for dx in range(-radius_px, radius_px + 1):
            for dy in range(-radius_px, radius_px + 1):
                if dx * dx + dy * dy > radius_px * radius_px:
                    continue
                if _is_obstacle(px + dx, py + dy):
                    return True

        return False

    # ------------------------------------------------------------------
    # SPL computation (POI-specific adjustment)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_poi_spl(
        success: bool,
        actual_path_length: float,
        shortest_path_length: float,
        arrive_threshold: float = 2.0,
    ) -> float:
        """Compute SPL with arrive_threshold subtracted from shortest path.

        ``effective_shortest = max(shortest - arrive_threshold, 1e-3)``
        ``SPL = success * (effective_shortest / max(actual, effective_shortest))``

        The threshold subtraction prevents SPL from degenerating to {0, 1}
        when the arrive_threshold is large relative to the path length.

        Parameters
        ----------
        success : bool
            Whether the task succeeded.
        actual_path_length : float
            The agent's actual travel distance.
        shortest_path_length : float
            The ground-truth shortest path length.
        arrive_threshold : float
            The arrival distance threshold.

        Returns
        -------
        float
            SPL value in [0, 1].
        """
        if not success:
            return 0.0
        if shortest_path_length <= 0:
            return 0.0
        effective_shortest = max(
            shortest_path_length - max(arrive_threshold, 0.0), 1e-3
        )
        return effective_shortest / max(actual_path_length, effective_shortest)

    # ------------------------------------------------------------------
    # Summary (overridden for POI-specific stats)
    # ------------------------------------------------------------------

    def evaluate(
        self,
        agent: BasePoiGoalAgent,
        result_path: Optional[str] = None,
        resume_dir: Optional[str] = None,
    ) -> List[dict]:
        """Run evaluation across all episodes and tasks.

        Identical to the parent class but uses :meth:`_build_poi_summary`
        for the final summary.
        """
        from abotn_evaluator.memory import ShortMemory

        # --- Build set of completed (episode_id, task_id) pairs ---
        done_keys: set = set()
        if resume_dir:
            import glob
            pattern = os.path.join(resume_dir, "*", "*", "result.json")
            for fpath in glob.glob(pattern):
                rel = os.path.relpath(fpath, resume_dir)
                parts = rel.split(os.sep)
                if len(parts) >= 3:
                    done_keys.add((parts[0], parts[1]))
            if done_keys:
                print(f"[resume] Found {len(done_keys)} completed tasks in {resume_dir}")

        all_results: List[dict] = []

        try:
            import torch
            _no_grad_ctx = torch.no_grad()
        except ImportError:
            import contextlib
            _no_grad_ctx = contextlib.nullcontext()

        with _no_grad_ctx:
            for episode in tqdm(self.scene.episodes, desc="Episodes"):
                episode_dir = os.path.join(self.output_dir, episode.episode_id)
                os.makedirs(episode_dir, exist_ok=True)

                for task in tqdm(
                    episode.tasks,
                    desc=f"Episode {episode.episode_id}",
                    leave=False,
                ):
                    if (episode.episode_id, task.task_id) in done_keys:
                        continue

                    short_memory = ShortMemory(
                        max_history_frames=self.config.max_history_frames,
                        resize_ratio=self.config.history_resize_ratio,
                    )

                    task_result = self._evaluate_task(
                        agent=agent,
                        episode=episode,
                        task=task,
                        short_memory=short_memory,
                        episode_dir=episode_dir,
                    )
                    all_results.append(task_result)

        # Save POI-specific summary
        summary = self._build_poi_summary(all_results)
        summary_path = result_path or os.path.join(
            self.output_dir, "eval_summary.json"
        )
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(
                _make_serialisable(summary), f, ensure_ascii=False, indent=2
            )
        self._print_poi_summary(summary)

        return all_results

    @staticmethod
    def _build_poi_summary(all_results: List[dict]) -> dict:
        """Aggregate per-task results into a POI-goal summary dict.

        Includes per-POI breakdown and collision statistics in addition
        to the standard point-goal summary fields.
        """
        total = len(all_results)
        if total == 0:
            return {"task_type": "poi_goal", "total_tasks": 0}

        success_count = sum(
            1 for r in all_results if r.get("success", False)
        )
        oracle_count = sum(
            1 for r in all_results if r.get("oracle_success", False)
        )
        avg_spl = float(
            np.mean([r.get("spl", 0.0) for r in all_results])
        )

        # Collision summary
        metrics_list = [r.get("metrics", {}) for r in all_results]
        collision_enabled_results = [
            m for m in metrics_list
            if m.get("collision_check_enabled", False)
        ]
        if collision_enabled_results:
            tasks_with_collision = sum(
                1 for m in collision_enabled_results
                if m.get("has_collision", False)
            )
            total_collision_steps = sum(
                int(m.get("collision_count", 0))
                for m in collision_enabled_results
            )
            collision_terminated_count = sum(
                1 for m in collision_enabled_results
                if m.get("collision_terminated", False)
            )
            n_coll = len(collision_enabled_results)
            collision_summary = {
                "collision_check_enabled": True,
                "tasks_evaluated_for_collision": n_coll,
                "tasks_with_any_collision": tasks_with_collision,
                "collision_task_rate": (
                    tasks_with_collision / n_coll if n_coll > 0 else 0.0
                ),
                "total_collision_steps": total_collision_steps,
                "avg_collision_steps_per_task": (
                    total_collision_steps / n_coll if n_coll > 0 else 0.0
                ),
                "avg_collision_step_rate": float(np.mean([
                    m.get("collision_rate", 0.0)
                    for m in collision_enabled_results
                ])),
                "collision_terminated_count": collision_terminated_count,
            }
        else:
            collision_summary = {"collision_check_enabled": False}

        # Status distribution
        status_dist: Dict[str, int] = {}
        for r in all_results:
            s = r.get("status", "unknown")
            status_dist[s] = status_dist.get(s, 0) + 1

        # Per-POI statistics
        poi_stats: Dict[str, Dict[str, Any]] = {}
        for r in all_results:
            m = r.get("metrics", {})
            poi = m.get("poi_name", r.get("target_label", "unknown"))
            if poi not in poi_stats:
                poi_stats[poi] = {
                    "total": 0,
                    "success": 0,
                    "spl_values": [],
                }
            poi_stats[poi]["total"] += 1
            if r.get("success", False):
                poi_stats[poi]["success"] += 1
            poi_stats[poi]["spl_values"].append(r.get("spl", 0.0))

        poi_stats_clean = {}
        for poi, stat in poi_stats.items():
            poi_stats_clean[poi] = {
                "total": stat["total"],
                "success": stat["success"],
                "success_rate": (
                    stat["success"] / stat["total"]
                    if stat["total"] > 0 else 0.0
                ),
                "avg_spl": float(np.mean(stat["spl_values"])),
            }

        summary = {
            "task_type": "poi_goal",
            "total_tasks": total,
            "success_count": success_count,
            "oracle_success_count": oracle_count,
            "success_rate": success_count / total if total > 0 else 0.0,
            "oracle_success_rate": oracle_count / total if total > 0 else 0.0,
            "spl": avg_spl,
            "status_distribution": status_dist,
            "avg_steps": float(
                np.mean([r.get("steps", 0) for r in all_results])
            ),
            "avg_travel_length": float(
                np.mean([r.get("travel_length", 0.0) for r in all_results])
            ),
            "avg_shortest_path_length": float(
                np.mean([
                    r.get("shortest_path_length", 0.0) for r in all_results
                ])
            ),
            "avg_initial_distance_to_goal": float(
                np.mean([
                    m.get("initial_distance_to_goal", 0.0)
                    for m in metrics_list
                ])
            ),
            "avg_min_distance_to_goal": float(
                np.mean([
                    m.get("min_distance_to_goal", 0.0)
                    for m in metrics_list
                    if np.isfinite(m.get("min_distance_to_goal", 0.0))
                ]) if any(
                    np.isfinite(m.get("min_distance_to_goal", 0.0))
                    for m in metrics_list
                ) else 0.0
            ),
            "avg_final_distance_to_goal": float(
                np.mean([
                    r.get("distance_to_goal", 0.0)
                    for r in all_results
                    if np.isfinite(r.get("distance_to_goal", 0.0))
                ]) if any(
                    np.isfinite(r.get("distance_to_goal", 0.0))
                    for r in all_results
                ) else 0.0
            ),
            "collision": collision_summary,
            "poi_stats": poi_stats_clean,
        }

        return summary

    @staticmethod
    def _print_poi_summary(summary: dict) -> None:
        """Print POI-goal evaluation summary to stdout."""
        total = summary.get("total_tasks", 0)
        if total == 0:
            print("No tasks evaluated.")
            return

        print(f"\n{'=' * 60}")
        print("POI Goal Evaluation Summary:")
        print(f"{'=' * 60}")
        print(f"Total tasks: {total}")
        sr = summary.get("success_rate", 0.0)
        osr = summary.get("oracle_success_rate", 0.0)
        print(f"Success: {summary.get('success_count', 0)} ({sr * 100:.2f}%)")
        print(
            f"Oracle Success: "
            f"{summary.get('oracle_success_count', 0)} ({osr * 100:.2f}%)"
        )
        print(f"SPL: {summary.get('spl', 0.0):.4f}")
        print(f"Avg Steps: {summary.get('avg_steps', 0.0):.2f}")
        print(
            f"Avg Travel Length: "
            f"{summary.get('avg_travel_length', 0.0):.2f}"
        )
        print(
            f"Avg Initial Distance: "
            f"{summary.get('avg_initial_distance_to_goal', 0.0):.2f}"
        )
        print(
            f"Avg Final Distance: "
            f"{summary.get('avg_final_distance_to_goal', 0.0):.2f}"
        )

        # Collision info
        coll = summary.get("collision", {})
        if coll.get("collision_check_enabled"):
            print(
                f"Collision: "
                f"{coll['tasks_with_any_collision']}/"
                f"{coll['tasks_evaluated_for_collision']} tasks "
                f"({coll['collision_task_rate'] * 100:.1f}%) hit obstacles, "
                f"avg {coll['avg_collision_steps_per_task']:.2f} steps/task "
                f"({coll['avg_collision_step_rate'] * 100:.2f}% per-step rate)"
            )
            if coll.get("collision_terminated_count", 0) > 0:
                print(
                    f"Collision terminated: "
                    f"{coll['collision_terminated_count']} tasks"
                )

        # Per-POI stats
        poi_stats = summary.get("poi_stats", {})
        if poi_stats:
            print(f"\nPer-POI Stats:")
            for poi, stat in poi_stats.items():
                print(
                    f"  {poi}: {stat['success']}/{stat['total']} "
                    f"({stat['success_rate'] * 100:.1f}%), "
                    f"avg_spl={stat['avg_spl']:.4f}"
                )

        print(f"{'=' * 60}")
