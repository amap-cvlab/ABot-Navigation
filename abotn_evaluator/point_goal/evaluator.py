"""Core evaluator for point-goal navigation.

Migrated from ``GaussianEvaluatorPointGoal`` in the original codebase.
Key design decisions:

* Constructor takes scene, renderer, config, and output_dir -- but NOT the
  agent.  The agent is passed to :meth:`evaluate`.
* :class:`ShortMemory` is managed internally.
* The evaluator treats the agent as a complete black box: it builds an
  observation, calls ``agent.predict(obs)``, and processes the result.
  Any internal architecture (dual-system, ensemble, etc.) is the agent's
  own responsibility.
* Collision detection logic is preserved exactly from the original.
"""

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from abotn_evaluator.interface.point_goal import BasePointGoalAgent, Observation, WaypointPrediction
from abotn_evaluator.scene import Episode, GaussianScene, Task
from abotn_evaluator.render_client import RenderFailureError


# ============================================================================
# Height-map Z lookup
# ============================================================================

def get_z_value(
    x: float,
    y: float,
    z_data_info: tuple,
    meta_data: dict,
    default_z: float = -0.1,
) -> float:
    """Look up the ground height at world coordinates ``(x, y)`` using
    the height map and metadata.

    Parameters
    ----------
    x, y : float
        World coordinates.
    z_data_info : tuple
        ``(z_data, width, height)`` where *z_data* is the height map array.
    meta_data : dict
        Coordinate-system metadata.
    default_z : float
        Fallback value if the height map is unavailable.

    Returns
    -------
    float
        Ground height in metres.
    """
    try:
        z_data, width, height = z_data_info
        if z_data is None:
            return default_z

        required_fields = [
            "TOP_LEFT_X", "TOP_LEFT_Y", "IMAGE_WIDTH", "IMAGE_HEIGHT",
            "COORDINATE_RANGE_X", "COORDINATE_RANGE_Y",
        ]
        for f in required_fields:
            if f not in meta_data:
                return default_z

        top_left_x = meta_data["TOP_LEFT_X"]
        top_left_y = meta_data["TOP_LEFT_Y"]
        range_x = meta_data["COORDINATE_RANGE_X"]
        range_y = meta_data["COORDINATE_RANGE_Y"]

        x_pixel = int((top_left_x - x) * width / range_x)
        y_pixel = int((y - top_left_y) * height / range_y)

        x_pixel = max(0, min(x_pixel, width - 1))
        y_pixel = max(0, min(y_pixel, height - 1))

        return float(z_data[y_pixel][x_pixel])
    except Exception:
        return default_z


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class EvalConfig:
    """Evaluation configuration for point-goal navigation.

    Attributes
    ----------
    render_url : str
        URL of the Gaussian splatting render service.
    max_steps : int
        Maximum number of agent steps per task before forced stop.
    arrive_threshold : float
        Distance (metres) at which the agent is considered to have reached
        the goal.
    camera_config : object or None
        Camera intrinsics/extrinsics.  If ``None``, the renderer's default
        is used.
    collision_threshold : int
        Path-collision count threshold for SR_NEW-style metrics computed
        during evaluation.
    provide_history : bool
        Whether to include ``history_images`` and ``history_poses`` in
        observations.
    provide_occ_map : bool
        Whether to include the occupancy map in observations.
    provide_height_map : bool
        Whether to include the height map in observations.
    max_history_frames : int
        Maximum number of history frames kept in ShortMemory.
    history_resize_ratio : float
        Downscale ratio applied to history frame images.
    save_render_images : bool
        Whether to save rendered images to disk.
    occ_dilation_meters : float
        Amount (metres) by which to dilate the free space in the occ_map.
    enable_visualization : bool
        If True, process ``prediction.extra`` for visualization outputs
        (e.g. affordance pixel overlay on rendered images).
    """

    render_url: str = "http://127.0.0.1:7001/render_gs"
    max_steps: int = 100
    arrive_threshold: float = 0.5
    camera_config: Optional[Any] = None
    collision_threshold: int = 3
    provide_history: bool = False
    provide_occ_map: bool = False
    provide_height_map: bool = False
    max_history_frames: int = 20
    history_resize_ratio: float = 0.25
    save_render_images: bool = True
    occ_dilation_meters: float = 0.0
    enable_visualization: bool = False


# Benchmark protocol parameters per evaluation mode. These define what counts
# as "success" for each track; changing them makes results non-comparable
# across agents. Shared by ``make_eval_config()`` (API) and the CLI runner.
POINT_GOAL_PROTOCOL = {
    "outdoor": {"arrive_threshold": 0.5, "collision_threshold": 3, "occ_dilation_meters": 0.5},
    "indoor": {"arrive_threshold": 0.5, "collision_threshold": 1, "occ_dilation_meters": 0.2},
}


def make_eval_config(mode: str = "outdoor", render_url: Optional[str] = None, **kwargs) -> "EvalConfig":
    """Build an :class:`EvalConfig` with protocol params auto-selected by *mode*.

    The benchmark protocol parameters (``arrive_threshold``,
    ``collision_threshold``, ``occ_dilation_meters``) are filled from the
    standard values for *mode* (``"outdoor"`` or ``"indoor"``). Pass any of
    them explicitly in *kwargs* to override (for ablation studies; results are
    then no longer comparable to the official protocol).

    All other EvalConfig fields (``max_steps``, ``save_render_images``,
    ``provide_history``, ``camera_config``, ...) are passed through from
    *kwargs*.

    Parameters
    ----------
    mode : str
        Evaluation track: ``"outdoor"`` or ``"indoor"``.
    render_url : str
        URL of the Gaussian splatting render service (required).
    **kwargs
        Any EvalConfig field; protocol params here override the mode default.

    Returns
    -------
    EvalConfig
    """
    if mode not in POINT_GOAL_PROTOCOL:
        raise ValueError(f"unknown mode {mode!r}, expected 'outdoor' or 'indoor'")
    if not render_url:
        raise ValueError("render_url is required")
    cfg = dict(POINT_GOAL_PROTOCOL[mode])
    cfg.update(kwargs)
    cfg["render_url"] = render_url
    return EvalConfig(**cfg)


# ============================================================================
# Evaluator
# ============================================================================

class PointGoalEvaluator:
    """Point-goal navigation evaluator for Gaussian splatting scenes.

    Parameters
    ----------
    scene : GaussianScene
        The scene manager with loaded episodes.
    renderer : object
        A Gaussian splatting renderer instance (must have
        ``render_at_pose(pose, scene_id, ...)`` method).
    config : EvalConfig
        Evaluation configuration.
    output_dir : str
        Directory for saving results and optional images.
    """

    def __init__(
        self,
        scene: GaussianScene,
        renderer: Any,
        config: Optional[EvalConfig] = None,
        output_dir: str = "./eval_output",
    ) -> None:
        self.scene = scene
        self.renderer = renderer
        self.config = config or EvalConfig()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Apply occ_map dilation to all episodes
        if self.config.occ_dilation_meters > 0:
            for ep in self.scene.episodes:
                ep.dilate_free_space_by_meter(self.config.occ_dilation_meters)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        agent: BasePointGoalAgent,
        result_path: Optional[str] = None,
        resume_dir: Optional[str] = None,
    ) -> List[dict]:
        """Run evaluation across all episodes and tasks.

        Parameters
        ----------
        agent : BasePointGoalAgent
            The agent to evaluate.
        result_path : str, optional
            Path for the final summary JSON.  Defaults to
            ``{output_dir}/eval_summary.json``.
        resume_dir : str, optional
            Directory of a previous run to resume from.  Scans for
            completed ``result.json`` files and skips those (episode, task)
            pairs.

        Returns
        -------
        list of dict
            Per-task result dictionaries.
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

        # Wrap the entire evaluation in torch.no_grad() to prevent gradient
        # accumulation from the agent's model calls, which would otherwise
        # cause VRAM to grow continuously across episodes.
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
                    # Skip completed tasks when resuming
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

        # Aggregated summary (overall + per-difficulty) is produced by
        # ``metrics.analyze_and_report`` after evaluation, which writes the
        # unified ``eval_summary.json``.
        return all_results

    # ------------------------------------------------------------------
    # Per-task evaluation
    # ------------------------------------------------------------------

    def _evaluate_task(
        self,
        agent: BasePointGoalAgent,
        episode: Episode,
        task: Task,
        short_memory: Any,
        episode_dir: str,
    ) -> dict:
        """Evaluate a single task and return the result dict."""
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
            print(f"[PointGoal] build path failed for {task.task_id}: {exc}")
            result = self._make_error_result(episode, task)
            self._save_task_result(result, task_dir)
            return result

        start_pose = self._pose_dict_to_matrix(point_path[0])
        goal_pose = self._pose_dict_to_matrix(point_path[-1])
        self._adjust_pose_z(start_pose, episode)
        self._adjust_pose_z(goal_pose, episode)
        goal_xy = np.array(
            [point_path[-1]["x"], point_path[-1]["y"]], dtype=np.float32
        )

        # Compute GT path length
        shortest_path_length = self.compute_path_length(point_path)

        # Render initial images
        try:
            start_images = self._render_images(
                start_pose, scene_id, render_image_path, short_memory.frame_count
            )
        except RenderFailureError as exc:
            print(f"[PointGoal] initial render failed for {task.task_id}: {exc}")
            result = self._make_error_result(episode, task)
            self._save_task_result(result, task_dir)
            return result
        short_memory.add_frame(start_images, start_pose)

        trajectory = [start_pose]
        initial_distance = self._distance_to_goal(start_pose, goal_xy)
        min_distance = initial_distance

        # Collision tracking
        path_collision_count = 0
        collision_path_length = 0.0
        total_distance = 0.0
        is_path_collided = False
        collision_steps = []

        prev_pose = start_pose
        cur_step = 0

        while cur_step < self.config.max_steps:
            # --- Build observation ---
            observation = self._build_observation(
                episode=episode,
                short_memory=short_memory,
                goal_xy=goal_xy,
                step_count=cur_step,
            )

            # --- Agent prediction ---
            prediction = agent.predict(observation)

            # --- Visualization: process agent's extra field ---
            if self.config.enable_visualization and prediction.extra:
                from abotn_evaluator.visualization import process_extra_visualization
                process_extra_visualization(
                    extra=prediction.extra,
                    images=short_memory.get_current_images(),
                    save_dir=render_image_path,
                    step_id=short_memory.frame_count - 1,
                )

            # --- Convert waypoint to world poses ---
            pred_poses = self.get_pred_poses(
                prediction.waypoint, prediction.directions, short_memory
            )
            self._adjust_poses_z(pred_poses, episode)

            # --- Select the best pose ---
            select_pose_idx = 1
            arrive = prediction.arrive
            for idx in range(len(pred_poses)):
                d2g = self._distance_to_goal(pred_poses[idx], goal_xy)
                if d2g < self.config.arrive_threshold:
                    select_pose_idx = idx
                    arrive = True
                    break

            select_pose_idx = min(select_pose_idx, len(pred_poses) - 1)
            select_pose = pred_poses[select_pose_idx]

            # --- Collision detection ---
            step_dist = GaussianScene.compute_pose_distance(prev_pose, select_pose)
            total_distance += step_dist

            if GaussianScene.check_line_collision_fast(
                prev_pose, select_pose, episode, step_size_pixel=2.0
            ):
                path_collision_count += 1
                is_path_collided = True
                collision_steps.append(cur_step)
                coll_len = GaussianScene.compute_line_collision_length(
                    prev_pose, select_pose, episode, sample_step_meter=0.05
                )
                collision_path_length += coll_len

            prev_pose = select_pose

            # --- Render at new pose (no afford_pixels — those are drawn
            # on the current frame above, not on the next frame). ---
            try:
                cur_images = self._render_images(
                    select_pose, scene_id, render_image_path,
                    short_memory.frame_count,
                )
            except RenderFailureError as exc:
                print(
                    f"[PointGoal] render failed at step {cur_step} for "
                    f"{task.task_id}: {exc}"
                )
                # Treat as forced termination; success is False because we did
                # not reach the goal under normal conditions.
                success = False
                travel_length = self.compute_path_length(trajectory)
                result = {
                    "episode_id": episode.episode_id,
                    "task_id": task.task_id,
                    "status": "render_error",
                    "target_label": "point_goal",
                    "gt_taget_instance": "point_goal_target",
                    "steps": cur_step,
                    "travel_length": travel_length,
                    "shortest_path_length": shortest_path_length,
                    "distance_to_goal": min_distance,
                    "success": False,
                    "oracle_success": False,
                    "metrics": {},
                }
                self._save_task_result(result, task_dir)
                return result

            cur_dis = self._distance_to_goal(select_pose, goal_xy)
            min_distance = min(min_distance, cur_dis)

            short_memory.add_frame(cur_images, select_pose)
            trajectory.append(select_pose)
            cur_step += 1

            # --- Check termination ---
            if arrive or cur_step >= self.config.max_steps:
                success = cur_dis <= self.config.arrive_threshold
                travel_length = self.compute_path_length(trajectory)

                # Compute collision-aware metrics
                success_new = (
                    success
                    and path_collision_count < self.config.collision_threshold
                )
                spl_new = self.compute_spl(
                    success_new, travel_length, shortest_path_length
                )
                if success_new and cur_step > 0:
                    tcr_new = (cur_step - path_collision_count) / cur_step
                else:
                    tcr_new = 0.0
                if success_new and total_distance > 0:
                    dcr_new = (total_distance - collision_path_length) / total_distance
                else:
                    dcr_new = 0.0

                collision_path_ratio = (
                    collision_path_length / total_distance
                    if total_distance > 0 else 0.0
                )

                result = {
                    "episode_id": episode.episode_id,
                    "task_id": task.task_id,
                    "status": "stop" if success else "max_steps",
                    "target_label": "point_goal",
                    "gt_taget_instance": "point_goal_target",
                    "steps": cur_step,
                    "travel_length": travel_length,
                    "shortest_path_length": shortest_path_length,
                    "distance_to_goal": cur_dis,
                    "success": success,
                    "oracle_success": success,
                    "metrics": {
                        "initial_distance_to_goal": initial_distance,
                        "min_distance_to_goal": min_distance,
                        "final_distance_to_goal": cur_dis,
                        "pointgoal_arrive_threshold": self.config.arrive_threshold,
                        "success_new": success_new,
                        "spl_new": spl_new,
                        "tcr_new": tcr_new,
                        "dcr_new": dcr_new,
                        "path_collision_count": path_collision_count,
                        "collision_path_length": collision_path_length,
                        "total_distance": total_distance,
                        "is_path_collided": is_path_collided,
                        "collision_path_ratio": collision_path_ratio,
                        "collision_steps": collision_steps,
                    },
                }
                self._save_task_result(result, task_dir)
                return result

        # Should not reach here, but just in case
        result = self._make_error_result(episode, task)
        self._save_task_result(result, task_dir)
        return result

    # ------------------------------------------------------------------
    # Observation building
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        episode: Episode,
        short_memory: Any,
        goal_xy: np.ndarray,
        step_count: int,
    ) -> Observation:
        """Construct an :class:`Observation` from current evaluator state."""
        last_pose = short_memory.get_last_pose()
        if last_pose is None:
            last_pose = np.eye(4)

        # Current multi-view images
        current_imgs = short_memory.get_current_images()
        images: Dict[str, Any] = {}
        view_names = ["left", "front", "right"]
        for i, name in enumerate(view_names):
            if i < len(current_imgs):
                images[name] = current_imgs[i]  # keep PIL as-is (JpegImageFile or Image)
            else:
                images[name] = np.zeros((640, 720, 3), dtype=np.uint8)

        # Target position in local coordinates [front, left]
        cur_pos_xy = last_pose[:2, 3]
        delta_world = goal_xy - cur_pos_xy
        rot_inv = last_pose[:2, :2].T
        delta_local = rot_inv @ delta_world
        target_position = np.array([delta_local[0], delta_local[1]], dtype=np.float32)

        # Heading (yaw)
        heading = float(np.arctan2(last_pose[1, 0], last_pose[0, 0]))

        # Distance to goal
        distance_to_goal = float(np.linalg.norm(delta_world))

        # Position
        position = last_pose[:3, 3].copy()

        obs = Observation(
            images=images,
            target_position=target_position,
            position=position,
            rotation=last_pose.copy(),
            heading=heading,
            step_count=step_count,
            distance_to_goal=distance_to_goal,
            goal_world = goal_xy
        )

        # Optional fields
        if self.config.provide_history:
            hist_imgs = short_memory.get_history_images()
            obs.history_images = []
            for img in hist_imgs:
                obs.history_images.append({"front": np.array(img)})
            obs.history_poses = short_memory.get_history_poses()

        if self.config.provide_occ_map and episode.occ_map is not None:
            obs.occ_map = episode.occ_map.copy()

        if self.config.provide_height_map and episode.height_map is not None:
            obs.height_map = episode.height_map.copy()

        if episode.meta_data is not None:
            obs.meta_data = dict(episode.meta_data)

        return obs

    # ------------------------------------------------------------------
    # Path / pose helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_point(p: Any) -> Optional[Dict[str, float]]:
        """Normalise a trajectory point to a standard dict format."""
        if isinstance(p, dict):
            if "x" in p and "y" in p:
                return {
                    "x": float(p["x"]),
                    "y": float(p["y"]),
                    "z": float(p.get("z", 0.0)),
                    "roll": float(p.get("roll", 0.0)),
                    "pitch": float(p.get("pitch", 0.0)),
                    "yaw": float(p.get("yaw", 0.0)),
                }
        elif isinstance(p, (list, tuple)) and len(p) >= 2:
            return {
                "x": float(p[0]),
                "y": float(p[1]),
                "z": float(p[2]) if len(p) > 2 else 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
            }
        return None

    @classmethod
    def _normalize_path(cls, path_like: Any) -> List[Dict[str, float]]:
        """Normalise a sequence of trajectory points."""
        if path_like is None:
            return []
        normalized = []
        for p in path_like:
            np_p = cls._normalize_point(p)
            if np_p is not None:
                normalized.append(np_p)
        return normalized

    @classmethod
    def _build_pointgoal_path(cls, task: Task) -> List[Dict[str, float]]:
        """Build the reference path for a point-goal task.

        Uses the task's ``start_pose`` / ``end_pose`` as endpoints and
        ``trajectory`` as intermediate subgoals.
        """
        if task.start_pose is None or task.end_pose is None:
            raise ValueError(f"Task {task.task_id} missing start_pose or end_pose")

        start = cls._normalize_point(task.start_pose)
        end = cls._normalize_point(task.end_pose)
        if start is None or end is None:
            raise ValueError(f"Task {task.task_id} invalid start_pose or end_pose")

        ref_traj = cls._normalize_path(task.trajectory)
        if len(ref_traj) == 0:
            return [start, end]

        ref_traj[0] = start
        ref_traj[-1] = end
        if len(ref_traj) == 1:
            return [start, end]
        return ref_traj

    def _pose_dict_to_matrix(self, pose_dict: Dict[str, float]) -> np.ndarray:
        """Convert a pose dict to a 4x4 matrix."""
        return GaussianScene.get_gaussian_pose(pose_dict)

    @staticmethod
    def _distance_to_goal(pose: np.ndarray, goal_xy: np.ndarray) -> float:
        """Compute 2D distance from pose to goal."""
        return float(np.linalg.norm(pose[:2, 3] - goal_xy))

    # ------------------------------------------------------------------
    # Z adjustment
    # ------------------------------------------------------------------

    def _adjust_pose_z(self, pose: np.ndarray, episode: Episode) -> None:
        """In-place: set ``pose[2, 3] = ground_z + camera_height``."""
        x, y = pose[0, 3], pose[1, 3]
        camera_height = (
            self.config.camera_config.extrinsic_height
            if self.config.camera_config is not None
            else 0.65
        )
        if episode.height_map is not None and episode.meta_data is not None:
            z_data_info = (
                episode.height_map,
                episode.height_map.shape[1],
                episode.height_map.shape[0],
            )
            ground_z = get_z_value(x, y, z_data_info, episode.meta_data, default_z=-0.1)
            if ground_z > 5:
                ground_z = 0.0
        else:
            ground_z = -0.1
        pose[2, 3] = ground_z + camera_height

    def _adjust_poses_z(self, poses: List[np.ndarray], episode: Episode) -> None:
        """In-place Z adjustment for a list of poses."""
        for pose in poses:
            self._adjust_pose_z(pose, episode)

    # ------------------------------------------------------------------
    # Waypoint -> world coordinate transform
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_camera_poses(
        trajectory: np.ndarray,
        directions: np.ndarray,
        start_height: float,
    ) -> List[np.ndarray]:
        """Convert trajectory points and direction vectors into 4x4 pose
        matrices.

        Parameters
        ----------
        trajectory : ndarray, shape (N, 2)
            XY world positions.
        directions : ndarray, shape (N, 2)
            Direction vectors at each point.
        start_height : float
            Z coordinate for all poses.

        Returns
        -------
        list of ndarray
            List of 4x4 pose matrices.
        """
        poses = []
        for i in range(len(trajectory)):
            pt = trajectory[i]
            t = np.array([pt[0], pt[1], start_height])
            yaw = np.arctan2(directions[i][1], directions[i][0])
            pose = np.eye(4)
            pose[:3, :3] = R.from_euler("xyz", [0, 0, yaw], degrees=False).as_matrix()
            pose[:3, 3] = t
            poses.append(pose)
        return poses

    @classmethod
    def get_pred_poses(
        cls,
        wp_pred: np.ndarray,
        pred_directions: Optional[np.ndarray],
        short_memory: Any,
    ) -> List[np.ndarray]:
        """Transform predicted waypoints from local to world coordinates.

        This mirrors the original ``get_pred_poses`` logic:
        1. Append a zero Z column.
        2. Negate the first column (front -> -x).
        3. Swap columns 0 and 1.
        4. Transform back to world frame using the current pose.
        5. Build 4x4 poses from the resulting world positions and
           transformed direction vectors.

        Parameters
        ----------
        wp_pred : ndarray, shape (N, 2)
            Predicted waypoints in ``[front, left]`` local coords.
        pred_directions : ndarray, shape (N, 2) or None
            Direction vectors for each waypoint.  If None, uses the
            difference between consecutive waypoints.
        short_memory : ShortMemory
            The memory buffer (to get the current pose).

        Returns
        -------
        list of ndarray
            List of 4x4 world-frame pose matrices.
        """
        def transform_back(positions, current_pose_matrix):
            return (
                (current_pose_matrix[:3, :3] @ positions.T).T
                + current_pose_matrix[:3, 3]
            )

        cur_pose = short_memory.get_last_pose()
        if cur_pose is None:
            cur_pose = np.eye(4)

        wp = np.atleast_2d(wp_pred).copy()
        wp = np.concatenate([wp, np.zeros((wp.shape[0], 1))], axis=1)
        wp[:, 0] *= -1
        wp[:, [0, 1]] = wp[:, [1, 0]]
        wp = transform_back(wp, cur_pose)

        if pred_directions is not None:
            dirs = np.atleast_2d(pred_directions).copy()
            dirs = np.concatenate(
                [dirs, np.zeros((dirs.shape[0], 1))], axis=1
            )
            dirs = (cur_pose[:3, :3] @ dirs.T).T
        else:
            # Fallback: compute directions from consecutive waypoints
            dirs = np.zeros((wp.shape[0], 3))
            for i in range(wp.shape[0]):
                if i < wp.shape[0] - 1:
                    d = wp[i + 1] - wp[i]
                else:
                    d = wp[i] - (wp[i - 1] if i > 0 else cur_pose[:3, 3])
                norm = np.linalg.norm(d[:2])
                if norm > 1e-6:
                    dirs[i] = d / norm
                else:
                    dirs[i] = cur_pose[:3, 0]  # Use current heading

        start_height = cur_pose[2, 3]
        return cls._compute_camera_poses(wp, dirs, start_height)

    # ------------------------------------------------------------------
    # Path length / SPL
    # ------------------------------------------------------------------

    @staticmethod
    def compute_path_length(poses: List[Any]) -> float:
        """Compute total 2D path length from a list of poses or dicts."""
        if len(poses) < 2:
            return 0.0
        total_length = 0.0
        for i in range(1, len(poses)):
            prev = poses[i - 1]
            curr = poses[i]

            if isinstance(prev, np.ndarray):
                prev_pos = prev[:2, 3]
            elif isinstance(prev, dict):
                prev_pos = np.array([prev["x"], prev["y"]])
            else:
                continue

            if isinstance(curr, np.ndarray):
                curr_pos = curr[:2, 3]
            elif isinstance(curr, dict):
                curr_pos = np.array([curr["x"], curr["y"]])
            else:
                continue

            total_length += np.linalg.norm(curr_pos - prev_pos)
        return total_length

    @staticmethod
    def compute_spl(
        success: bool, actual_path_length: float, shortest_path_length: float
    ) -> float:
        """Compute SPL (Success weighted by Path Length).

        ``SPL = success * (shortest / max(actual, shortest))``
        """
        if not success:
            return 0.0
        if shortest_path_length <= 0:
            return 0.0
        return shortest_path_length / max(actual_path_length, shortest_path_length)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_images(
        self,
        pose: np.ndarray,
        scene_id: str,
        save_dir: str,
        image_id: int,
    ) -> List:
        """Render multi-view images at the given pose."""
        images = self.renderer.render_at_pose(
            pose,
            scene_id,
            save_dir=save_dir if self.config.save_render_images else None,
            image_id=image_id,
        )

        return images

    # ------------------------------------------------------------------
    # Result saving
    # ------------------------------------------------------------------

    @staticmethod
    def _make_error_result(episode: Episode, task: Task) -> dict:
        """Create a minimal error result dict."""
        return {
            "episode_id": episode.episode_id,
            "task_id": task.task_id,
            "status": "error",
            "target_label": "point_goal",
            "gt_taget_instance": "point_goal_target",
            "steps": 0,
            "travel_length": 0.0,
            "shortest_path_length": 0.0,
            "distance_to_goal": float("inf"),
            "success": False,
            "oracle_success": False,
            "metrics": {},
        }

    @staticmethod
    def _save_task_result(result: dict, output_dir: str) -> None:
        """Save a single task result to ``result.json``."""
        result_path = os.path.join(output_dir, "result.json")
        # Make a serialisable copy (convert numpy types)
        serialisable = _make_serialisable(result)
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(serialisable, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Result saving helpers end here; aggregation is handled by metrics.py
    # ------------------------------------------------------------------


# ============================================================================
# JSON serialisation helper
# ============================================================================

def _make_serialisable(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON."""
    if isinstance(obj, dict):
        return {k: _make_serialisable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serialisable(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
        return str(obj)
    return obj
