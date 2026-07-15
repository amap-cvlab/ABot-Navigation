"""Scene management for Gaussian splatting point-goal evaluation.

Handles loading of episodes and tasks from a local directory structure,
coordinate conversions between world and pixel space, and collision
detection on occupancy maps.

No OSS (Object Storage Service) loading -- all data is read from local files.
Visualization methods are deliberately excluded; use ``common.visualization``
for rendering overlays on occupancy maps.
"""

import glob
import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class Task:
    """A single navigation task (trajectory) within an episode.

    Attributes
    ----------
    task_id : str
        Unique identifier, typically the trajectory filename stem
        (e.g. ``"traj_0"``).
    trajectory : list of dict
        Sequence of waypoints, each a dict with at least ``x``, ``y``, ``z``,
        ``roll``, ``pitch``, ``yaw`` keys.
    label : dict
        Metadata label from the original trajectory JSON.
    start_pose : dict or None
        First point in *trajectory* (convenience accessor).
    end_pose : dict or None
        Last point in *trajectory* (convenience accessor).
    goal_label : str
        Semantic label of the navigation target (extracted from *label*).
    """

    task_id: str
    trajectory: List[Dict[str, float]]
    label: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, json_path: str) -> "Task":
        """Load a task from a ``traj_*.json`` file."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        task_id = os.path.basename(json_path).replace(".json", "")
        return cls(
            task_id=task_id,
            trajectory=data.get("trajectory", []),
            label=data.get("label", {}),
        )

    @property
    def start_pose(self) -> Optional[Dict[str, float]]:
        if self.trajectory:
            return self.trajectory[0]
        return None

    @property
    def end_pose(self) -> Optional[Dict[str, float]]:
        if self.trajectory:
            return self.trajectory[-1]
        return None

    @property
    def goal_label(self) -> str:
        return self.label.get("extend", {}).get("goal_label", "")


@dataclass
class Episode:
    """A scene-level episode containing one or more navigation tasks.

    Attributes
    ----------
    episode_id : str
        Unique scene identifier (typically the directory name).
    scene_path : str
        Absolute path to the local scene directory.
    tasks : list of Task
        Navigation tasks loaded from ``traj_*.json`` files.
    occ_map : ndarray or None
        Occupancy grid (grayscale uint8).  White (>=128) = free, black (<128) = obstacle.
    height_map : ndarray or None
        Per-pixel ground height (metres).
    meta_data : dict or None
        Coordinate-system metadata parsed from ``occ_map_meta.txt``.
    """

    episode_id: str
    scene_path: str
    tasks: List[Task] = field(default_factory=list)
    occ_map: Optional[np.ndarray] = None
    height_map: Optional[np.ndarray] = None
    meta_data: Optional[Dict] = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_scene_dir(cls, scene_dir: str) -> "Episode":
        """Create an episode from a local scene directory.

        Loads trajectory tasks and, if present, map / height data.
        """
        episode_id = os.path.basename(scene_dir)
        episode = cls(episode_id=episode_id, scene_path=scene_dir)
        episode._load_tasks()
        episode._load_map_data(scene_dir)
        return episode

    # ------------------------------------------------------------------
    # Internal loaders
    # ------------------------------------------------------------------

    def _load_tasks(self) -> None:
        traj_files = sorted(glob.glob(os.path.join(self.scene_path, "traj_*.json")))
        for traj_file in traj_files:
            task = Task.from_json(traj_file)
            self.tasks.append(task)

    def _load_map_data(self, base_path: str) -> None:
        """Load occ_map, height_map, and meta_data from local files.

        Expected directory layout::

            {base_path}/
                map/
                    occ_map.png
                    occ_map_meta.txt
                    occ_map_height.tiff
        """
        map_dir = os.path.join(base_path, "map")

        # Occupancy map
        occ_path = os.path.join(map_dir, "occ_map.png")
        if os.path.exists(occ_path):
            self.occ_map = cv2.imread(occ_path, cv2.IMREAD_GRAYSCALE)

        # Metadata
        meta_path = os.path.join(map_dir, "occ_map_meta.txt")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta_data = _parse_meta_text(f.read())

        # Height map
        height_path = os.path.join(map_dir, "occ_map_height.tiff")
        if os.path.exists(height_path):
            try:
                img = Image.open(height_path)
                self.height_map = np.array(img)
            except Exception as exc:
                print(f"Warning: failed to load height_map from {height_path}: {exc}")

    # ------------------------------------------------------------------
    # Occupancy-map dilation
    # ------------------------------------------------------------------

    def get_meter_per_pixel(self) -> Optional[float]:
        """Return the average resolution (metres / pixel) of the occ_map."""
        if self.meta_data is None:
            return None
        try:
            mpp_x = float(self.meta_data["COORDINATE_RANGE_X"]) / float(
                self.meta_data["IMAGE_WIDTH"]
            )
            mpp_y = float(self.meta_data["COORDINATE_RANGE_Y"]) / float(
                self.meta_data["IMAGE_HEIGHT"]
            )
        except Exception:
            return None
        return (mpp_x + mpp_y) / 2.0

    def dilate_free_space(self, dilation_pixels: int) -> None:
        """Dilate the free-space region of the occ_map by *dilation_pixels*.

        This effectively shrinks obstacles, providing tolerance for
        annotation inaccuracies.
        """
        if dilation_pixels <= 0 or self.occ_map is None:
            return
        occ = self.occ_map
        if len(occ.shape) == 3:
            gray = cv2.cvtColor(occ, cv2.COLOR_BGR2GRAY)
        else:
            gray = occ
        free_mask = (gray >= 128).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * dilation_pixels + 1, 2 * dilation_pixels + 1),
        )
        dilated_free = cv2.dilate(free_mask, kernel, iterations=1)
        if len(occ.shape) == 3:
            self.occ_map = cv2.cvtColor(dilated_free, cv2.COLOR_GRAY2BGR)
        else:
            self.occ_map = dilated_free

    def dilate_free_space_by_meter(self, dilation_meters: float) -> None:
        """Dilate free space by a physical distance (metres).

        Converts *dilation_meters* to pixels using the scene metadata and
        delegates to :meth:`dilate_free_space`.
        """
        if dilation_meters <= 0 or self.occ_map is None:
            return
        mpp = self.get_meter_per_pixel()
        if mpp is None or mpp <= 0:
            print(
                f"Warning: [Episode {self.episode_id}] cannot compute "
                f"meter_per_pixel, skipping dilation"
            )
            return
        dilation_pixels = max(1, int(round(dilation_meters / mpp)))
        self.dilate_free_space(dilation_pixels)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def has_map_data(self) -> bool:
        """Check whether the minimum map data for point-goal evaluation is available."""
        return all([
            self.occ_map is not None,
            self.meta_data is not None,
            self.height_map is not None,
        ])

    @property
    def num_tasks(self) -> int:
        return len(self.tasks)


# ============================================================================
# Meta-text parser (standalone function)
# ============================================================================

def _parse_meta_text(content: str) -> Optional[Dict]:
    """Parse the contents of an ``occ_map_meta.txt`` file.

    Returns a dict with keys like ``TOP_LEFT_X``, ``IMAGE_WIDTH``,
    ``COORDINATE_RANGE_X``, etc.
    """
    try:
        data: Dict[str, Any] = {}
        for line in content.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("Top Left:"):
                coords = line.split(":", 1)[1].strip().strip("()").split(",")
                data["TOP_LEFT_X"] = float(coords[0].strip())
                data["TOP_LEFT_Y"] = float(coords[1].strip())
            elif line.startswith("Top Right:"):
                coords = line.split(":", 1)[1].strip().strip("()").split(",")
                data["TOP_RIGHT_X"] = float(coords[0].strip())
                data["TOP_RIGHT_Y"] = float(coords[1].strip())
            elif line.startswith("Bottom Left:"):
                coords = line.split(":", 1)[1].strip().strip("()").split(",")
                data["BOTTOM_LEFT_X"] = float(coords[0].strip())
                data["BOTTOM_LEFT_Y"] = float(coords[1].strip())
            elif line.startswith("Bottom Right:"):
                coords = line.split(":", 1)[1].strip().strip("()").split(",")
                data["BOTTOM_RIGHT_X"] = float(coords[0].strip())
                data["BOTTOM_RIGHT_Y"] = float(coords[1].strip())
            elif line.startswith("Image size in pixels:"):
                size_str = line.split(":", 1)[1].strip()
                width, height = map(int, size_str.split(","))
                data["IMAGE_WIDTH"] = width
                data["IMAGE_HEIGHT"] = height

        range_x = math.sqrt(
            (data["BOTTOM_RIGHT_X"] - data["BOTTOM_LEFT_X"]) ** 2
            + (data["BOTTOM_RIGHT_Y"] - data["BOTTOM_LEFT_Y"]) ** 2
        )
        range_y = math.sqrt(
            (data["BOTTOM_LEFT_X"] - data["TOP_LEFT_X"]) ** 2
            + (data["BOTTOM_LEFT_Y"] - data["TOP_LEFT_Y"]) ** 2
        )
        data["COORDINATE_RANGE_X"] = range_x
        data["COORDINATE_RANGE_Y"] = range_y
        return data
    except Exception as exc:
        print(f"Warning: failed to parse meta file: {exc}")
        return None


# ============================================================================
# GaussianScene
# ============================================================================

class GaussianScene:
    """Manager for a collection of Gaussian splatting scene episodes.

    Loads episodes from a local directory structure and provides coordinate
    conversion and collision detection utilities.

    Parameters
    ----------
    local_data_path : str
        Root directory containing per-scene subdirectories.
    local_map_path : str, optional
        Separate directory for map data.  If provided, map data is read
        from ``{local_map_path}/{episode_id}/`` instead of the episode's
        own ``scene_path``.
    """

    def __init__(
        self,
        local_data_path: str,
        local_map_path: Optional[str] = None,
    ) -> None:
        self.local_data_path = local_data_path
        self.local_map_path = local_map_path
        self.episodes: List[Episode] = []
        self._current_episode_idx = 0
        self._load_all_episodes()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_all_episodes(self) -> None:
        scene_dirs = sorted(
            d
            for d in glob.glob(os.path.join(self.local_data_path, "*"))
            if os.path.isdir(d)
        )
        print(f"Found {len(scene_dirs)} scene directories")

        for scene_dir in scene_dirs:
            episode = Episode.from_scene_dir(scene_dir)

            # If a separate map directory is provided, reload map data from there
            if self.local_map_path:
                map_base = os.path.join(self.local_map_path, episode.episode_id)
                if os.path.exists(map_base):
                    episode._load_map_data(map_base)

            self.episodes.append(episode)

        print(
            f"Loaded {len(self.episodes)} episodes, "
            f"{self.total_tasks} tasks total"
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_episode(self, idx: int) -> Episode:
        return self.episodes[idx]

    def get_episode_by_id(self, episode_id: str) -> Optional[Episode]:
        for ep in self.episodes:
            if ep.episode_id == episode_id:
                return ep
        return None

    def __len__(self) -> int:
        return len(self.episodes)

    def __iter__(self):
        self._current_episode_idx = 0
        return self

    def __next__(self) -> Episode:
        if self._current_episode_idx >= len(self.episodes):
            raise StopIteration
        episode = self.episodes[self._current_episode_idx]
        self._current_episode_idx += 1
        return episode

    @property
    def total_tasks(self) -> int:
        return sum(ep.num_tasks for ep in self.episodes)

    def iter_tasks(self):
        """Iterate over all ``(episode, task)`` pairs."""
        for episode in self.episodes:
            for task in episode.tasks:
                yield episode, task

    def summary(self) -> Dict[str, Any]:
        return {
            "num_episodes": len(self.episodes),
            "total_tasks": self.total_tasks,
            "episode_ids": [ep.episode_id for ep in self.episodes],
        }

    # ------------------------------------------------------------------
    # Pose conversion
    # ------------------------------------------------------------------

    @staticmethod
    def get_gaussian_pose(cur_info: Dict[str, float]) -> np.ndarray:
        """Convert a pose dict to a 4x4 homogeneous transform matrix.

        Parameters
        ----------
        cur_info : dict
            Must contain ``x``, ``y``, ``z``, ``roll``, ``pitch``, ``yaw``
            (Euler angles in radians).

        Returns
        -------
        ndarray
            4x4 camera-to-world pose matrix.
        """
        pitch = cur_info["pitch"]
        roll = cur_info["roll"]
        yaw = cur_info["yaw"]
        x = cur_info["x"]
        y = cur_info["y"]
        z = cur_info["z"]

        euler_angles_w = np.array([roll, pitch, yaw])
        rotation = R.from_euler("xyz", euler_angles_w, degrees=False).as_matrix()

        pose = np.eye(4)
        pose[:3, :3] = rotation
        pose[:3, 3] = np.array([x, y, z])
        return pose

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    @staticmethod
    def convert_actual_to_pixel(
        x_actual: float,
        y_actual: float,
        episode: Episode,
    ) -> tuple:
        """Convert world coordinates to occ_map pixel coordinates.

        Returns ``(x_pixel, y_pixel)`` as integers.
        """
        meta = episode.meta_data
        top_left_x = meta["TOP_LEFT_X"]
        top_left_y = meta["TOP_LEFT_Y"]
        width = meta["IMAGE_WIDTH"]
        height = meta["IMAGE_HEIGHT"]
        range_x = meta["COORDINATE_RANGE_X"]
        range_y = meta["COORDINATE_RANGE_Y"]

        sx = 1 if (meta["TOP_RIGHT_X"] - meta["TOP_LEFT_X"]) >= 0 else -1
        sy = 1 if (meta["BOTTOM_LEFT_Y"] - meta["TOP_LEFT_Y"]) >= 0 else -1

        if sx == 1:
            x_pixel = int((x_actual - top_left_x) * width / range_x)
        else:
            x_pixel = int((top_left_x - x_actual) * width / range_x)

        if sy == 1:
            y_pixel = int((y_actual - top_left_y) * height / range_y)
        else:
            y_pixel = int((top_left_y - y_actual) * height / range_y)

        return (x_pixel, y_pixel)

    @staticmethod
    def convert_pixel_to_actual(
        x_pixel: int,
        y_pixel: int,
        episode: Episode,
    ) -> tuple:
        """Convert occ_map pixel coordinates to world coordinates.

        Returns ``(x_actual, y_actual)`` as floats.
        """
        meta = episode.meta_data
        top_left_x = meta["TOP_LEFT_X"]
        top_left_y = meta["TOP_LEFT_Y"]
        width = meta["IMAGE_WIDTH"]
        height = meta["IMAGE_HEIGHT"]
        range_x = meta["COORDINATE_RANGE_X"]
        range_y = meta["COORDINATE_RANGE_Y"]

        sx = 1 if (meta["TOP_RIGHT_X"] - meta["TOP_LEFT_X"]) >= 0 else -1
        sy = 1 if (meta["BOTTOM_LEFT_Y"] - meta["TOP_LEFT_Y"]) >= 0 else -1

        if sx == 1:
            x_actual = top_left_x + (x_pixel * range_x / width)
        else:
            x_actual = top_left_x - (x_pixel * range_x / width)

        if sy == 1:
            y_actual = top_left_y + (y_pixel * range_y / height)
        else:
            y_actual = top_left_y - (y_pixel * range_y / height)

        return (x_actual, y_actual)

    # ------------------------------------------------------------------
    # Collision detection
    # ------------------------------------------------------------------

    @classmethod
    def check_point_collision_status(
        cls,
        pose: Any,
        episode: Episode,
        robot_radius_pixel: int = 2,
    ) -> bool:
        """Check whether a single pose falls on an obstacle.

        Parameters
        ----------
        pose : ndarray or dict
            Agent pose (4x4 matrix or dict with ``x``, ``y``).
        episode : Episode
            Scene episode (must have ``occ_map`` and ``meta_data``).
        robot_radius_pixel : int
            Half-size of the square footprint to check (pixels).

        Returns
        -------
        bool
            True if the point is in collision.
        """
        if episode.occ_map is None or episode.meta_data is None:
            return False

        occ_map = episode.occ_map
        if len(occ_map.shape) == 3:
            occ_map = cv2.cvtColor(occ_map, cv2.COLOR_BGR2GRAY)

        h, w = occ_map.shape[:2]

        if isinstance(pose, dict):
            wx, wy = pose["x"], pose["y"]
        elif isinstance(pose, np.ndarray):
            wx, wy = pose[0, 3], pose[1, 3]
        else:
            return False

        try:
            px, py = cls.convert_actual_to_pixel(wx, wy, episode)
        except Exception:
            return False

        if px < 0 or px >= w or py < 0 or py >= h:
            return True

        if robot_radius_pixel == 0:
            return bool(occ_map[py, px] < 128)
        else:
            y_min = max(0, py - robot_radius_pixel)
            y_max = min(h, py + robot_radius_pixel + 1)
            x_min = max(0, px - robot_radius_pixel)
            x_max = min(w, px + robot_radius_pixel + 1)
            local_patch = occ_map[y_min:y_max, x_min:x_max]
            return bool(np.any(local_patch < 128))

    @classmethod
    def check_line_collision_fast(
        cls,
        pose_start: Any,
        pose_end: Any,
        episode: Episode,
        step_size_pixel: float = 2.0,
    ) -> bool:
        """Check whether the straight-line path between two poses crosses an obstacle.

        Uses uniform sampling along the line in pixel space for speed.

        Returns
        -------
        bool
            True if any sample point along the line is in collision.
        """
        if episode.occ_map is None or episode.meta_data is None:
            return False

        occ_map = episode.occ_map
        if len(occ_map.shape) == 3:
            occ_map = cv2.cvtColor(occ_map, cv2.COLOR_BGR2GRAY)
        h, w = occ_map.shape[:2]

        def _get_xy(p):
            if isinstance(p, dict):
                return p["x"], p["y"]
            elif isinstance(p, np.ndarray):
                return p[0, 3], p[1, 3]
            return None, None

        x1, y1 = _get_xy(pose_start)
        x2, y2 = _get_xy(pose_end)
        if x1 is None or x2 is None:
            return False

        try:
            u1, v1 = cls.convert_actual_to_pixel(x1, y1, episode)
            u2, v2 = cls.convert_actual_to_pixel(x2, y2, episode)
        except Exception:
            return False

        dist_pixel = np.sqrt((u2 - u1) ** 2 + (v2 - v1) ** 2)
        if dist_pixel < 1e-6:
            return cls.check_point_collision_status(pose_end, episode)

        num_samples = max(2, int(dist_pixel / step_size_pixel))
        us = np.linspace(u1, u2, num_samples).astype(int)
        vs = np.linspace(v1, v2, num_samples).astype(int)

        if np.any(us < 0) or np.any(us >= w) or np.any(vs < 0) or np.any(vs >= h):
            return True

        sampled_values = occ_map[vs, us]
        return bool(np.any(sampled_values < 128))

    @classmethod
    def compute_line_collision_length(
        cls,
        pose_start: Any,
        pose_end: Any,
        episode: Episode,
        sample_step_meter: float = 0.05,
    ) -> float:
        """Compute the physical length (metres) of the path segment that lies
        inside obstacles.

        Uses dense sampling in world coordinates, then maps each sample to the
        occupancy grid to check collision status.

        Returns
        -------
        float
            Total collision length in metres.
        """
        if episode.occ_map is None or episode.meta_data is None:
            return 0.0

        occ_map = episode.occ_map
        if len(occ_map.shape) == 3:
            occ_map = cv2.cvtColor(occ_map, cv2.COLOR_BGR2GRAY)
        h, w = occ_map.shape[:2]

        def _get_xy(p):
            if isinstance(p, dict):
                return np.array([p["x"], p["y"]])
            elif isinstance(p, np.ndarray):
                return p[:2, 3]
            return np.array([0.0, 0.0])

        p1 = _get_xy(pose_start)
        p2 = _get_xy(pose_end)
        total_dist = float(np.linalg.norm(p2 - p1))
        if total_dist < 1e-6:
            return 0.0

        num_samples = max(2, int(total_dist / sample_step_meter))
        xs = np.linspace(p1[0], p2[0], num_samples)
        ys = np.linspace(p1[1], p2[1], num_samples)

        meta = episode.meta_data
        top_left_x = meta["TOP_LEFT_X"]
        top_left_y = meta["TOP_LEFT_Y"]
        range_x = meta["COORDINATE_RANGE_X"]
        range_y = meta["COORDINATE_RANGE_Y"]

        sx = 1 if (meta.get("TOP_RIGHT_X", top_left_x + range_x) - top_left_x) >= 0 else -1
        sy = 1 if (meta.get("BOTTOM_LEFT_Y", top_left_y + range_y) - top_left_y) >= 0 else -1

        if sx == 1:
            us = ((xs - top_left_x) * w / range_x).astype(int)
        else:
            us = ((top_left_x - xs) * w / range_x).astype(int)
        if sy == 1:
            vs = ((ys - top_left_y) * h / range_y).astype(int)
        else:
            vs = ((top_left_y - ys) * h / range_y).astype(int)

        in_bounds = (us >= 0) & (us < w) & (vs >= 0) & (vs < h)

        # Out-of-bounds points are treated as obstacles
        is_obstacle = np.ones(num_samples, dtype=bool)
        valid_us = us[in_bounds]
        valid_vs = vs[in_bounds]
        if len(valid_us) > 0:
            is_obstacle[in_bounds] = occ_map[valid_vs, valid_us] < 128

        # Accumulate lengths of contiguous obstacle segments
        collision_length = 0.0
        dists = np.linspace(0, total_dist, num_samples)
        in_collision = False
        enter_dist = 0.0

        for i in range(num_samples):
            if is_obstacle[i]:
                if not in_collision:
                    in_collision = True
                    enter_dist = dists[i]
            else:
                if in_collision:
                    in_collision = False
                    collision_length += dists[i] - enter_dist

        if in_collision:
            collision_length += total_dist - enter_dist

        return float(collision_length)

    # ------------------------------------------------------------------
    # Distance helpers
    # ------------------------------------------------------------------

    @staticmethod
    def compute_pose_distance(pose1: Any, pose2: Any) -> float:
        """Compute the 2D Euclidean distance between two poses."""
        if pose1 is None or pose2 is None:
            return 0.0

        def _get_xy(p):
            if isinstance(p, dict):
                return np.array([p["x"], p["y"]])
            elif isinstance(p, np.ndarray):
                return p[:2, 3]
            return np.array([0.0, 0.0])

        return float(np.linalg.norm(_get_xy(pose2) - _get_xy(pose1)))
