"""Agent interface for point-goal navigation.

Defines the observation and prediction dataclasses, plus the abstract base
class that every point-goal agent must implement.

The evaluator treats the agent as a complete black box: it sends an
:class:`Observation` and receives a :class:`WaypointPrediction`.  Any
internal architecture (single-system, dual-system, ensemble, etc.) is the
agent's own responsibility and is invisible to the benchmark.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Observation:
    """A single observation delivered to the agent at each evaluation step.

    Required fields are always populated by the evaluator.  Optional fields are
    filled only when the corresponding runner/evaluator flag is enabled (e.g.
    ``provide_history``, ``provide_occ_map``).

    Attributes
    ----------
    images : dict
        Multi-view RGB images keyed by camera name.  Default keys are
        ``"left"``, ``"front"``, ``"right"``. Values may be PIL images
        (preferred for ABot-N1 parity with the legacy pipeline) or uint8
        ndarray values of shape ``(640, 720, 3)``.
    target_position : ndarray
        Goal position in the agent's local coordinate frame as
        ``[front, left]`` in metres.
    position : ndarray
        Agent position in world coordinates ``[x, y, z]``.
    rotation : ndarray
        4x4 camera-to-world pose matrix.
    heading : float
        Agent yaw angle in radians.
    step_count : int
        Number of steps taken so far in the current episode.
    distance_to_goal : float
        Euclidean distance (metres) from the agent to the goal in the XY
        plane.
    history_images : list, optional
        List of per-step image dicts from previous steps.
    history_poses : list, optional
        List of 4x4 pose matrices from previous steps.
    occ_map : ndarray, optional
        Occupancy grid for the current scene.
    height_map : ndarray, optional
        Height map for the current scene.
    meta_data : dict, optional
        Scene metadata (coordinate transform parameters, etc.).
    extra : dict
        Catch-all for task-specific or experimental data.
    """

    # Required fields
    images: Dict[str, Any]
    target_position: np.ndarray
    position: np.ndarray
    rotation: np.ndarray
    heading: float
    step_count: int
    distance_to_goal: float

    # Optional fields (enabled by runner flags)
    goal_world: Optional[np.ndarray] = None
    history_images: Optional[List[Dict[str, np.ndarray]]] = None
    history_poses: Optional[List[np.ndarray]] = None
    occ_map: Optional[np.ndarray] = None
    height_map: Optional[np.ndarray] = None
    meta_data: Optional[Dict] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WaypointPrediction:
    """Agent's prediction for the next waypoint.

    Attributes
    ----------
    waypoint : ndarray
        Predicted next waypoint in the agent's local coordinate frame as
        ``[front, left]`` in metres.
    arrive : bool
        Whether the agent believes it has arrived at the goal and wishes
        to stop.
    directions : ndarray, optional
        Unit direction vector(s) associated with the predicted waypoint(s).
        Shape ``(N, 2)`` where N matches the number of waypoints.
    confidence : float, optional
        A scalar confidence score in ``[0, 1]`` for the prediction.
    extra : dict
        Catch-all for agent-specific debug information (e.g. slow-system
        affordance pixels) that the evaluator may optionally visualise.
    """

    waypoint: np.ndarray
    arrive: bool = False
    directions: Optional[np.ndarray] = None
    confidence: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class BasePointGoalAgent(ABC):
    """Abstract base class for point-goal navigation agents.

    Subclasses must implement :meth:`reset` and :meth:`predict`.  These are
    the only two methods the evaluator will ever call.

    The evaluator treats the agent as a **black box**: it sends an
    :class:`Observation` and expects a :class:`WaypointPrediction` back.
    Any internal architecture decisions -- single model, dual-system
    fast/slow reasoning, ensembles, caching, etc. -- belong entirely
    inside the agent's :meth:`predict` implementation.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset the agent's internal state at the start of a new episode."""

    @abstractmethod
    def predict(self, observation: Observation) -> WaypointPrediction:
        """Predict the next waypoint given the current observation.

        Parameters
        ----------
        observation : Observation
            The current observation, including images, target position,
            and optional history data.

        Returns
        -------
        WaypointPrediction
            The agent's waypoint prediction and stop decision.
        """
