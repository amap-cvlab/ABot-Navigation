"""Agent interface for POI-goal navigation.

Extends the point-goal agent interface with a POI name field in the
observation.  The agent navigates to a named Point of Interest (e.g. a
shop name) instead of raw coordinates.

The evaluator treats the agent as a complete black box: it sends a
:class:`PoiGoalObservation` and receives a :class:`WaypointPrediction`.
Any internal architecture (single-system, dual-system, ensemble, etc.)
is the agent's own responsibility and is invisible to the benchmark.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from abotn_evaluator.interface.point_goal import (
    Observation,
    WaypointPrediction,
)


@dataclass
class PoiGoalObservation(Observation):
    """Observation for POI-goal navigation.

    Inherits all fields from the point-goal :class:`Observation` and adds a
    ``poi_name`` field identifying the target Point of Interest.

    Attributes
    ----------
    poi_name : str
        The name of the POI the agent must navigate to (e.g. a shop or
        landmark name).  This replaces the explicit coordinate-based
        target used in point-goal navigation.
    """

    poi_name: str = ""


class BasePoiGoalAgent(ABC):
    """Abstract base class for POI-goal navigation agents.

    Subclasses must implement :meth:`reset` and :meth:`predict`.  These are
    the only two methods the evaluator will ever call.

    The evaluator treats the agent as a **black box**: it sends a
    :class:`PoiGoalObservation` and expects a :class:`WaypointPrediction`
    back.  Any internal architecture decisions -- single model, dual-system
    fast/slow reasoning, ensembles, caching, etc. -- belong entirely
    inside the agent's :meth:`predict` implementation.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset the agent's internal state at the start of a new episode."""

    @abstractmethod
    def predict(self, observation: PoiGoalObservation) -> WaypointPrediction:
        """Predict the next waypoint given the current observation.

        Parameters
        ----------
        observation : PoiGoalObservation
            The current observation, including images, POI name,
            target position, and optional history data.

        Returns
        -------
        WaypointPrediction
            The agent's waypoint prediction and stop decision.
        """
