"""Example random point-goal navigation agent.

A minimal agent that moves toward the target position with additive noise.
Useful for testing the evaluation pipeline and as a template for new agents.

Usage::

    python -m abotn_evaluator.point_goal.runner \\
        --data-dir /path/to/scenes \\
        --render-url http://localhost:7001/render_gs \\
        --output-dir ./eval_output \\
        --agent-module agent_examples.point_goal_random:RandomPointGoalAgent \\
        --max-steps 50 \\
        --arrive-threshold 0.5
"""

import numpy as np

from abotn_evaluator.interface.point_goal import BasePointGoalAgent, Observation, WaypointPrediction


class RandomPointGoalAgent(BasePointGoalAgent):
    """A simple agent that navigates toward the target with random noise.

    At each step the agent:
    1. Reads the target position in local coordinates from the observation.
    2. Computes a unit direction vector toward the target.
    3. Adds Gaussian noise to the direction.
    4. Outputs a waypoint at a fixed step distance along the noisy direction.
    5. Declares arrival when the distance to goal is below a threshold.

    Parameters
    ----------
    step_size : float
        Distance (metres) to advance per step.
    noise_std : float
        Standard deviation of Gaussian noise added to the direction vector.
    arrive_distance : float
        Distance threshold (metres) below which the agent declares arrival.
    """

    def __init__(
        self,
        step_size: float = 1.0,
        noise_std: float = 0.3,
        arrive_distance: float = 0.5,
    ) -> None:
        self.step_size = step_size
        self.noise_std = noise_std
        self.arrive_distance = arrive_distance
        self._step_count = 0

    def reset(self) -> None:
        """Reset internal step counter."""
        self._step_count = 0

    def predict(self, observation: Observation) -> WaypointPrediction:
        """Predict the next waypoint by moving toward the target with noise.

        Parameters
        ----------
        observation : Observation
            Current observation from the evaluator.

        Returns
        -------
        WaypointPrediction
            Waypoint in ``[front, left]`` local coordinates.
        """
        self._step_count += 1
        target = observation.target_position  # [front, left]
        dist = float(np.linalg.norm(target))

        # Check if we should declare arrival
        if observation.distance_to_goal < self.arrive_distance:
            return WaypointPrediction(
                waypoint=np.array([0.0, 0.0], dtype=np.float32),
                arrive=True,
                confidence=1.0,
            )

        # Compute direction toward target
        if dist > 1e-6:
            direction = target / dist
        else:
            direction = np.array([1.0, 0.0], dtype=np.float32)

        # Add noise
        noise = np.random.randn(2).astype(np.float32) * self.noise_std
        noisy_direction = direction + noise
        norm = float(np.linalg.norm(noisy_direction))
        if norm > 1e-6:
            noisy_direction = noisy_direction / norm

        # Compute waypoint at step_size distance
        actual_step = min(self.step_size, dist)
        waypoint = noisy_direction * actual_step

        # Build direction array for pose computation
        directions = noisy_direction.reshape(1, 2)

        return WaypointPrediction(
            waypoint=waypoint.reshape(1, 2),
            arrive=False,
            directions=directions,
            confidence=max(0.0, 1.0 - self.noise_std),
        )
