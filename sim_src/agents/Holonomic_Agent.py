import numpy as np
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class HolonomicAgent:
    position: np.ndarray
    velocity: np.ndarray
    current_trgt: np.ndarray # Current target position for the agent
    radius: float = 1.0
    path: Optional[List[np.ndarray]] = None  # Optional path for the agent to follow

    def wpt_cycle(self):
        """Waypoint cycling function to update the current target based on the path."""
        if self.path is not None and len(self.path) > 0:
            # Check if the agent is close enough to the current target
            if np.linalg.norm(self.position - self.current_trgt) < self.radius:
                # Move to the next waypoint in the path
                self.path.append(self.path.pop(0))  # Cycle the waypoints
                self.current_trgt = self.path[0]  # Update the current target to the next waypoint

    def control_input(self, target: np.ndarray) -> np.ndarray:
        """Compute the control input to move towards the target."""
        direction = target - self.position
        norm = np.linalg.norm(direction)

        constant_speed = 5.0  # You can adjust this value as needed

        if norm > 0:
            return direction / norm * constant_speed  # Normalize to get unit vector and scale by constant speed
        else:
            return np.zeros_like(direction)  # No movement if already at the target

    def step(self, dt: float) -> None:
        """Update the agent's position and velocity based on the control input."""

        self.wpt_cycle()

        self.velocity = self.control_input(self.current_trgt)
        self.position += self.velocity * dt

    