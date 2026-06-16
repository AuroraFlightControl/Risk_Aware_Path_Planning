import numpy as np
from dataclasses import dataclass
from typing import Optional, List
from plan_src.planner_base import Planner, PlanResult
from sim_src.enviroment.World import World
import logging

@dataclass
class HolonomicAgent:
    position: np.ndarray
    velocity: np.ndarray
    current_trgt: np.ndarray # Current target position for the agent
    world: World  # Reference to the world for planner action
    radius: float = 1.0
    planner: Optional[Planner] = None  # Optional planner for the agent
    plan: Optional[List[np.ndarray]] = None  # Optional path for the agent to follow
    planner_timout: bool = False

    def run_planner(self):
        """Run the planner to update the path and current target."""

        if self.planner is not None and self.plan is None:
            if np.allclose(self.position, self.world.start):
                self.PlanResult = self.planner.plan(start=self.world.start)
            else:
                self.PlanResult = self.planner.plan(start=self.position)
            self.plan = self.PlanResult.plan
            if self.plan is not None and len(self.plan) > 0:
                self.current_trgt = self.plan[0]  # Update current target to the first waypoint in the path
            else:
                self.current_target = self.world.goal
                logging.error('Failed to plan')
                self.planner_timout = True

    def wpt_cycle(self):
        """Waypoint cycling function to update the current target based on the path."""
        if self.plan is not None and len(self.plan) > 0:
            # Check if the agent is close enough to the current target
            if np.linalg.norm(self.position - self.current_trgt) < self.radius:
                # Move to the next waypoint in the path
                self.plan.append(self.plan.pop(0))  # Cycle the waypoints
                self.current_trgt = self.plan[0]  # Update the current target to the next waypoint

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
        if not self.planner_timout:
            self.run_planner()
        self.wpt_cycle()

        self.velocity = self.control_input(self.current_trgt)
        self.position += self.velocity * dt

    