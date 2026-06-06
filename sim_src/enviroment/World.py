from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from sim_src.enviroment.Obstacle import Obstacle
from .Bounds import Bounds



@dataclass
class World:
    bounds: Bounds
    start: np.ndarray
    goal: np.ndarray
    obstacles: Optional[List[Obstacle]] = None
    traffic: Optional[List[object]] = None


    def step(self, dt: float) -> None:
        """Advance the state of all dynamic elements in the world."""
        pass

    
    def is_collision_free(self, point: np.ndarray, vehicle_radius: float = 1.0, time=0.0) -> bool:
        # Hard World Boundry and Collision Check
        if not self.bounds.in_bounds(point):
            return False

        # Obstacle Collision Check
        if self.obstacles is not None:
            for obs in self.obstacles:
                if obs.contains(point, inflate=vehicle_radius):
                    return False

        # Check Collisions with Traffic

        return True