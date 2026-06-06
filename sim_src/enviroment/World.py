from dataclasses import dataclass
from typing import List, Optional
import numpy as np

from sim_src.enviroment import Obstacle
from .Bounds import Bounds



@dataclass
class World:
    bounds: Bounds
    start: np.ndarray
    goal: np.ndarray
    obstacles: Optional[List[object]] = None
    Traffic: Optional[List[object]] = None


    def step(self, dt: float) -> None:
        """Advance the state of all dynamic elements in the world."""
        pass

    
    def collision_check(self, point: np.ndarray, vehicle_radius: float = 1.0, time=0.0) -> bool:
        # Hard World Boundry and Collision Check
        if not self.bounds.in_bounds(point):
            return False

        # Obstacle Collision Check

        # Check Collisions with Traffic

        return True