from dataclasses import dataclass
from typing import Optional, Protocol, List
import numpy as np

from sim_src.enviroment.World import World

@dataclass
class PlanResult:
    plan: List[np.ndarray]  # List of waypoints in the planned path
    success: bool           # Flag indicating if the planning was successful
    reason: Optional[str] = None  # Optional reason for failure if planning was not successful

class Planner(Protocol):
    def plan(self, world: World, start: Optional[np.ndarray] = None) -> PlanResult:
        """Plan a path from start to goal in the given world."""
        ...
        

