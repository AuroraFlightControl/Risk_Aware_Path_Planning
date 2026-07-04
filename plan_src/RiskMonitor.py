from dataclasses import dataclass
import numpy as np


# Enviroment Loading
from sim_src.enviroment.World import World
from sim_src.enviroment.Obstacle import *


@dataclass
class RiskMonitor:
    world: World
    d_safe_obs: float = 3.0
    d_safe_tfc: float = 5.0

    def Eval_Static_Risk(self, point: np.ndarray):
        if self.world.obstacles is None:
            return 0.0
        
        max_risk = 0.0

        for obs in self.world.obstacles:
            d_clear = obs.distance(point)

            if d_clear < self.d_safe_obs:
                risk = (self.d_safe_obs - d_clear) / self.d_safe_obs

                if risk > max_risk:
                    max_risk = risk

        return max_risk
    
    def generate_static_risk_field(self, resolution: float = 1.0) -> dict:
        """
        Generates a discrete grid representation of the risk field.
        Returns a dictionary containing the grid and its spatial metadata.
        """
        x_min, x_max = self.world.bounds.xmin, self.world.bounds.xmax
        y_min, y_max = self.world.bounds.ymin, self.world.bounds.ymax

        x_size = int(np.ceil((x_max - x_min) / resolution))
        y_size = int(np.ceil((y_max - y_min) / resolution))
        
        # Initialize grid with 0.0 (free space)
        grid = np.zeros((x_size, y_size))

        for x in range(x_size):
            for y in range(y_size):
                query_pt = np.array([x_min + x * resolution, y_min + y * resolution])
                grid[x, y] = self.Eval_Static_Risk(query_pt)
                
        # Return the field as a packaged data structure, do not pollute self
        return {
            "risk_array": grid,
            "resolution": resolution,
            "origin": (x_min, y_min),
            "shape": (x_size, y_size)
        }

    def Eval_Traffic_Risk(self, point: np.ndarray):
        if self.world.traffic is None:
            return 0.0
        
        max_risk = 0.0

        for tfc in self.world.traffic:
            d_clear = tfc.distance(point)

            if d_clear < self.d_safe_tfc:
                risk = (self.d_safe_tfc - d_clear) / self.d_safe_tfc

                if risk > max_risk:
                    max_risk = risk

        return max_risk
            