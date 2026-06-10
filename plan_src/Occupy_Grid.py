import numpy as np
from typing import Optional
from dataclasses import dataclass
import logging

# Enviroment Imports
from sim_src.enviroment.World import World
from sim_src.enviroment.Bounds import Bounds
from sim_src.enviroment.Obstacle import CircularObstacle, RectObstacle_Aligned, PolyObstacle

@dataclass
class OccupyGrid:
    world: World
    resolution: float

    def build_grid(self):
        # Calculate grid dimensions
        self.x_min, self.x_max = self.world.bounds.xmin, self.world.bounds.xmax
        self.y_min, self.y_max = self.world.bounds.ymin, self.world.bounds.ymax

        x_size = int(np.ceil((self.x_max - self.x_min) / self.resolution))
        y_size = int(np.ceil((self.y_max - self.y_min) / self.resolution))

        
        # Initialize grid with False (free space)
        self.grid = np.zeros((x_size, y_size), dtype=bool)

        self.mark_obstacles()

    def mark_obstacles(self):
            # Iterate over the exact size of the numpy array
            for i in range(self.grid.shape[0]):
                for j in range(self.grid.shape[1]):
                    
                    # Calculate the exact physical coordinate for the CENTER of the cell
                    x = self.x_min + (i + 0.5) * self.resolution
                    y = self.y_min + (j + 0.5) * self.resolution
                    
                    # Check collision (using your newly renamed method!)
                    if not self.world.is_collision_free(np.array([x, y]), vehicle_radius=1.0):
                        self.grid[i, j] = True

    def to_idx(self, point: np.ndarray) -> tuple[int, int]:
         # Get grid indicies for a given point
        x_idx = int((point[0] - self.x_min) / self.resolution)
        y_idx = int((point[1] - self.y_min) / self.resolution)
        return (x_idx, y_idx)
    

    def to_point(self, point: tuple[int, int]) -> np.ndarray:
         # Get world coodinates from grid index
         x = 0.0
         y = 0.0
         return np.array([x, y], dtype=float)

'''
    def mark_obstacles(self):
        if self.world.obstacles is not None:
            for obs in self.world.obstacles:
                if isinstance(obs, CircularObstacle):
                    self.mark_circle_on_grid(obs)
                elif isinstance(obs, RectObstacle_Aligned):
                    self.mark_rectangle_on_grid(obs)
                elif isinstance(obs, PolyObstacle):
                    self.mark_polygon_on_grid(obs)
                else:
                    logging.error(f"Unknown obstacle type: {type(obs)}")
                    raise ValueError(f"Unknown obstacle type: {type(obs)}")


    def mark_circle_on_grid(self, obs: CircularObstacle):
        for i, x in enumerate(np.arange(self.world.bounds.xmin, self.world.bounds.xmax, self.resolution)):
            for j, y in enumerate(np.arange(self.world.bounds.ymin, self.world.bounds.ymax, self.resolution)):
                if (x - obs.c[0])**2 + (y - obs.c[1])**2 <= obs.r**2:
                    self.grid[i, j] = True  # Mark as occupied

    def mark_rectangle_on_grid(self, obs: RectObstacle_Aligned):
        for i, x in enumerate(np.arange(self.world.bounds.xmin, self.world.bounds.xmax, self.resolution)):
            for j, y in enumerate(np.arange(self.world.bounds.ymin, self.world.bounds.ymax, self.resolution)):
                if obs.xmin <= x <= obs.xmax and obs.ymin <= y <= obs.ymax:
                    self.grid[i, j] = True  # Mark as occupied

    def mark_polygon_on_grid(self, obs: PolyObstacle):
        pass
'''