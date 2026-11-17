import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
from matplotlib.patches import Circle
from dataclasses import dataclass
import math


# Simple Bounds Class to hold the bounds and bouundry checking for the world.
@dataclass
class Bounds:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def in_bounds(self, point: np.ndarray) -> bool:
        """Check if a point is within the bounds of the world."""
        x, y = point
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax

@dataclass
class Obstacle:
    position: np.ndarray                        # Position of the obstacle
    radius: float                               # Radius of the obstacle

    def distance(self, point: np.ndarray) -> float:
        dist_to_center = math.dist(point, self.position)
        return  max(0.0, dist_to_center - self.radius)


# Simple World Class to hold the obstacles and the bounds of the world
@dataclass
class World:
    bounds: Bounds                              # Bounds of the world
    start: np.ndarray                           # Start position of the agent
    goal: np.ndarray                            # Goal position of the agent
    obstacles: Optional[List[Obstacle]] = None    # List of obstacles in the world

def static_risk(world: World, point: np.ndarray, safety_radius: float = 5.0) -> float:
    if world.obstacles is None:
        return float(0.0)
    
    max_risk = 0.0

    for obs in world.obstacles:
        d_clear = obs.distance(point)

        if d_clear < safety_radius:
            risk = (safety_radius - d_clear) / safety_radius

            if risk > max_risk:
                max_risk = risk

    return float(max_risk)

def risk_map(world: World, resolution: float, safety_radius: float = 1.0) -> np.ndarray:
    # Discritize the world into a grid based on the resolution
    x_min, x_max = world.bounds.xmin, world.bounds.xmax
    y_min, y_max = world.bounds.ymin, world.bounds.ymax

    x_size = int(np.ceil((x_max - x_min) / resolution))
    y_size = int(np.ceil((y_max - y_min) / resolution))

    # Initialize grid with 0.0 (free space)
    grid = np.zeros((x_size, y_size))

    for x in range(x_size):
        for y in range(y_size):
            query_pt = np.array([x_min + x * resolution, y_min + y * resolution])
            grid[x, y] = static_risk(world, query_pt, safety_radius)

    return grid


def visualize(world: World) -> None:
    """Visualize the world with obstacles, start, and goal positions."""
    fig, ax = plt.subplots()

    # Plot the bounds of the world
    ax.set_xlim(world.bounds.xmin, world.bounds.xmax)
    ax.set_ylim(world.bounds.ymin, world.bounds.ymax)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('World Visualization')
    ax.set_xlabel('Position East (ft)')
    ax.set_ylabel('Position North (ft)')

    # Plot start and goal positions
    #ax.plot(world.start[0], world.start[1], 'go', label='Start')  # Green dot for start
    #ax.plot(world.goal[0], world.goal[1], 'ro', label='Goal')     # Red dot for goal

    # Plot obstacles
    if world.obstacles:
        for obs in world.obstacles:
            circle = Circle((obs.position[0], obs.position[1]), obs.radius, color='black', alpha=0.1)
            ax.add_patch(circle)


    risk_field = risk_map(world, resolution=0.1, safety_radius=10.0)
    extent = (world.bounds.xmin, world.bounds.xmax, world.bounds.ymin, world.bounds.ymax)
    im = ax.imshow(risk_field.T, extent=extent, origin='lower', cmap='hot', alpha=0.5)
    plt.colorbar(im, ax=ax, label='Risk Level')



    ax.legend()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()


if __name__ == "__main__":
    # Example usage
    bounds = Bounds(xmin=0, xmax=50, ymin=0, ymax=50) 
    start = np.array([10, 10])
    goal = np.array([90, 90])

    # Define some obstacles
    obstacles = [
        Obstacle(position=np.array([15, 37.5]), radius=10),
        Obstacle(position=np.array([37.5, 15]), radius=10),
        #Obstacle(position=np.array([40, 40]), radius=20)
    ]


    world = World(bounds=bounds, start=start, goal=goal, obstacles=obstacles)



    visualize(world)