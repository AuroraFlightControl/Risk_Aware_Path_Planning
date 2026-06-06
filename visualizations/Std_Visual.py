import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
import logging

from sim_src.enviroment.Obstacle import CircularObstacle, RectObstacle_Aligned, PolyObstacle
from sim_src.enviroment.World import World



def visualize_world(world: World):
    fig, ax = plt.subplots()
    
    # Set the limits of the plot based on world bounds
    ax.set_xlim(world.bounds.xmin, world.bounds.xmax)
    ax.set_ylim(world.bounds.ymin, world.bounds.ymax)
    ax.set_aspect('equal', adjustable='box')

    # Plot Start and Goal Points
    ax.plot(world.start[0], world.start[1], 'go', label='Start')
    ax.plot(world.goal[0], world.goal[1], 'ro', label='Goal')

    # Plot Obstacles
    if world.obstacles is not None:
        for obs in world.obstacles:
            if isinstance(obs, CircularObstacle):
                logging.info(f"Visualizing Circular Obstacle: Center=({obs.c[0]}, {obs.c[1]}), Radius={obs.r}")
                circle = Circle((obs.c[0], obs.c[1]), obs.radius(), color='gray', alpha=0.5)
                ax.add_patch(circle)
            elif isinstance(obs, RectObstacle_Aligned):
                logging.info(f"Visualizing Rectangular Obstacle: ({obs.xmin}, {obs.ymin}) to ({obs.xmax}, {obs.ymax})")
                rect = Rectangle((obs.xmin, obs.ymin), obs.xmax - obs.xmin, obs.ymax - obs.ymin, color='gray', alpha=0.5)
                ax.add_patch(rect)
            elif isinstance(obs, PolyObstacle):
                logging.info(f"Visualizing Polygonal Obstacle with vertices: {obs.verts}")
                polygon = Polygon(obs.verts, color='gray', alpha=0.5)
                ax.add_patch(polygon)

    else:
        logging.info("No obstacles to visualize.")

    # Plot With Grid
    ax.grid(True)
    ax.legend()

    ax.set_title('World Visualization')
    ax.set_xlabel('Position X (feet)')
    ax.set_ylabel('Position Y (feet)')

def visualize_enviroment(world: World):
    visualize_world(world)
    plt.legend()
    plt.title('World Visualization')
    plt.show()

def visualize_trajectory(world: World, trajectory: np.ndarray):
    visualize_world(world)
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label='Trajectory')
    plt.legend()
    plt.title('Trajectory Visualization')
    plt.show()