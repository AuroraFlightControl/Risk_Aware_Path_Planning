import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon
import logging
from matplotlib.animation import FuncAnimation # Added import

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

def visualize_trajectory_with_time(world: World, trajectory: np.ndarray, time: np.ndarray):
    visualize_world(world)
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label='Trajectory')
    for i in range(len(trajectory)):
        if i % 100 == 0:  # Annotate every 100th point to avoid clutter
            plt.text(trajectory[i, 0], trajectory[i, 1], f'{time[i]:.1f}s', fontsize=8, ha='right')
    plt.legend()
    plt.title('Trajectory Visualization with Time Annotations')
    plt.show()

def animate_trajectory_with_time(world: World, trajectory: np.ndarray, time: np.ndarray, playback_speed_ms: int = 50):
    """
    Animates the agent's trajectory over time on the simulation world.
    
    :param world: The simulation world object.
    :param trajectory: (N, 2) numpy array of [x, y] positions.
    :param time: (N,) numpy array of simulation times.
    :param playback_speed_ms: Milliseconds between each frame rendering.
    """
    # 1. Setup the static background using your existing function
    visualize_world(world)
    
    # Grab the current figure and axes created by visualize_world
    fig = plt.gcf()
    ax = plt.gca()

    # 2. Initialize empty graphic objects for the animation
    # The comma after the variable name unpacks the single Line2D object from the list
    path_line, = ax.plot([], [], 'b-', alpha=0.6, label='Path History')
    agent_dot, = ax.plot([], [], 'ko', markersize=8, label='Agent') 
    
    # Create a dynamic text box for the simulation time in the top left corner
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, 
                        fontsize=12, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Update the legend and title to reflect the animation elements
    ax.legend(loc='upper right')
    ax.set_title('Agent Trajectory Animation')

    # 3. The update function called for each frame
    def update(frame):
        # Update the trajectory line history up to the current frame
        path_line.set_data(trajectory[:frame+1, 0], trajectory[:frame+1, 1])
        
        # Update the agent's current position marker
        # Note: set_data expects sequences (lists/arrays), hence the brackets
        agent_dot.set_data([trajectory[frame, 0]], [trajectory[frame, 1]])
        
        # Update the timer text
        time_text.set_text(f'Sim Time: {time[frame]:.2f} s')
        
        return path_line, agent_dot, time_text

    # 4. Create the animation
    # blit=True optimizes rendering by only updating the changing graphic elements
    ani = FuncAnimation(fig, update, frames=len(trajectory), 
                        interval=playback_speed_ms, blit=True, repeat=False)
    
    plt.show()
    
    # Note: It is often necessary to return the 'ani' object so Python's 
    # garbage collector doesn't destroy the animation if run in certain IDEs.
    return ani