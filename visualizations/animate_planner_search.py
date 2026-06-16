import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import logging

from sim_src.enviroment.World import World
from visualizations.Std_Visual import visualize_world

def animate_planner_search(world: World, log_dir: str, filename_prefix: str, plan_id: int = 0, nodes_per_frame: int = 25, playback_speed_ms: int = 30):
    """
    Animates the node-by-node expansion of the planner's search tree.
    
    :param nodes_per_frame: Batch size of nodes to draw per frame. Increase for large A* grids.
    """
    # 1. Setup the Base Environment
    visualize_world(world)
    fig = plt.gcf()
    ax = plt.gca()

    # 2. Load the Logged Data
    try:
        nodes_df = pd.read_csv(f"{log_dir}/{filename_prefix}_sampled_nodes.csv")
        paths_df = pd.read_csv(f"{log_dir}/{filename_prefix}_paths.csv")
    except FileNotFoundError:
        logging.error("Log files not found. Ensure the planner was run in debug_mode.")
        return

    # Filter data to the specific replanning event
    nodes = nodes_df[nodes_df['plan_id'] == plan_id].copy()
    path = paths_df[paths_df['plan_id'] == plan_id].copy()

    if nodes.empty:
        logging.warning(f"No sampled nodes found for plan_id {plan_id}.")
        return

    # 3. Pre-process segments and colors outside the animation loop for speed
    segments = [((row['parent_x'], row['parent_y']), (row['x'], row['y'])) for _, row in nodes.iterrows()]
    
    # Map colors: Standard samples are gray, RRT* rewires are green
    color_map = {'sampled': 'lightgray', 'rewired': 'lightgreen'}
    types = nodes['type'].values
    segment_colors = [color_map.get(t, 'lightgray') for t in types]

    # Initialize the high-performance LineCollection
    tree_lc = LineCollection([], linewidths=0.8, alpha=0.8, zorder=1)
    ax.add_collection(tree_lc)

    # Initialize the final path line
    final_path_line, = ax.plot([], [], color='red', linewidth=2.5, marker='.', markersize=8, zorder=3, label='Final Path')

    # Calculate required frames based on batch size, adding buffer frames at the end to show the final path
    total_frames = (len(segments) // nodes_per_frame) + 15 

    # Formatting
    plt.title(f"Search Space Animation (Plan ID: {plan_id})")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()

    # 4. The Animation Update Loop
    def update(frame):
        # Calculate how many nodes should be visible in this frame
        current_idx = min(frame * nodes_per_frame, len(segments))

        # Update the LineCollection with the current batch of edges and their corresponding colors
        tree_lc.set_segments(segments[:current_idx])
        tree_lc.set_color(segment_colors[:current_idx])

        # Once the tree has fully expanded, draw the final path
        if current_idx >= len(segments) and not path.empty and path.iloc[0]['success']:
            sorted_path = path.sort_values(by='waypoint_idx')
            final_path_line.set_data(sorted_path['x'], sorted_path['y'])
        else:
            final_path_line.set_data([], [])

        return tree_lc, final_path_line

    # 5. Execute Animation
    ani = FuncAnimation(fig, update, frames=total_frames, interval=playback_speed_ms, blit=True, repeat=False)
    plt.show()
    return ani