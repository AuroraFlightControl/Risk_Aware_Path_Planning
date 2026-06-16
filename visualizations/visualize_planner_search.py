import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sim_src.enviroment.World import World
from visualizations.Std_Visual import visualize_world

def visualize_planner_search(world: World, log_dir: str, filename_prefix: str, plan_id: int = 0):
    """
    Renders the occupancy grid/world, the algorithm's search tree, and the final path.
    """
    # 1. Setup the Base Environment
    visualize_world(world)
    ax = plt.gca()

    # 2. Load the Logged Data
    try:
        nodes_df = pd.read_csv(f"{log_dir}/{filename_prefix}_sampled_nodes.csv")
        paths_df = pd.read_csv(f"{log_dir}/{filename_prefix}_paths.csv")
    except FileNotFoundError:
        print("Log files not found. Ensure the planner was run in debug_mode.")
        return

    # Filter data to the specific replanning event (plan_id)
    nodes = nodes_df[nodes_df['plan_id'] == plan_id]
    path = paths_df[paths_df['plan_id'] == plan_id]

    # 3. Plot the Search Tree / Explored Nodes
    if not nodes.empty:
        # Separate standard samples from optimized/rewired edges (for RRT*)
        sampled = nodes[nodes['type'] == 'sampled']
        rewired = nodes[nodes['type'] == 'rewired']

        # Format line segments for Matplotlib LineCollection: [[(x0, y0), (x1, y1)], ...]
        def build_segments(df):
            return [((row['parent_x'], row['parent_y']), (row['x'], row['y'])) for _, row in df.iterrows()]

        # Draw standard exploration (Light Gray)
        if not sampled.empty:
            lc_sampled = LineCollection(build_segments(sampled), colors='lightgray', linewidths=0.5, alpha=0.7, zorder=1)
            ax.add_collection(lc_sampled)

        # Draw RRT* Rewiring Optimizations (Light Green)
        if not rewired.empty:
            lc_rewired = LineCollection(build_segments(rewired), colors='lightgreen', linewidths=1.0, alpha=0.9, zorder=2)
            ax.add_collection(lc_rewired)

    # 4. Plot the Final Solution Path (Bold Red)
    if not path.empty and path.iloc[0]['success']:
        # Sort by waypoint_idx to ensure the line draws in the correct order
        path = path.sort_values(by='waypoint_idx')
        ax.plot(path['x'], path['y'], color='red', linewidth=2.5, marker='.', markersize=8, label='Final Path', zorder=3)

    plt.title(f"Planner Search Space (Plan ID: {plan_id})")
    # Move legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()