import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, Polygon
from sim_src.enviroment.Obstacle import *

def plot_risk_heatmap(risk_field_data: dict, obstacles: list, gui_mode = False):
    """
    Renders the risk field heatmap and overlays hard obstacle boundaries.
    """
    grid = risk_field_data["risk_array"]
    res = risk_field_data["resolution"]
    origin_x, origin_y = risk_field_data["origin"]
    x_size, y_size = risk_field_data["shape"]
    
    # Calculate extents for matplotlib imshow
    # Matplotlib expects extent as a tuple of four floats: (xmin, xmax, ymin, ymax)
    extent = (
        float(origin_x),
        float(origin_x + x_size * res),
        float(origin_y),
        float(origin_y + y_size * res),
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot the heatmap (Risk ranges from 0.0 to 1.0)
    # cmap='Reds' or 'magma' is standard for risk visualization
    im = ax.imshow(grid.T, extent=extent, origin='lower', cmap='YlOrRd', alpha=0.8, vmin=0.0, vmax=1.0)
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Obstacle Risk (R_obs)', rotation=270, labelpad=15)

    if obstacles is not None:
        for obs in obstacles:
            if isinstance(obs, CircularObstacle):
                
                circle = Circle((obs.c[0], obs.c[1]), obs.radius(), color='darkred', alpha=0.5)
                ax.add_patch(circle)
            elif isinstance(obs, RectObstacle_Aligned):
                rect = Rectangle((obs.xmin, obs.ymin), obs.xmax - obs.xmin, obs.ymax - obs.ymin, color='darkred', alpha=0.5)
                ax.add_patch(rect)
            elif isinstance(obs, PolyObstacle):
                polygon = Polygon(obs.verts, color='darkred', alpha=0.5)
                ax.add_patch(polygon)

    ax.set_title(f"Static Risk Field (Resolution: {res}m)")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.grid(False)
    
    plt.tight_layout()
    if not gui_mode:
        plt.show()