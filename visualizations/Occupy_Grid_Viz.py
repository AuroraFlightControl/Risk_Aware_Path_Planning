import numpy as np
import matplotlib.pyplot as plt
from plan_src.Occupy_Grid import OccupyGrid
from sim_src.enviroment.World import World
from visualizations.Std_Visual import visualize_world

def Display_Occupy_Grid(world: World, occupyGrid: OccupyGrid):
    fig, ax = plt.subplots()
    
    # Set the limits of the plot based on world bounds
    ax.set_xlim(world.bounds.xmin, world.bounds.xmax)
    ax.set_ylim(world.bounds.ymin, world.bounds.ymax)
    ax.set_aspect('equal', adjustable='box')

    plt.imshow(occupyGrid.grid.T, origin='lower', cmap='Greys')
    plt.title("Occupancy Grid")
    plt.xlabel("Position East")
    plt.ylabel("Position North")
        
    plt.show()

def visualize_occupancy_grid(world: World, occ_grid: OccupyGrid):
    # 1. Setup the static background 
    visualize_world(world)
    ax = plt.gca()

    # 2. Plot the grid
    # - We transpose the grid (.T) to swap X and Y back to image format (Row, Col)
    # - origin='lower' flips the Y-axis so 0 is at the bottom
    # - extent stretches the array pixels to match the physical world bounds
    ax.imshow(
        occ_grid.grid.T, 
        cmap='Reds',           # Red shows collisions well
        alpha=0.3,             # Make it semi-transparent so we can see obstacles underneath
        origin='lower', 
        extent=(occ_grid.x_min, occ_grid.x_max, occ_grid.y_min, occ_grid.y_max)
    )

    plt.title(f'Occupancy Grid (Resolution: {occ_grid.resolution})')
    plt.show()