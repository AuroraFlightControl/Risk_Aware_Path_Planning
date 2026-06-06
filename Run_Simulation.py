import json, logging, copy, time
from pathlib import Path
from typing import Any

from Run_Functions import load_Config, load_Simulation, load_World, run_Simulation
from visualizations.Std_Visual import visualize_world

"""
This script is used to run the simulation from a given configuration file. 

This script reads the configuration file, initializes the simulation enviroment and the agents, and then runs the simulation.
"""

# Write the name of the configuration file here
CONFIG_FILE = "config.json"
CONFIG_PATH = Path("scenarios") / CONFIG_FILE

logging.basicConfig(level=logging.INFO, format='%(message)s')

def main():
    # Load the configuration file
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    # Load the configuration
    world: Any = load_Config(config)

    visualize_world(world=world)
    

if __name__ == "__main__":
    main()