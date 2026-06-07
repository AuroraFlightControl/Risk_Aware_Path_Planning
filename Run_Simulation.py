import json, logging, copy, time
from pathlib import Path
from typing import Any

from Run_Functions import load_Agent, load_Config, load_Simulation, load_World, run_Simulation, run_Single
from visualizations.Std_Visual import visualize_enviroment, visualize_trajectory
from sim_src.enviroment.World import World
from sim_src.simulation.Simulation import Simulation, SimConfig, SimLog, Agent

# Remove once agent loading functionality moved to Run_Functions.py
from sim_src.agents.Holonomic_Agent import HolonomicAgent
import numpy as np

"""
This script is used to run the simulation from a given configuration file. 

This script reads the configuration file, initializes the simulation enviroment and the agents, and then runs the simulation.
"""

# Write the name of the configuration file here
CONFIG_FILE = "All_Obstacles.json"
CONFIG_PATH = Path("scenarios") / CONFIG_FILE

logging.basicConfig(level=logging.INFO, format='%(message)s')

def main():
    # Load the configuration file
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    if config is None:
        logging.error(f"Failed to load configuration from {CONFIG_PATH}")
        return
    
    if config["Run_Type"] == "Single":
        logging.info(f"Running Single Simulation with configuration from {CONFIG_PATH}")
        run_Single(config_Data=config)

    if config["Run_Type"] == "Batch":
        logging.info(f"Running Batch Simulation with configuration from {CONFIG_PATH}")
        for run in config["Files"]:
            logging.info(f"Running Simulation {run}")
                # Load the configuration file
            with open((Path("scenarios") / run), 'r') as f:
                config = json.load(f)
            run_Single(config_Data=config)


if __name__ == "__main__":
    main()