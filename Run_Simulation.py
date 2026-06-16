import json, logging, copy, time
from pathlib import Path
from typing import Any

from Run_Functions import load_Agent, load_Config, load_Simulation, load_World, run_Simulation, run_Single
from visualizations.Std_Visual import visualize_enviroment, visualize_trajectory
from sim_src.enviroment.World import World
from sim_src.simulation.Simulation import Simulation, SimConfig, SimLog, Agent

from sim_src.simulation.RunSessionManager import RunSessionManager

"""
This script is used to run the simulation from a given configuration file. 

This script reads the configuration file, initializes the simulation enviroment and the agents, and then runs the simulation.
"""

# Write the name of the configuration file here
CONFIG_FILE = "Alpha_1.json"
CONFIG_PATH = Path("scenarios") / CONFIG_FILE

logging.basicConfig(level=logging.INFO, format='%(message)s')

def main():
    # Load the configuration file
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)

    if config is None:
        logging.error(f"Failed to load configuration from {CONFIG_PATH}")
        return
    
    session_manager = RunSessionManager(root_config=config)

    if config["Run_Type"] == "Single":
        logging.info(f"Running Single Simulation with configuration from {CONFIG_PATH}")
        save_directory = session_manager.get_save_dir()
        run_Single(config_Data=config, save_dir=save_directory)

    if config["Run_Type"] == "Batch":
        logging.info(f"Running Batch Simulation with configuration from {CONFIG_PATH}")
        for run in config["Files"]:
            logging.info(f"Running Simulation {run}")
                # Load the configuration file
            with open((Path("scenarios") / run), 'r') as f:
                config = json.load(f)

            save_directory = session_manager.get_save_dir()
            run_Single(config_Data=config, save_dir=save_directory)

    if config["Run_Type"] == "Comparison":
        logging.info(f"Running Comparison SImulation from {CONFIG_PATH}")
        


if __name__ == "__main__":
    main()