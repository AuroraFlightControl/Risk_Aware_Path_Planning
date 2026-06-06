import logging, json, time

# Enviroment Imports

# Agent Imports

# Simulation Imports

# Planning Modules Imports



# Functions to load the enviroment for the simulation and establish the World Object
def load_Obstacles(scene_Data: dict):
    pass

def load_Traffic(scene_Data: dict):
    pass

def load_Points(scene_Data: dict):
    pass

def load_World(scene_Data: dict) -> object:
    pass


# Functions to load agents and planners for the simulation and establish the Agent Objects
def load_Planner(planner_Data: dict) -> object:
    pass

def load_Agent(agent_Data: dict) -> object:
    pass

# Function to load the configuration file
def load_Config(config_Data: dict) -> object:
    Project_Name = config_Data["Project_Name"]
    Project_Description = config_Data["Project_Description"]
    Run_Type = config_Data["Run_Type"]

    print(f"Project Name: {Project_Name}")
    print(f"Project Description: {Project_Description}")
    print(f"Run Type: {Run_Type}")




# Function to load, establish, and run the Simulation Object
def load_Simulation(sim_Data: dict) -> object:
    pass

def run_Simulation(simulation: object) -> None:
    pass