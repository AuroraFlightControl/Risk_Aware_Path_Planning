import logging, json, time
import numpy as np

# Enviroment Imports
from sim_src.enviroment import World
from sim_src.enviroment.Obstacle import CircularObstacle, RectObstacle_Aligned, PolyObstacle
# Agent Imports

# Simulation Imports

# Planning Modules Imports



# Functions to load the enviroment for the simulation and establish the World Object
def load_Obstacles(scene_Data: dict):
    obstacles = []
    for obs in scene_Data["Obstacles"]:
        if obs["Type"] == "Circle":
            obstacles.append(CircularObstacle(c=np.array([obs["Center_X"], obs["Center_Y"]]), r=obs["Radius"]))
        elif obs["Type"] == "Rectangle":
            obstacles.append(RectObstacle_Aligned(xmin=obs["xmin"], xmax=obs["xmax"], ymin=obs["ymin"], ymax=obs["ymax"]))
        elif obs["Type"] == "Polygon":
            x = obs['Vert_x']
            y = obs['Vert_y']
            verticies = np.array([x, y]).transpose()
            obstacles.append(PolyObstacle(verts=verticies))
        else:
            raise ValueError(f"Unknown obstacle type: {obs['Type']}")

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

    Obstacles = load_Obstacles(config_Data)


# Function to load, establish, and run the Simulation Object
def load_Simulation(sim_Data: dict) -> object:
    pass

def run_Simulation(simulation: object) -> None:
    pass