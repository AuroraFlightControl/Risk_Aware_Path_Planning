import logging, json, time
import numpy as np

# Enviroment Imports
from sim_src.enviroment.World import World
from sim_src.enviroment.Bounds import Bounds
from sim_src.enviroment.Obstacle import CircularObstacle, RectObstacle_Aligned, PolyObstacle
# Agent Imports

# Simulation Imports

# Planning Modules Imports



# Functions to load the enviroment for the simulation and establish the World Object
def load_Obstacles(scene_Data: dict):
    logging.info(f"Loading Obstacles from configuration...")
    obstacles = []
    for obs in scene_Data["Obstacles"]:
        if obs["Type"] == "Circle":
            logging.info(f"Loading Circular Obstacle: Center=({obs['Center_X']}, {obs['Center_Y']}), Radius={obs['Radius']}")
            obstacles.append(CircularObstacle(c=np.array([obs["Center_X"], obs["Center_Y"]]), r=obs["Radius"]))
        elif obs["Type"] == "Rectangle":
            logging.info(f"Loading Rectangular Obstacle: ({obs['xmin']}, {obs['ymin']}) to ({obs['xmax']}, {obs['ymax']})")
            obstacles.append(RectObstacle_Aligned(xmin=obs["xmin"], xmax=obs["xmax"], ymin=obs["ymin"], ymax=obs["ymax"]))
        elif obs["Type"] == "Polygon":
            logging.info(f"Loading Polygonal Obstacle with vertices: {list(zip(obs['Vert_x'], obs['Vert_y']))}")
            x = obs['Vert_x']
            y = obs['Vert_y']
            verticies = np.array([x, y]).transpose()
            obstacles.append(PolyObstacle(verts=verticies))
        else:
            logging.error(f"Unknown obstacle type: {obs['Type']}")
            raise ValueError(f"Unknown obstacle type: {obs['Type']}")
    
    return obstacles

def load_Traffic(scene_Data: dict):
    pass

def load_World(scene_Data: dict) -> object:

    bounds = Bounds(xmin=scene_Data["Bounds"]["X_Min"], xmax=scene_Data["Bounds"]["X_Max"], ymin=scene_Data["Bounds"]["Y_Min"], ymax=scene_Data["Bounds"]["Y_Max"])
    start = np.array([scene_Data["Start"]["X"], scene_Data["Start"]["Y"]])
    goal = np.array([scene_Data["Goal"]["X"], scene_Data["Goal"]["Y"]])
    obstacles = load_Obstacles(scene_Data)
    traffic = load_Traffic(scene_Data)

    return World(bounds=bounds, start=start, goal=goal, obstacles=obstacles, Traffic=traffic)


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

    logging.info(f"Project Name: {Project_Name}")
    logging.info(f"Project Description: {Project_Description}")
    logging.info(f"Run Type: {Run_Type}")

    world = load_World(config_Data)

    return world


# Function to load, establish, and run the Simulation Object
def load_Simulation(sim_Data: dict) -> object:
    pass

def run_Simulation(simulation: object) -> None:
    pass