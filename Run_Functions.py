import logging, json, time
import numpy as np
from typing import Optional

# Enviroment Imports
from plan_src.planner_base import Planner
from sim_src.agents.Holonomic_Agent import HolonomicAgent
from sim_src.enviroment.World import World
from sim_src.enviroment.Bounds import Bounds
from sim_src.enviroment.Obstacle import CircularObstacle, RectObstacle_Aligned, PolyObstacle
# Agent Imports

# Simulation Imports
from sim_src.simulation.Simulation import Agent, Simulation, SimConfig, SimLog

# Visualization Imports
from visualizations.Std_Visual import *
from visualizations.Occupy_Grid_Viz import *

# Planning Modules Imports
from plan_src.Occupy_Grid import OccupyGrid


# Functions to load the enviroment for the simulation and establish the World Object
def load_Obstacles(scene_Data: dict):
    logging.info(f"Loading Obstacles from configuration...")
    obstacles = []
    for obs in scene_Data["Obstacles"]:
        if obs["Type"] == "Circle":
            #logging.info(f"Loading Circular Obstacle: Center=({obs['Center_X']}, {obs['Center_Y']}), Radius={obs['Radius']}")
            obstacles.append(CircularObstacle(c=np.array([obs["Center_X"], obs["Center_Y"]]), r=obs["Radius"]))
        elif obs["Type"] == "Rectangle":
            #logging.info(f"Loading Rectangular Obstacle: ({obs['xmin']}, {obs['ymin']}) to ({obs['xmax']}, {obs['ymax']})")
            obstacles.append(RectObstacle_Aligned(xmin=obs["xmin"], xmax=obs["xmax"], ymin=obs["ymin"], ymax=obs["ymax"]))
        elif obs["Type"] == "Polygon":
            #logging.info(f"Loading Polygonal Obstacle with vertices: {list(zip(obs['Vert_x'], obs['Vert_y']))}")
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

def load_World(scene_Data: dict) -> World:

    bounds = Bounds(xmin=scene_Data["Bounds"]["X_Min"], xmax=scene_Data["Bounds"]["X_Max"], ymin=scene_Data["Bounds"]["Y_Min"], ymax=scene_Data["Bounds"]["Y_Max"])
    start = np.array([scene_Data["Start"]["X"], scene_Data["Start"]["Y"]])
    goal = np.array([scene_Data["Goal"]["X"], scene_Data["Goal"]["Y"]])
    obstacles = load_Obstacles(scene_Data)
    traffic = load_Traffic(scene_Data)

    return World(bounds=bounds, start=start, goal=goal, obstacles=obstacles, traffic=traffic)


# Functions to load agents and planners for the simulation and establish the Agent Objects
def load_Planner(planner_Data: dict) -> None: # Planner
    pass

def load_Agent(config: dict, world: World, planner: Optional[Planner] = None) -> Agent:
    if "Agent" not in config:
        logging.error("Agent configuration missing in config file.")
        if planner is not None:
            logging.info(f"Loading default Holonomic Agent with radius {config['Radius']} and planner {planner}")
            return HolonomicAgent(position=world.start, velocity=np.array([0, 0]), radius=config["Radius"], current_trgt=world.goal, planner=planner, world=world)
        else:
            return HolonomicAgent(position=world.start, velocity=np.array([0, 0]), radius=config["Radius"], current_trgt=world.goal, world=world)
    if config["Agent"]["Type"] == "Holonomic" or config["Agent"]["Type"] == "Quad":
        logging.info(f"Loading {config['Agent']['Type']} Agent with radius {config['Agent']['Radius']}")
        return HolonomicAgent(position=world.start, velocity=np.array([0, 0]), radius=config["Agent"]["Radius"], current_trgt=world.goal, planner=planner, world=world)
    else:
        logging.error(f"Unknown agent type: {config['Agent']['Type']}")
        raise ValueError(f"Unknown agent type: {config['Agent']['Type']}")

# Function to load the configuration file
def load_Config(config_Data: dict) -> World:
    Project_Name = config_Data["Project_Name"]
    Project_Description = config_Data["Project_Description"]
    Run_Type = config_Data["Run_Type"]

    logging.info(f"Project Name: {Project_Name}")
    logging.info(f"Project Description: {Project_Description}")
    logging.info(f"Run Type: {Run_Type}")

    world = load_World(config_Data)

    return world


# Function to load, establish, and run the Simulation Object
def load_Simulation(config_Data: dict, world: World, agent: Agent) -> Simulation:
    simConfig = SimConfig(dt=config_Data["Sim_Params"]["Time_Step"], max_Time=config_Data["Sim_Params"]["Max_Time"])
    simulation = Simulation(world=world, agent=agent, config=simConfig)
    return simulation

def run_Simulation(simulation: Simulation) -> SimLog:
    logging.info(f"Running Simulation...")
    simulation.run()
    logging.info(f"Simulation Completed.")
    return simulation.log


def run_Single(config_Data: dict, planner: Optional[Planner] = None):
    world = load_World(config_Data)
    agent = load_Agent(config=config_Data, world=world, planner=planner)
    simulation = load_Simulation(config_Data=config_Data, world=world, agent=agent)
    simulation_log = run_Simulation(simulation)


    occupyGrid = OccupyGrid(world=world, resolution=0.5)
    occupyGrid.build_grid()

    visualize_occupancy_grid(world=world, occ_grid=occupyGrid)



    # Visualize the results
    #visualize_enviroment(world)
    #visualize_trajectory(world, np.array(simulation_log.agent_positions))
    visualize_trajectory_with_time(world, np.array(simulation_log.agent_positions), np.array(simulation_log.time))
    #animate_trajectory_with_time(world, np.array(simulation_log.agent_positions), np.array(simulation_log.time), playback_speed_ms=50)
    


    return simulation_log