import logging, json, time
from pathlib import Path
import numpy as np
from typing import Optional, Any

# Enviroment Imports
from sim_src.enviroment.World import World
from sim_src.enviroment.Bounds import Bounds
from sim_src.enviroment.Obstacle import CircularObstacle, RectObstacle_Aligned, PolyObstacle

# Agent Imports
from sim_src.agents.Traffic_Agent import *
from sim_src.agents.Holonomic_Agent import HolonomicAgent

# Simulation Imports
from sim_src.simulation.Simulation import Agent, Simulation, SimConfig
from sim_src.simulation.logging import *
from sim_src.simulation.MetricsEvaluator import MetricsEvaluator

# Visualization Imports
from visualizations.Std_Visual import *
from visualizations.Occupy_Grid_Viz import *
from visualizations.visualize_planner_search import visualize_planner_search
from visualizations.animate_planner_search import animate_planner_search

# Planning Modules Imports
from plan_src.Occupy_Grid import OccupyGrid
from plan_src.planner_base import Planner
from plan_src.A_Star import AStarGridPlanner


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
    traffic = []
    traffic_id = 0
    for intruder in scene_Data["Intruders"]:
        
        if intruder["Type"] == "Constant_Velocity":
            logging.info("Loading Constant Velocity Traffic Agent")
            x = intruder["Start_X"]
            y = intruder["Start_Y"]
            vx = intruder["Velocity_X"]
            vy = intruder["Velocity_Y"]

            traffic.append(TrafficAgent(agent_id=traffic_id, initial_state=np.array([x, y, vx, vy]), strategy=ConstantVelocityStrategy()))

            traffic_id += 1
            
        else:
            raise ValueError(f"Unknown traffic type {intruder['Type']}")
        
    return traffic

def load_World(scene_Data: dict) -> World:

    bounds = Bounds(xmin=scene_Data["Bounds"]["X_Min"], xmax=scene_Data["Bounds"]["X_Max"], ymin=scene_Data["Bounds"]["Y_Min"], ymax=scene_Data["Bounds"]["Y_Max"])
    start = np.array([scene_Data["Start"]["X"], scene_Data["Start"]["Y"]])
    goal = np.array([scene_Data["Goal"]["X"], scene_Data["Goal"]["Y"]])
    obstacles = load_Obstacles(scene_Data)
    traffic = load_Traffic(scene_Data)

    return World(bounds=bounds, start=start, goal=goal, obstacles=obstacles, traffic=traffic)


# Functions to load agents and planners for the simulation and establish the Agent Objects
def load_Planner(world: World,  save_dir: Path, planner_Config: str = "AStar_Alpha"): # Planner
    filename = f"{planner_Config}.json" if not planner_Config.endswith('.json') else planner_Config
    PLAN_CONFIG = Path("plan_src") / Path("Planner_Config") / filename
    # Load the configuration file
    with open(PLAN_CONFIG, 'r') as f:
        planconfig = json.load(f)

    p_logger = PlannerLogger(log_dir=save_dir, debug_mode=True)
    if planconfig['name'] == 'A* Grid Planner':
        planner = AStarGridPlanner(world=world, resolution=planconfig['resolution'], vehicle_radius=planconfig['vehicle_radius'], connect8=(planconfig["move_type"] == 'Connect8'), logger=p_logger)
        logging.info(f'Loading A* Grid Planner')
    else:
        planner = AStarGridPlanner(world=world, logger=p_logger)
        logging.error(f'Planner of Unkown Type {planconfig['name']}, Loading A* Default')

    return planner

def load_Agent(config: dict, world: World, planner: Optional[Any] = None) -> Agent:
    if "Agent" not in config:
        logging.error("Agent configuration missing in config file.")
        if planner is not None:
            logging.info(f"Loading default Holonomic Agent with radius {config['Radius']} and planner {planner}")
            return HolonomicAgent(position=world.start.copy(), velocity=np.array([0, 0]), radius=config["Radius"], current_trgt=world.goal.copy(), planner=planner, world=world)
        else:
            return HolonomicAgent(position=world.start.copy(), velocity=np.array([0, 0]), radius=config["Radius"], current_trgt=world.goal.copy(), world=world)
    if config["Agent"]["Type"] == "Holonomic" or config["Agent"]["Type"] == "Quad":
        logging.info(f"Loading {config['Agent']['Type']} Agent with radius {config['Agent']['Radius']}")
        return HolonomicAgent(position=world.start.copy(), velocity=np.array([0, 0]), radius=config["Agent"]["Radius"], current_trgt=world.goal.copy(), planner=planner, world=world)
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


def run_Single(config_Data: dict, save_dir: Path, planner_Config: str = "AStar_Alpha", plot_run: bool = True, run_id: str = 'run', gui_mode: bool = False):
    world = load_World(config_Data)
    planner = load_Planner(world=world, planner_Config=planner_Config, save_dir=save_dir)
    agent = load_Agent(config=config_Data, world=world, planner=planner)
    simulation = load_Simulation(config_Data=config_Data, world=world, agent=agent)
    simulation_log = run_Simulation(simulation)

    evaluator = MetricsEvaluator(log=simulation_log, world=world)
    metrics = evaluator.generate_report()

    with open(save_dir / "run_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    simulation_log.export_telemetry(save_dir=str(save_dir), filename_prefix=run_id)

    # Trigger the planner logger export
    if hasattr(planner, 'logger') and planner.logger is not None:
        # Instead of prefixing the file, we just save standard file names because 
        # the folder hierarchy already provides all the context needed.
        planner.logger.save_log(filename_prefix=run_id)

        if planner.logger.debug_mode:
            visualize_planner_search(world=world, log_dir=str(save_dir), filename_prefix=run_id, plan_id=0 )
            animate_planner_search(world=world, log_dir=str(save_dir), filename_prefix=run_id, plan_id=0)

    
    # Visualizations
    if plot_run and not gui_mode:
        visualize_SimLog(world=world, log=simulation_log)
        animate_trajectory_with_traffic(world=world, log=simulation_log)


    


    return simulation_log