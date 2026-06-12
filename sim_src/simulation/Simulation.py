from dataclasses import dataclass
from typing import Optional, Protocol
import numpy as np
import math, logging


from sim_src.enviroment.World import World
from sim_src.simulation.logging import *
from sim_src.agents.Traffic_Agent import TrafficAgent

class Agent(Protocol):
    position: np.ndarray
    velocity: np.ndarray
    radius: float


    def step(self, dt: float) -> None:
        ...

@dataclass
class SimConfig:
    """
    This is a class to store the configuration of the simulation. It is used to initialize the simulation environment and the agents.
    """
    # Simulation Parameters
    dt: float = 0.1         # Time step for the simulation (in seconds)
    max_Time: float = 60.0  # Maximum time for the simulation to run (in seconds)

'''
@dataclass
class SimLog:
    """
    This is a class to store the log of the simulation. It is used to store the data of the simulation for later analysis.
    """
    time: list[float]       # List to store the time of each step in the simulation
    agent_positions: list[np.ndarray] # List to store the positions of the agents at each step in the simulation
    agent_velocities: list[np.ndarray] # List to store the velocities of the agents at each step in the simulation
    plan_ids: list[int] # List to store the plan ids of the agents at each step in the simulation

    traffic_telemetry: dict[int, list[np.ndarray]]

    success: bool = False # Flag to indicate if the simulation was successful or not
    reason: Optional[str] = None # Reason for failure if the simulation was not successful
'''
    

class Simulation:
    """
    This is the main class for the simulation. It is used to run the simulation and store the log of the simulation.
    """
    def __init__(self, config: SimConfig, world: World, agent: Agent):
        self.config = config
        self.world = world
        self.agent = agent
        self.log = SimLog(time=[], agent_positions=[], agent_velocities=[], plan_ids=[])

    def run(self):
        """
        This function is used to run the simulation. It updates the positions and velocities of the agents at each step in the simulation and stores the data in the log.
        """
        # Run the simulation loop
        current_time = 0.0
        while current_time < self.config.max_Time:
 
            # Update the positions and velocities of the agents here
            self.agent.step(self.config.dt)

            # Check for collisions and other failure conditions
            if not self.world.is_collision_free(self.agent.position, vehicle_radius=self.agent.radius, time=current_time):
                self.log.success = False
                self.log.reason = "Collision with obstacle or traffic"
                logging.info(f"Simulation failed at time {current_time:.2f} seconds: {self.log.reason}")
                break

            # Check if the agent has reached the goal
            if np.linalg.norm(self.agent.position - self.world.goal) < self.agent.radius:
                self.log.success = True
                logging.info(f"Simulation successful! Agent reached the goal at time {current_time:.2f} seconds.")
                break

            # Store the data in the log here
            self.log.time.append(current_time)
            self.log.agent_positions.append(self.agent.position.copy())
            self.log.agent_velocities.append(self.agent.velocity.copy())

            # Increment the time
            current_time += self.config.dt
            if current_time >= self.config.max_Time:
                self.log.success = False
                self.log.reason = "Maximum simulation time reached without reaching the goal"
                logging.info(f"Simulation failed at time {current_time:.2f} seconds: {self.log.reason}")
                break
        
        return self.log