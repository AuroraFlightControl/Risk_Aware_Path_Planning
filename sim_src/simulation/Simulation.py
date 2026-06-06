from dataclasses import dataclass
from typing import Optional
import numpy as np
import math

@dataclass
class SimConfig:
    """
    This is a class to store the configuration of the simulation. It is used to initialize the simulation environment and the agents.
    """
    # Simulation Parameters
    dt: float = 0.1         # Time step for the simulation (in seconds)
    max_Time: float = 60.0  # Maximum time for the simulation to run (in seconds)

@dataclass
class SimLog:
    """
    This is a class to store the log of the simulation. It is used to store the data of the simulation for later analysis.
    """
    time: list       # List to store the time of each step in the simulation
    agent_positions: list[np.ndarray] # List to store the positions of the agents at each step in the simulation
    agent_velocities: list[np.ndarray] # List to store the velocities of the agents at each step in the simulation
    plan_ids: list[int] # List to store the plan ids of the agents at each step in the simulation
    success: bool = False # Flag to indicate if the simulation was successful or not
    reason: Optional[str] = None # Reason for failure if the simulation was not successful

    traffic_positions: Optional[list[np.ndarray]] = None # List to store the positions of the traffic at each step in the simulation

class Simulation:
    """
    This is the main class for the simulation. It is used to run the simulation and store the log of the simulation.
    """
    def __init__(self, config: SimConfig):
        self.config = config
        self.log = SimLog(time=[], agent_positions=[], agent_velocities=[], plan_ids=[])

    def run(self):
        """
        This function is used to run the simulation. It updates the positions and velocities of the agents at each step in the simulation and stores the data in the log.
        """
        # Run the simulation loop
        current_time = 0.0
        while current_time < self.config.max_Time:
            # Update the positions and velocities of the agents here

            # Store the data in the log here

            # Check for success or failure conditions here

            # Increment the time
            current_time += self.config.dt
        
        return self.log