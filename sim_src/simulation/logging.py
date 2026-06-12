from dataclasses import dataclass
from typing import Optional
import numpy as np

from sim_src.enviroment.World import World
from plan_src.planner_base import PlanResult


@dataclass
class SimLog:
    """
    This is a class to store the log of the simulation. It is used to store the data of the simulation for later analysis.
    """
    time: list[float]       # List to store the time of each step in the simulation
    agent_positions: list[np.ndarray] # List to store the positions of the agents at each step in the simulation
    agent_velocities: list[np.ndarray] # List to store the velocities of the agents at each step in the simulation
    plan_ids: list[int] # List to store the plan ids of the agents at each step in the simulation
    success: bool = False # Flag to indicate if the simulation was successful or not
    reason: Optional[str] = None # Reason for failure if the simulation was not successful

    traffic_positions: Optional[dict[int, list[np.ndarray]]] = None # List to store the positions of the traffic at each step in the simulation mapped to traffic ID
    replan_records: Optional[list[dict]] = None # Stores {'time': t, 'computation_time': tc, 'success': bool}

class MetricsEvaluator:
    """Computes the Statement of Work evaluation metrics post-run"""
    def __init__(self, log: SimLog, world: World):
        self.log = log
        self.world = world

    def calculate_safe_seperation_distance(self) -> float:
        '''Finds the absolute minimum Euclidian Distance to any traffic agent durring the run'''
        # TODO: Implement Step-By-Step vector norm matching the ownership agent to traffic
        safe_sep_dis = 0.0 # Placeholder.
        return safe_sep_dis
    
    def evaluate_constraint_violations(self, min_allowable_sep: float) -> dict:
        '''Checks for breaches in static obstacle or traffic safety distances'''
        # TODO: Build out the checking logic here

        static_violation = False
        traffic_violation = True

        violations = {
            'static_violation': static_violation,
            'traffic_violation': traffic_violation
        }
        return violations
    
    def generate_safety_margin_over_time(self) -> list[float]:
        """Returns time-series data of clearance values for plotting."""
        # TODO: Calculate the satey margin and record.

        safety_margin_TH = [0.0, 0.0] # Placeholder

        return safety_margin_TH
    
    def compile_Metrics(self) -> None:
        pass


    class PlannerLogger:
        def __init__(self, log_dir: str, plan_id: int = 0, debug_mode = False):
            self.log_dir = log_dir
            self.plan_id = plan_id
            self.debug_mode = debug_mode
            self.sampled_points = []
            self.plans = []

        def log_node(self, point: np.ndarray, point_type: str, point_cost: float, edge_cost: float, total_cost: float):
            # Point Type: "sampled", "rewire", "goal", etc
            if not self.debug_mode:
                return
            
            self.sampled_points.append({
                "plan_id": self.plan_id,
                "x": float(point[0]),
                "y": float(point[1]),
                "type": point_type,
                "point_cost": float(point_cost),
                "edge_cost": float(edge_cost),
                "total_path_cost": float(total_cost)
            })

        def log_result(self, Timestamp: float, PlannerResult: PlanResult):
            pass

        def save_log(self, filename: str):
            # TODO: Add Standard Plan log for replanning. Log and save the final path with the plan id and the computation time
            
            
            if self.debug_mode:
                # TODO: Craft the CSV file saving logic for the logged sampled points 
                pass
            
            