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


import os
import csv
import numpy as np
from plan_src.planner_base import PlanResult

class PlannerLogger:
    def __init__(self, log_dir: str, plan_id: int = 0, debug_mode: bool = False):
        self.log_dir = log_dir
        self.plan_id = plan_id
        self.debug_mode = debug_mode
        self.sampled_points = []
        self.plans = []

    def log_node(self, point: np.ndarray, point_type: str, point_cost: float, edge_cost: float, total_cost: float):
        # Point Type: "sampled", "rewired", "goal", etc.
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

    def log_result(self, timestamp: float, result: PlanResult, computation_time: float):
        """
        Logs the final result of a planning cycle. 
        computation_time is added to capture the evaluation metric.
        """
        self.plans.append({
            "plan_id": self.plan_id,
            "timestamp": timestamp,
            "success": result.success,
            "info": result.info,
            "computation_time": computation_time,
            # Convert numpy arrays to standard lists for easier flat-file saving
            "path": [pt.tolist() for pt in result.plan] if result.plan is not None else []
        })
        
        # Increment plan_id so the next dynamic replan trigger is uniquely tracked
        self.plan_id += 1

    def save_log(self, filename_prefix: str = "run_0"):
        """Saves the standard plan metrics and the debug sampled points to CSVs."""
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 1. Save the Standard Plan Log (The final paths and computation metrics)
        paths_filepath = os.path.join(self.log_dir, f"{filename_prefix}_paths.csv")
        with open(paths_filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(["plan_id", "timestamp", "success", "computation_time", "waypoint_idx", "x", "y", "reason"])
            
            for plan in self.plans:
                if plan["success"] and len(plan["path"]) > 0:
                    for idx, pt in enumerate(plan["path"]):
                        writer.writerow([
                            plan["plan_id"], plan["timestamp"], plan["success"], 
                            plan["computation_time"], idx, pt[0], pt[1], plan["reason"]
                        ])
                else:
                    # Log the failure event without waypoints
                    writer.writerow([
                        plan["plan_id"], plan["timestamp"], plan["success"], 
                        plan["computation_time"], -1, "", "", plan["reason"]
                    ])

        # 2. Save the Debug Sampled Points (Only if debug_mode was True)
        if self.debug_mode and len(self.sampled_points) > 0:
            debug_filepath = os.path.join(self.log_dir, f"{filename_prefix}_sampled_nodes.csv")
            with open(debug_filepath, 'w', newline='') as f:
                # Extract headers dynamically from the first dictionary keys
                fieldnames = ["plan_id", "x", "y", "type", "point_cost", "edge_cost", "total_path_cost"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                writer.writeheader()
                writer.writerows(self.sampled_points)
            