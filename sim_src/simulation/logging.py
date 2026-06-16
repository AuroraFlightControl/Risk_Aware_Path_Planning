from dataclasses import dataclass
from typing import Optional
import numpy as np
from pathlib import Path

from sim_src.enviroment.World import World
from plan_src.planner_base import PlanResult

import os, csv



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

    def export_telemetry(self, save_dir: str, filename_prefix: str = "run_0"):
        """
        Flattens the SimLog time-series arrays into a relational CSV format.
        Optimal for Pandas ingestion and multi-agent tracking.
        """
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f"{filename_prefix}_telemetry.csv")
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write the relational headers
            writer.writerow(["time", "agent_type", "agent_id", "x", "y", "vx", "vy"])
            
            for i in range(len(self.time)):
                t = self.time[i]
                
                # 1. Write Ownship Data
                if i < len(self.agent_positions) and i < len(self.agent_velocities):
                    o_pos = self.agent_positions[i]
                    o_vel = self.agent_velocities[i]
                    # Log ownship as ID 0
                    writer.writerow([f"{t:.3f}", "ownship", 0, o_pos[0], o_pos[1], o_vel[0], o_vel[1]])
                
                # 2. Write Traffic Data
                if self.traffic_positions is not None:
                    for traffic_id, t_pos_list in self.traffic_positions.items():
                        if i < len(t_pos_list):
                            t_pos = t_pos_list[i].flatten()
                            
                            # Note: If you eventually track traffic velocities in SimLog, 
                            # you can extract and log them here. For now, leaving them blank.
                            writer.writerow([f"{t:.3f}", "traffic", traffic_id, t_pos[0], t_pos[1], "", ""])




class PlannerLogger:
    def __init__(self, log_dir: Path, plan_id: int = 0, debug_mode: bool = False):
        self.log_dir = log_dir
        self.plan_id = plan_id
        self.debug_mode = debug_mode
        self.sampled_points = []
        self.plans = []

    def log_node(self, point: np.ndarray, point_type: str, point_cost: float, edge_cost: float, total_cost: float, parent_point: Optional[np.ndarray]):
        # Point Type: "sampled", "rewired", "goal", etc.
        if not self.debug_mode:
            return
        
        self.sampled_points.append({
            "plan_id": self.plan_id,
            "x": float(point[0]),
            "y": float(point[1]),
            "parent_x": float(parent_point[0]) if parent_point is not None else float(point[0]),
            "parent_y": float(parent_point[1]) if parent_point is not None else float(point[1]),
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
                            plan["computation_time"], idx, pt[0], pt[1], ""
                        ])
                else:
                    # Log the failure event without waypoints
                    writer.writerow([
                        plan["plan_id"], plan["timestamp"], plan["success"], 
                        plan["computation_time"], -1, "", "", ""
                    ])

        # 2. Save the Debug Sampled Points (Only if debug_mode was True)
        if self.debug_mode and len(self.sampled_points) > 0:
            debug_filepath = os.path.join(self.log_dir, f"{filename_prefix}_sampled_nodes.csv")
            with open(debug_filepath, 'w', newline='') as f:
                # Extract headers dynamically from the first dictionary keys
                fieldnames = ["plan_id", "x", "y", "parent_x", "parent_y", "type", "point_cost", "edge_cost", "total_path_cost"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                
                writer.writeheader()
                writer.writerows(self.sampled_points)
            