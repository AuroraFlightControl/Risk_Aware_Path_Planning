import numpy as np
from sim_src.enviroment.World import World
from sim_src.simulation.logging import SimLog

class MetricsEvaluator:
    """Computes the evaluation metrics post-run for batch and comparative analysis."""
    def __init__(self, log: SimLog, world: World):
        self.log = log
        self.world = world

    def calculate_safe_separation_distance(self) -> float:
        """Finds the absolute minimum Euclidean distance to any traffic agent during the run."""
        own_pos = np.array(self.log.agent_positions)
        min_sep = float('inf')

        # Check if traffic data exists and contains arrays
        traffic_positions = getattr(self.log, 'traffic_positions', None)
        if traffic_positions is not None:
            for traffic_id, t_pos_list in traffic_positions.items():
                if not t_pos_list:
                    continue
                
                # Reshape to avoid dimensionality errors and ensure 2D structure
                t_pos = np.array(t_pos_list).reshape(-1, 2)
                
                # Truncate to the minimum length in case the log stopped tracking mid-frame
                min_len = min(len(own_pos), len(t_pos))
                if min_len == 0:
                    continue
                
                # Vectorized distance calculation across all time steps simultaneously
                dists = np.linalg.norm(own_pos[:min_len] - t_pos[:min_len], axis=1)
                
                # Track the global minimum across all traffic agents
                min_sep = min(min_sep, np.min(dists))

        return float(min_sep)
    
    def evaluate_constraint_violations(self, min_allowable_sep: float, vehicle_radius: float = 1.0) -> dict:
        """Checks for breaches in static obstacle or traffic safety distances."""
        traffic_violation = False
        static_violation = False

        # 1. Traffic Violation Check
        if self.calculate_safe_separation_distance() < min_allowable_sep:
            traffic_violation = True

        # 2. Static Violation Check
        own_pos = np.array(self.log.agent_positions)
        if self.world.obstacles is not None:
            for obs in self.world.obstacles:
                # Iterate through ownship positions to see if the obstacle's boundary was breached
                for pos in own_pos:
                    if obs.contains(pos, inflate=vehicle_radius):
                        static_violation = True
                        break  # Break inner loop if violation found
                if static_violation:
                    break      # Break outer loop if violation found

        return {
            'static_violation': static_violation,
            'traffic_violation': traffic_violation
        }
    
    def generate_safety_margin_over_time(self) -> list[float]:
        """Returns time-series data of clearance values to the nearest hazard for plotting."""
        own_pos = np.array(self.log.agent_positions)
        margins = []

        for i, pos in enumerate(own_pos):
            step_min_margin = float('inf')

            # Calculate distance to nearest static obstacle at this time step
            if self.world.obstacles is not None:
                for obs in self.world.obstacles:
                    step_min_margin = min(step_min_margin, obs.distance(pos))

            # Calculate distance to nearest traffic agent at this time step
            traffic_positions = getattr(self.log, 'traffic_positions', None)
            if traffic_positions is not None:
                for traffic_id, t_pos_list in traffic_positions.items():
                    if i < len(t_pos_list):
                        t_pos = np.asarray(t_pos_list[i]).reshape(-1)
                        step_min_margin = min(step_min_margin, np.linalg.norm(pos - t_pos))

            margins.append(float(step_min_margin))

        return margins
    
    def generate_report(self, vehicle_radius: float = 1.0) -> dict:
        """
        Compiles the primary metrics: Mission success rate, constraint violations, 
        safe separation, and computational cost per replanning cycle.
        """
        # Calculate computational cost metrics
        avg_replan_time = 0.0
        replan_count = 0
        
        records = getattr(self.log, 'replan_records', []) or []
        if records:
            times = [r['computation_time'] for r in records]
            if times:
                avg_replan_time = np.mean(times)
                replan_count = len(times)

        # Standard traffic separation buffer (e.g., 2x vehicle radius)
        min_allowable_sep = vehicle_radius * 2.0
        
        min_traffic_sep = self.calculate_safe_separation_distance()
        violations = self.evaluate_constraint_violations(min_allowable_sep, vehicle_radius)

        return {
            "mission_success": self.log.success,
            "failure_reason": self.log.reason,
            "min_safe_separation_ft": min_traffic_sep if min_traffic_sep != float('inf') else None,
            "static_violation": violations['static_violation'],
            "traffic_violation": violations['traffic_violation'],
            "avg_replan_computation_sec": float(avg_replan_time),
            "total_replans": replan_count
        }