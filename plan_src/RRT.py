from typing import List, Optional, Dict, Any
import numpy as np
import math
from dataclasses import dataclass
from random import Random

# Import Planner Resources
from plan_src.planner_base import PlanResult
from sim_src.simulation.logging import PlannerLogger

# Import Enviroment
from sim_src.enviroment.World import World

class Node:
    def __init__(self, point: np.ndarray):
        self.point: np.ndarray = point
        self.parent: Optional["Node"] = None
        self.cost: float = 0.0

@dataclass
class RRT_Planner:
    world: World
    name: str = "RRT Planner"
    max_iter: int = 1000
    step_size: float = 1.0
    vehicle_radius: float = 1.0
    collision_step: float = 1.0
    goal_sample_rate: float = 0.1
    seed: Optional[int] = 0
    SF: float = 5.0
    logger: Optional[PlannerLogger] = None

    def steer(self, from_node: Node, to_point: np.ndarray) -> Node:
        # Steering Function to move from node to another point with a fixed step size
        direction = to_point - from_node.point
        dist = np.linalg.norm(direction)

        if dist <= self.step_size:
            return Node(point=to_point)
        return Node(point=(from_node.point + (direction / dist) * self.step_size))
    
    def nearest(self, nodes: List[Node], point: np.ndarray) -> Node:
        # Return the nearest nod in the tree to a given point
        return min(nodes, key=lambda node: np.linalg.norm(node.point - point))
    
    def reconstruct(self, goal_idx: int, nodes: List[Node]) -> List[np.ndarray]:
        # Reconstruct the path from the goal node back to the start node
        path = []
        node = nodes[goal_idx]

        while node is not None:
            path.append(node.point)
            node = node.parent
        path.reverse()

        return path

    def plan(self, start: Optional[np.ndarray] = None) -> PlanResult:
        # Planning Function
        rng = Random(self.seed)
        
        if start is not None:
            start_node = Node(point=start)
        else:
            start_node = Node(point=self.world.start)

        if not self.world.is_collision_free(point=start_node.point):
            return PlanResult(plan=[], success=False, info={"reason": "Start Position Invalid"})
        if not self.world.is_collision_free(point=self.world.goal):
            return PlanResult(plan=[], success=False, info={"reason": "Goal Position Invalid"})
        
        
        goal_node = Node(point=self.world.goal)
        nodes = [start_node]

        for it in range(self.max_iter):
            if rng.random() < self.goal_sample_rate:
                sample_point = goal_node.point
            else:
                sample_point = np.array([rng.uniform(self.world.bounds.xmin, self.world.bounds.xmax), rng.uniform(self.world.bounds.ymin, self.world.bounds.ymax)])

            nearest_node = self.nearest(nodes, sample_point)
            new_node = self.steer(nearest_node, sample_point)

            if not self.world.is_collision_free(point=new_node.point, vehicle_radius=self.vehicle_radius*self.SF):
                continue

            # Check for collision along the path
            direction = new_node.point - nearest_node.point
                        # Check for collision along the path from nearest_node to new_node
            direction = new_node.point - nearest_node.point
            dist = np.linalg.norm(direction)
            steps = int(math.ceil(dist / self.collision_step))
            collision_free = True
            for i in range(1, steps + 1):
                intermediate_point = nearest_node.point + (direction / dist) * (i * self.collision_step*self.SF)
                if not self.world.is_collision_free(intermediate_point, vehicle_radius=self.vehicle_radius*self.SF):
                    collision_free = False
                    break
            
            if not collision_free:
                continue

            new_node.parent = nearest_node
            new_node.cost = float(nearest_node.cost + dist)
            nodes.append(new_node)

            # Logging Functionality
            if self.logger:
                self.logger.log_node(
                    point=new_node.point,          # The new RRT sample
                    parent_point=nearest_node.point, # Where the branch connects
                    point_type="sampled",             # Use "rewired" if doing RRT* optimizations
                    point_cost=0.0,
                    edge_cost= float(dist),
                    total_cost=new_node.cost
    )

            if np.linalg.norm(new_node.point - goal_node.point) <= self.step_size:
                goal_node.parent = new_node
                goal_node.cost = float(new_node.cost + np.linalg.norm(new_node.point - goal_node.point))
                nodes.append(goal_node)
                path = self.reconstruct(len(nodes) - 1, nodes)
                return PlanResult(plan=path, success=True, info={"iterations": it+1, "path_length": len(path), "cost": goal_node.cost})

        return PlanResult(plan=[], success=False, info={"reason": f"Failed to Find Path in {self.max_iter}"})
