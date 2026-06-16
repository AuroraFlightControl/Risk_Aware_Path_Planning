from typing import List, Optional
import numpy as np
import math, heapq, time
from dataclasses import dataclass

from sim_src.enviroment.World import World
from plan_src.planner_base import PlanResult
from plan_src.Occupy_Grid import OccupyGrid
from sim_src.simulation.logging import PlannerLogger

@dataclass
class AStarGridPlanner:
    world: World
    name: str = "A* Grid Planner"
    resolution: float = 0.5
    vehicle_radius: float = 1.0
    connect8: bool = True

    logger: Optional[PlannerLogger] = None

    def search(self):
        ''' A* Search Algorithm implementation '''

        # Establish the move set for an expansion
        moves4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        moves8 = moves4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        moves = moves8 if self.connect8 else moves4

        # Setup and open the heap for searching
        open_heap: List[tuple[float, tuple[int, int]]] = []
        heapq.heappush(open_heap, (0.0, self.start_idx))

        came_from: dict[tuple[int, int], Optional[tuple[int, int]]] = {self.start_idx: None}
        g_score: dict[tuple[int, int], float] = {self.start_idx: 0.0}

        expansions = 0

        while open_heap:
            _, cur = heapq.heappop(open_heap)
            expansions += 1

            # Check if the current location is the gaol
            if cur == self.goal_idx:
                # Reconstruct Path
                path = []

                node = cur
                
                while node is not None:
                    path.append(self.Occupancy_Grid.to_point(node))
                    node = came_from[node]
                path.reverse()
                return PlanResult(plan=path, success=True, info={"expansions": expansions, "cost": g_score[cur]})

            for di, dj in moves:
                nxt = (cur[0] + di, cur[1] + dj)

                if not self.Occupancy_Grid.in_bounds(nxt) or self.Occupancy_Grid.is_occupied(nxt):
                    continue

                step_cost = self.resolution * (math.sqrt(2.0) if (di != 0 and dj != 0) else 1.0)
                ten_gscore = g_score[cur] + step_cost

                if nxt not in g_score or ten_gscore < g_score[nxt]:
                    came_from[nxt] = cur
                    g_score[nxt] = ten_gscore
                    f_score = ten_gscore + self.heuristic(a=nxt, b=self.goal_idx)
                    heapq.heappush(open_heap, (f_score, nxt))

                    if self.logger:
                        nxt_point = self.Occupancy_Grid.to_point(nxt)
                        cur_point = self.Occupancy_Grid.to_point(cur)
                        self.logger.log_node(
                            point=nxt_point,
                            point_type="sampled",
                            point_cost=step_cost,
                            edge_cost=step_cost,
                            total_cost=ten_gscore,
                            parent_point=cur_point
                        )

        return PlanResult(plan=[], success=False, info={"reason": "No Path Found", "expansions": expansions})

    def heuristic(self, a: tuple[int, int], b: tuple[int, int]):
        return math.hypot(a[0] - b[0], a[1] - b[1])*self.resolution

    def plan(self, world: World, start: np.ndarray):
        self.Occupancy_Grid = OccupyGrid(world=world, resolution=self.resolution)
        self.Occupancy_Grid.build_grid()

        self. start_idx = self.Occupancy_Grid.to_idx(point=start)
        if not self.Occupancy_Grid.in_bounds(self.start_idx):
            return PlanResult(plan=[], success=False, info={"reason": "Start Position Out of Bounds"})
        if self.Occupancy_Grid.is_occupied(self.start_idx):
            return PlanResult(plan=[], success=False, info={"reason": "Start Position is Occupied"})

        self.goal_idx = self.Occupancy_Grid.to_idx(point=world.goal)
        if not self.Occupancy_Grid.in_bounds(self.goal_idx):
            return PlanResult(plan=[], success=False, info={"reason": "Goal Position Out of Bounds"})
        if self.Occupancy_Grid.is_occupied(self.goal_idx):
            return PlanResult(plan=[], success=False, info={"reason": "Goal Position is Occupied"})
        
        start_time = time.time()

        result = self.search()

        comp_time = time.time() - start_time

        if self.logger:
            self.logger.log_result(timestamp=0.0, result=result, computation_time=comp_time)
        
        return result



        