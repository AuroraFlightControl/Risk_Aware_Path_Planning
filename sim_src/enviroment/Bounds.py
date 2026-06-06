from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Bounds:
    xmin: float = 0
    xmax: float = 100
    ymin: float = 0
    ymax: float = 100
    zmin: Optional[float] = None
    zmax: Optional[float] = None

    def in_bounds(self, point: np.ndarray) -> bool:
        if self.zmin == None and self.zmax == None:
            return (self.xmin <= point[0] <= self.xmax and self.ymin <= point[1] <= self.ymax)
        else:
            return (self.xmin <= point[0] <= self.xmax and self.ymin <= point[1] <= self.ymax and self.zmin <= point[2] <= self.zmax)