import math
from typing import List, Dict, Tuple, Optional, Protocol
from dataclasses import dataclass, field
import numpy as np

class Obstacle(Protocol):
    """Obstacle interface. Obstacles are static in the world. Dynamic Obstacles are instances of agents"""
    def position(self) -> np.ndarray:
        ...
    def radius(self) -> float:
        ...
    def contains(self, point: np.ndarray, inflate: float = 0.0) -> bool:
        ...
    def distance(self, point: np.ndarray) -> float:
        ...

@dataclass
class CircularObstacle:
    c: np.ndarray
    r: float = 1.0

    def position(self) -> np.ndarray:
        return self.c
    
    def radius(self) -> float:
        return self.r
    
    def contains(self, point: np.ndarray, inflate: float = 0.0) -> bool:
        r_eff = self.r + max(0.0, inflate)
        dist_sq = self.distance(point=point)**2
        return dist_sq <= r_eff**2

    def distance(self, point: np.ndarray) -> float:
        dist_sq = 0
        for i in range(0, len(point)):
            dist_sq += (point[i] - self.c[i])**2
        return math.sqrt(dist_sq)
    
@dataclass
class RectObstacle_Aligned:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def position(self) -> np.ndarray:
        return np.array([(self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0], dtype=float)
    
    def radius(self) -> float:
        hx = (self.xmax - self.xmin) / 2.0
        hy = (self.ymax - self.ymin) / 2.0
        return math.hypot(hx, hy)
    
    def contains(self, point: np.ndarray, inflate: float=0.0) -> bool:
        return (self.xmin - max(0.0, inflate) <= point[0] <= self.xmax + max(0.0, inflate) and self.ymin - max(0.0, inflate) <= point[1] <= self.ymax + max(0.0, inflate))

    def distance(self, point: np.ndarray) -> float:
        dx = max(self.xmin - point[0], point[0] - self.xmax)
        dy = max(self.ymin - point[1], point[1] - self.ymax)
        return math.hypot(dx, dy)

@dataclass
class PolyObstacle:
    verts: np.ndarray

    def __post_init__(self):
        self.verts = np.array(self.verts, dtype=float)
        if self.verts.ndim != 2 or self.verts.shape[1] != 2 or len(self.verts) < 3:
            raise ValueError("Verticies Mus be an (N,2) array with N > 3")
    
    def position(self) -> np.ndarray:
        return np.mean(self.verts, axis=0)
    
    def radius(self) -> float:
        c = self.position()
        return float(np.max(np.linalg.norm(self.verts - c, axis=1)))
    
    def contains(self, point: np.ndarray, inflate: float = 0.0) -> bool:
        eff_inflate = max(0.0, inflate)
        return self.distance(point) <= eff_inflate

    def distance(self, point: np.ndarray) -> float:
        p = np.asarray(point, dtype=float)
        if _point_in_poly(p, self.verts):
            return 0.0
        dmin = float("inf")
        n = len(self.verts)
        for i in range(n):
            a = self.verts[i]
            b = self.verts[(i + 1) % n]
            dmin = min(dmin, _dist_point_to_segment(p, a, b))
        return dmin


def _point_in_poly(point: np.ndarray, verts: np.ndarray) -> bool:
    x, y = float(point[0]) , float(point[1])
    inside = False
    n = len(verts)
    for i in range(n):
        x1, y1 = float(verts[i][0]), float(verts[i][1])
        x2, y2 = float(verts[(i+1) % n][0]), float(verts[(i+1) % n][1])
        intersects = ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1)
        if intersects:
            inside = not inside
    return inside

def _dist_point_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    # Euclidean distance from point p to segment ab
    ap = p - a
    ab = b - a
    denom = float(ab @ ab)
    if denom <= 1e-12:
        return float(np.linalg.norm(ap))
    t = float((ap @ ab) / denom)
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))

