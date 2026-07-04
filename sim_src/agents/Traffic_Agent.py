from typing import Protocol
import numpy as np


class TrafficModelStrategy(Protocol):
    def propogate(self, state: np.ndarray, dt: float) -> np.ndarray:
        ...

class ConstantVelocityStrategy:
    '''Implements Eq. 8 and 9 from Extended Abstract: Constant Linear Velocity State Transition'''
    def propogate(self, state: np.ndarray, dt: float) -> np.ndarray:
        # State = [x, y, vx, vy]^T
        A = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        return A @ state
    
class TrafficAgent:
    def __init__(self, agent_id: int, initial_state: np.ndarray, strategy: TrafficModelStrategy):
        self.id = agent_id
        self.state = initial_state
        self.strategy = strategy

        # For Later Use
        self.covariance = np.eye(4) * 0.1 

    def update(self, dt: float) -> None:
        '''Propogates the true state of the intruder forward in time'''
        self.state = self.strategy.propogate(state=self.state, dt=dt)

    @property
    def position(self) -> np.ndarray:
        # Grabs index 0 and 1: [x, y]
        return self.state[0:2]
    
    @property
    def velocity(self) -> np.ndarray:
        # Grabs index 2 and 3: [vx, vy]
        return self.state[2:4]
    
    def distance(self, point: np.ndarray) -> float:
        '''Calculates the Euclidean distance from the agent to a given point'''
        return float(np.linalg.norm(self.position - point))
