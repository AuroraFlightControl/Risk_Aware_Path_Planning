import numpy as np
from typing import Protocol


class PredictionModel(Protocol):
    def predict(self, state: np.ndarray, dt: float) -> np.ndarray:
        ...

    def Collision_Risk(self, traf_state: np.ndarray, agent_state: np.ndarray, d_safe: float) -> float:
        ...
    
class ConstantVelocityModel:
    '''Implements Eq. 8 and 9 from Extended Abstract: Constant Linear Velocity State Transition'''
    def predict(self, state: np.ndarray, dt: float) -> np.ndarray:
        # State = [x, y, vx, vy]^T
        A = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        return A @ state
    

    def Collision_Risk(self, traf_state: np.ndarray, agent_state: np.ndarray, d_safe: float) -> float:
        traf_position = traf_state[0:2]
        agent_position = agent_state[0:2]
        traf_velocity = traf_state[2:4]
        agent_velocity = agent_state[2:4]

        relative_position = traf_position - agent_position
        relative_velocity = traf_velocity - agent_velocity

        relative_speed_squared = np.dot(relative_velocity, relative_velocity)
        
        # If agents are moving in parallel or stationary relative to each other
        if relative_speed_squared == 0:
            distance = np.linalg.norm(relative_position)
        else:
            # Calculate time to closest point of approach (t_cpa)
            t_cpa = -np.dot(relative_position, relative_velocity) / relative_speed_squared
            
            if t_cpa < 0:
                # The closest point was in the past (vehicles are moving apart).
                # The minimum distance in the future is their current distance.
                distance = np.linalg.norm(relative_position)
            else:
                # Calculate the exact distance at the future t_cpa
                pca_vector = relative_position + (t_cpa * relative_velocity)
                distance = np.linalg.norm(pca_vector)

        # Normalize by safe distance, but prevent negative risk
        if d_safe == 0:
            return 0.0
            
        risk = (d_safe - distance) / d_safe
        
        return max(0.0, float(risk))
    