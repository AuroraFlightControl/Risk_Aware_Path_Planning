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


class CVUncertinty:

    def __init__(self, sigma_a: float = 0.5) -> None:
        self.sigma_a = sigma_a  # Acceleration noise standard deviation

    def predict(self, state: tuple[np.ndarray,  np.ndarray], dt: float) -> tuple[np.ndarray, np.ndarray]:
        # State = [x, y, vx, vy]^T
        mu, sigma = state

        F = np.array([
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        Q = self.sigma_a**2 * np.array([[(dt**4)/4, 0, (dt**3)/2, 0],
                            [0, (dt**4)/4, 0, (dt**3)/2],
                            [(dt**3)/2, 0, dt**2, 0],
                            [0, (dt**3)/2, 0, dt**2]])

        mu_next = F @ mu
        sigma_next = F @ sigma @ F.T + Q

        return mu_next, sigma_next

    def Collision_Risk(self, traf_state: tuple[np.ndarray, np.ndarray], agent_state: np.ndarray, gamma: float) -> float:

        traf_mu, traf_sigma = traf_state

        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]])

        # Extract 2x2 position Covariance
        traf_sigma_p = H @ traf_sigma @ H.T

        # Invert Position Covariance
        try:
            traf_sigma_p_inv = np.linalg.inv(traf_sigma_p)
        except np.linalg.LinAlgError:
            traf_sigma_p_inv = np.linalg.pinv(traf_sigma_p)

        # Position Difference
        delta_pos = traf_mu[0:2] - agent_state[0:2]

        # Mahalanobis Distance
        mahalanobis_sq = delta_pos.T @ traf_sigma_p_inv @ delta_pos

        # Layered Constraint Handling
        if mahalanobis_sq <= gamma:
            return float('inf') # High risk if within the threshold
        else: 
            return float(np.exp(-0.5 * mahalanobis_sq)) # Risk decreases with distance
        



