import numpy as np

from sim_src.enviroment.World import World




class Risk_Model:
    def __init__(self, world: World, resolution: float) -> None:
        self.world = world
        self.resolution = resolution
        

        x_min, x_max = world.bounds.xmin, world.bounds.xmax
        y_min, y_max = world.bounds.ymin, world.bounds.ymax

        x_size = int(np.ceil((x_max - x_min) / resolution))
        y_size = int(np.ceil((y_max - y_min) / resolution))

        self.risk_map = np.zeros((x_size, y_size))


    def Obs_risk(self, point: np.ndarray, safety_radius: float = 10.0) -> float:
        if self.world.obstacles is None:
            return float(0.0)

        max_risk = 0.0

        for obs in self.world.obstacles:
            d_clear = obs.distance(point)

            if d_clear < safety_radius:
                risk = (safety_radius - d_clear) / safety_radius

                if risk > max_risk:
                    max_risk = risk

        return float(max_risk)

    def Traf_risk(self, point: np.ndarray, traf_state: np.ndarray, dt: float) -> float:

        sigma_p = 10 * 3.281    # Positional deviation NACp 10
        sigma_v = 3 * 3.281     # Horizontal Velocity deviation NACv 2
        sigma_a = 5             # Acceleration standard deviation parameter, Not verified yet.

        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]
                      ], type=np.float64)

        Sigma_0 = np.diag([sigma_p, sigma_p, sigma_v, sigma_v], type=np.float64)

        Q = sigma_a**2 * np.array([[(dt**4)/4, 0, (dt**3)/2, 0],
                                   [0, (dt**4)/4, 0, (dt**3)/2],
                                   [(dt**3)/2, 0, dt**2, 0],
                                   [0, (dt**3)/2, 0, dt**2]
                                   ], type=np.float64)

        State = F @ traf_state
        Sigma = F @ Sigma_0 @ F.T + Q

        H = np.array([[1, 0, 0, 0], 
                      [0, 1, 0, 0]
                      ], type=np.float64)

        Traf_pos    = H @ State
        Sigma_p     = H @ Sigma @ H.T

        dp = Traf_pos - point

        # Invert Position Covariance
        try:
            Sigma_p_inv = np.linalg.inv(Sigma_p)
        except np.linalg.LinAlgError:
            Sigma_p_inv = np.linalg.pinv(Sigma_p)

        dM2 = dp.T @ Sigma_p_inv @ dp
        



