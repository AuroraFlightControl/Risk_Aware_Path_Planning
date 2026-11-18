import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.patches import Ellipse

sigma_p = 0.5
sigma_v = 0.5
sigma_a = 0.5

dt = 0.1
head = 45
speed = 10.0

display_times = [0.0, 2.0, 4.0, 6.0, 8.0]

initial_state = np.array([0.0, 0.0, speed * np.cos(np.radians(head)), speed * np.sin(np.radians(head))], dtype=np.float64)

def ellipse_patch(mean, covariance, gamma, *, facecolor='none', **kwargs):
    evals, evecs = np.linalg.eigh(covariance)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:,order]
    angle = math.degrees(math.atan2(evecs[1,0], evecs[0,0]))
    size = 2 * np.sqrt(gamma * evals)
    return Ellipse(xy=mean, width=size[0], height=size[1], angle=angle,
                   facecolor=facecolor, **kwargs)

state = initial_state
covariance = np.diag([sigma_p**2, sigma_p**2, sigma_v**2, sigma_v**2])

fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlabel("X Position (m)")
ax.set_ylabel("Y Position (m)")

# FIX 2: Expanded plot boundaries to capture the full 8-second trajectory at 10 m/s
ax.set_xlim(-5, 80)
ax.set_ylim(-5, 80)
ax.grid()

for step_index in range(int(10.0 / dt) + 1):
    t = step_index * dt

    Q = sigma_a**2 * np.array([
        [(dt**4)/4, 0.0, (dt**3)/2, 0.0], 
        [0.0, (dt**4)/4, 0.0, (dt**3)/2], 
        [(dt**3)/2, 0.0, (dt**2), 0.0], 
        [0.0, (dt**3)/2, 0.0, (dt**2)]
        ], dtype=np.float64)

    F = np.array([
        [1.0, 0.0, dt, 0.0],
        [0.0, 1.0, 0.0, dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)

    # 1. Propagate Mean and Covariance
    mue_t = F @ state
    Sigma_t = F @ covariance @ F.T + Q

    # 2. Extract Position for Plotting
    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
    Sigma_t_pos = H @ Sigma_t @ H.T
    mue_t_pos = H @ mue_t

    # 3. Plot at specific time intervals
    if np.isclose(t, display_times).any():
        ellipse = ellipse_patch(mue_t_pos, Sigma_t_pos, gamma=5.991, edgecolor='blue', lw=2)
        ax.add_patch(ellipse)
        ax.plot(mue_t_pos[0], mue_t_pos[1], 'ro', label=f't = {t:.1f}s')

    # FIX 1: Overwrite state and covariance to propagate recursively
    state = mue_t
    covariance = Sigma_t

ax.legend()
plt.show()

"""Publication-ready dynamic traffic uncertainty figure.

Visualizes the spatial-temporal covariance growth of a non-cooperative 
traffic agent using a Piecewise Constant White Noise Acceleration (CWNA) model.
This illustrates the uncertainty bounding used by the Risk Oracle's 
Mahalanobis distance metric for edge evaluation.

Outputs:
  Dynamic_Traffic_Covariance.{pdf,svg,png}

Requires: numpy and matplotlib. Run as:
  python Dynamic_Traffic_Risk_Figures.py
"""

import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless rendering for clean file export
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# ---------------------------------------------------------
# Configuration and Constants
# ---------------------------------------------------------
OUT = Path(__file__).resolve().parent

# Kinematic and Noise Parameters
SIGMA_P = 0.5   # Initial position standard deviation (ft)
SIGMA_V = 0.5   # Initial velocity standard deviation (ft/s)
SIGMA_A = 0.5   # Process noise acceleration standard deviation (ft/s^2)

DT = 0.1        # Simulation time step (s)
SPEED = 10.0    # Initial agent speed (ft/s)
HEADING = 45.0  # Initial agent heading (deg)

# Times to extract and plot the covariance ellipses
DISPLAY_TIMES = [0.0, 2.0, 4.0, 6.0, 8.0]
MAX_TIME = 8.0

# ---------------------------------------------------------
# Publication Formatting (Aligned with Static Obstacle Style)
# ---------------------------------------------------------
plt.rcParams.update({
    'font.family': 'Liberation Sans',
    'font.size': 8.0,
    'axes.titlesize': 9.5,
    'axes.labelsize': 8.4,
    'xtick.labelsize': 7.4,
    'ytick.labelsize': 7.4,
    'legend.fontsize': 7.0,
    'savefig.facecolor': 'white',
    'figure.facecolor': 'white',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'axes.linewidth': 0.75,
})

TXT = '#1e293b'
GRAY = '#64748b'
ACCENT_BLUE = '#25678a'
ACCENT_RED = '#7d263c'

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def ellipse_patch(mean, covariance, gamma, **kwargs):
    """Generates a Matplotlib Ellipse patch from a 2D covariance matrix."""
    evals, evecs = np.linalg.eigh(covariance)
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    angle = math.degrees(math.atan2(evecs[1, 0], evecs[0, 0]))
    size = 2 * np.sqrt(gamma * evals)
    return Ellipse(xy=mean, width=size[0], height=size[1], angle=angle, **kwargs)

def save_many(fig, stem):
    """Exports the figure in multiple publication formats."""
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(OUT / f'{stem}.{ext}', dpi=420, facecolor='white', bbox_inches='tight')
        print(f'Wrote {stem}.{ext}')
    plt.close(fig)

# ---------------------------------------------------------
# Main Execution
# ---------------------------------------------------------
def generate_covariance_figure():
    # 1. Initialize State and Covariance
    state = np.array([
        0.0, 
        0.0, 
        SPEED * np.cos(np.radians(HEADING)), 
        SPEED * np.sin(np.radians(HEADING))
    ], dtype=np.float64)
    
    covariance = np.diag([SIGMA_P**2, SIGMA_P**2, SIGMA_V**2, SIGMA_V**2])

    # 2. Pre-compute Constant Matrices (F and Q)
    # This avoids recalculating constant matrices inside the simulation loop
    F = np.array([
        [1.0, 0.0, DT, 0.0],
        [0.0, 1.0, 0.0, DT],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ], dtype=np.float64)

    Q = SIGMA_A**2 * np.array([
        [(DT**4)/4, 0.0,       (DT**3)/2, 0.0], 
        [0.0,       (DT**4)/4, 0.0,       (DT**3)/2], 
        [(DT**3)/2, 0.0,       (DT**2),   0.0], 
        [0.0,       (DT**3)/2, 0.0,       (DT**2)]
    ], dtype=np.float64)

    H = np.array([
        [1.0, 0.0, 0.0, 0.0], 
        [0.0, 1.0, 0.0, 0.0]
    ], dtype=np.float64)

    # 3. Setup the Figure
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    
    # Track the mean path for plotting the trajectory line
    path_x, path_y = [state[0]], [state[1]]

    # 4. Simulation Loop
    num_steps = int(MAX_TIME / DT) + 1
    for step_index in range(num_steps):
        t = step_index * DT

        # Extract position data
        mue_t_pos = H @ state
        Sigma_t_pos = H @ covariance @ H.T

        # Plot specific intervals
        if any(np.isclose(t, d_time, atol=1e-3) for d_time in DISPLAY_TIMES):
            # 95% Confidence Interval (Gamma = 5.991 for 2 DOF)
            ellipse = ellipse_patch(mue_t_pos, Sigma_t_pos, gamma=5.991, 
                                    facecolor='#e2e8f0', alpha=0.6, 
                                    edgecolor=ACCENT_BLUE, lw=1.2, zorder=2)
            ax.add_patch(ellipse)
            
            # Center marker
            ax.scatter([mue_t_pos[0]], [mue_t_pos[1]], marker='D', s=20,
                       facecolors='white', edgecolors=ACCENT_RED,
                       linewidths=1.0, zorder=3)
            
            # Time annotation with bounding box
            ax.text(mue_t_pos[0] - 2.5, mue_t_pos[1] + 3.0, f'$t={t:.1f}$s', 
                    fontsize=7.5, color=TXT, weight='bold', ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.85, edgecolor='none', pad=1.5), 
                    zorder=4)

        # Propagate for the next step (Linear Covariance Growth)
        state = F @ state
        covariance = F @ covariance @ F.T + Q
        
        path_x.append(state[0])
        path_y.append(state[1])

    # Plot the mean trajectory line
    ax.plot(path_x, path_y, '--', color=GRAY, lw=1.0, alpha=0.7, zorder=1, label='Predicted Mean Trajectory')

    # 5. Styling and Limits
    ax.set_aspect('equal', adjustable='box') # CRITICAL: Ensures ellipses aren't distorted
    ax.set_xlim(-5, 85)
    ax.set_ylim(-5, 85)
    ax.set_xlabel("East position (ft)")
    ax.set_ylabel("North position (ft)")
    ax.set_title("Dynamic Traffic: CWNA Spatial Covariance Growth", fontweight='bold', loc='left', pad=10, color=TXT)
    
    # Tick formatting to match reference
    ax.tick_params(colors=GRAY, direction='out', length=3.0, width=0.65)
    for spine in ax.spines.values(): 
        spine.set_color('#94a3b8')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    ax.grid(True, color='#f1f5f9', linewidth=0.6)
    ax.legend(loc='lower right', frameon=True, facecolor='white', edgecolor='#e2e8f0')

    # 6. Export
    save_many(fig, 'Dynamic_Traffic_Covariance')

if __name__ == '__main__':
    generate_covariance_figure()