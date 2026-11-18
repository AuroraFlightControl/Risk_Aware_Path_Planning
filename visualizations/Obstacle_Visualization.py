from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
from matplotlib.patches import Circle
from dataclasses import dataclass
import math


# Simple Bounds Class to hold the bounds and bouundry checking for the world.
@dataclass
class Bounds:
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def in_bounds(self, point: np.ndarray) -> bool:
        """Check if a point is within the bounds of the world."""
        x, y = point
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax

@dataclass
class Obstacle:
    position: np.ndarray                        # Position of the obstacle
    radius: float                               # Radius of the obstacle

    def distance(self, point: np.ndarray) -> float:
        dist_to_center = math.dist(point, self.position)
        return  max(0.0, dist_to_center - self.radius)


# Simple World Class to hold the obstacles and the bounds of the world
@dataclass
class World:
    bounds: Bounds                              # Bounds of the world
    start: np.ndarray                           # Start position of the agent
    goal: np.ndarray                            # Goal position of the agent
    obstacles: Optional[List[Obstacle]] = None    # List of obstacles in the world

def static_risk(world: World, point: np.ndarray, safety_radius: float = 5.0) -> float:
    if world.obstacles is None:
        return float(0.0)
    
    max_risk = 0.0

    for obs in world.obstacles:
        d_clear = obs.distance(point)

        if d_clear < safety_radius:
            risk = (safety_radius - d_clear) / safety_radius

            if risk > max_risk:
                max_risk = risk

    return float(max_risk)

def risk_map(world: World, resolution: float, safety_radius: float = 1.0) -> np.ndarray:
    # Discritize the world into a grid based on the resolution
    x_min, x_max = world.bounds.xmin, world.bounds.xmax
    y_min, y_max = world.bounds.ymin, world.bounds.ymax

    x_size = int(np.ceil((x_max - x_min) / resolution))
    y_size = int(np.ceil((y_max - y_min) / resolution))

    # Initialize grid with 0.0 (free space)
    grid = np.zeros((x_size, y_size))

    for x in range(x_size):
        for y in range(y_size):
            query_pt = np.array([x_min + x * resolution, y_min + y * resolution])
            grid[x, y] = static_risk(world, query_pt, safety_radius)

    return grid


def visualize(world: World) -> None:
    """Visualize the world with obstacles, start, and goal positions."""
    fig, ax = plt.subplots()

    # Plot the bounds of the world
    ax.set_xlim(world.bounds.xmin, world.bounds.xmax)
    ax.set_ylim(world.bounds.ymin, world.bounds.ymax)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('World Visualization')
    ax.set_xlabel('Position East (ft)')
    ax.set_ylabel('Position North (ft)')

    # Plot start and goal positions
    #ax.plot(world.start[0], world.start[1], 'go', label='Start')  # Green dot for start
    #ax.plot(world.goal[0], world.goal[1], 'ro', label='Goal')     # Red dot for goal

    # Plot obstacles
    if world.obstacles:
        for obs in world.obstacles:
            circle = Circle((obs.position[0], obs.position[1]), obs.radius, color='black', alpha=0.1)
            ax.add_patch(circle)


    risk_field = risk_map(world, resolution=0.1, safety_radius=10.0)
    extent = (world.bounds.xmin, world.bounds.xmax, world.bounds.ymin, world.bounds.ymax)
    im = ax.imshow(risk_field.T, extent=extent, origin='lower', cmap='hot', alpha=0.5)
    plt.colorbar(im, ax=ax, label='Risk Level')



    ax.legend()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()


if __name__ == "__main__":
    # Example usage
    bounds = Bounds(xmin=0, xmax=50, ymin=0, ymax=50) 
    start = np.array([10, 10])
    goal = np.array([90, 90])

    # Define some obstacles
    obstacles = [
        Obstacle(position=np.array([15, 37.5]), radius=10),
        Obstacle(position=np.array([37.5, 15]), radius=10),
        #Obstacle(position=np.array([40, 40]), radius=20)
    ]


    world = World(bounds=bounds, start=start, goal=goal, obstacles=obstacles)



    visualize(world)


"""Publication-ready static obstacle risk figures.

Reproduces the risk law, obstacle positions, obstacle radii, and safety
influence distance from Obstacle_Visualization.py (provided by author).

Risk is a normalized, geometrically defined penalty (NOT a collision
probability). For each obstacle, clearance is zero inside the obstacle;
per-obstacle risk decays linearly from 1 at the obstacle boundary to 0
at the distance r_s outside the boundary. Overall risk is the MAX of
the per-obstacle fields, not their sum.

Outputs:
  Static_Obstacle_Risk_Field.{pdf,svg,png}      plan view + 3D surface
  Static_Obstacle_Saddle_Analysis.{pdf,svg,png} plan view + transverse profiles
  Static_Obstacle_Risk_Surface_3D.{pdf,svg,png} standalone 3D figure

Requires: numpy and matplotlib. Run as:
  python Static_Obstacle_Risk_Figures.py
"""


import math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.ticker import FormatStrFormatter

OUT = Path(__file__).resolve().parent
RESOLUTION = 0.125  # ft, grid for smooth surface / contour render
BOUNDS = (0.0, 50.0, 0.0, 50.0)
SAFETY_RADIUS = 10.0  # ft, measured from each obstacle BOUNDARY

@dataclass(frozen=True)
class Obstacle:
    x: float
    y: float
    radius: float

OBSTACLES = (
    Obstacle(15.0, 37.5, 10.0),
    Obstacle(37.5, 15.0, 10.0),
)

MID = np.array([(OBSTACLES[0].x+OBSTACLES[1].x)/2,
                (OBSTACLES[0].y+OBSTACLES[1].y)/2])
D = np.array([OBSTACLES[1].x-OBSTACLES[0].x,
              OBSTACLES[1].y-OBSTACLES[0].y])
ALONG = D / np.linalg.norm(D)
ACROSS = np.array([-ALONG[1], ALONG[0]])


def scalar_risk(x: float, y: float) -> float:
    """Exact scalar form of the original supplied risk function."""
    r = 0.0
    for ob in OBSTACLES:
        clearance = max(0.0, math.hypot(x - ob.x, y - ob.y) - ob.radius)
        if clearance < SAFETY_RADIUS:
            r = max(r, (SAFETY_RADIUS - clearance) / SAFETY_RADIUS)
    return r


def risk_field(x, y):
    """Vectorized equivalent of scalar_risk for continuous plotting."""
    risk = np.zeros(np.broadcast_shapes(np.shape(x), np.shape(y)), dtype=float)
    for ob in OBSTACLES:
        clearance = np.maximum(0.0, np.hypot(x - ob.x, y - ob.y) - ob.radius)
        contribution = np.clip(1.0 - clearance / SAFETY_RADIUS, 0.0, 1.0)
        risk = np.maximum(risk, contribution)
    return risk


# Dense exact-position grid (no imshow extents misalignment).
xvals = np.arange(BOUNDS[0], BOUNDS[1] + RESOLUTION/2, RESOLUTION)
yvals = np.arange(BOUNDS[2], BOUNDS[3] + RESOLUTION/2, RESOLUTION)
X, Y = np.meshgrid(xvals, yvals)
Z = risk_field(X, Y)
R_MID = scalar_risk(*MID)

CMAP = LinearSegmentedColormap.from_list('static_risk', [
    (0.00, '#f8fafc'),
    (0.10, '#f3e8e6'),
    (0.28, '#f7c6a5'),
    (0.50, '#f18e72'),
    (0.75, '#c54358'),
    (1.00, '#68243d'),
], N=256)
NORM = Normalize(0,1)

plt.rcParams.update({
    'font.family':'Liberation Sans',
    'font.size':8.0,
    'axes.titlesize':9.5,
    'axes.labelsize':8.4,
    'xtick.labelsize':7.4,
    'ytick.labelsize':7.4,
    'legend.fontsize':7.0,
    'savefig.facecolor':'white',
    'figure.facecolor':'white',
    'pdf.fonttype':42,
    'ps.fonttype':42,
    'svg.fonttype':'none',
    'axes.linewidth':0.75,
})
TXT = '#1e293b'
GRAY = '#64748b'


def style_plan(ax, transects=False, saddle_callout=True):
    ax.pcolormesh(X, Y, Z, cmap=CMAP, norm=NORM,
                  shading='nearest', rasterized=True)
    ax.contour(X, Y, Z, levels=[.2,.4,.6,.8],
               colors='#6e4850', linewidths=.47, alpha=.58, zorder=3)
    for i, obs in enumerate(OBSTACLES, start=1):
        ax.add_patch(Circle((obs.x,obs.y), obs.radius,
                            edgecolor='#172331', facecolor='none',
                            linewidth=1.35, zorder=6))
        ax.plot(obs.x,obs.y, '.', color='#172331', markersize=2.2, zorder=7)
        ax.text(obs.x, obs.y, f'O{i}', fontsize=7.5,
                color='#172331', weight='bold', ha='center', va='center',
                bbox=dict(facecolor='white', alpha=.92,
                          edgecolor='none', pad=1.8), zorder=8)
    if transects:
        half=11.5
        pa = np.vstack([MID-half*ALONG, MID+half*ALONG])
        pp = np.vstack([MID-half*ACROSS, MID+half*ACROSS])
        ax.plot(pa[:,0],pa[:,1],'-',lw=1.8,color='#7d263c',
                zorder=8,solid_capstyle='round')
        ax.plot(pp[:,0],pp[:,1],'-',lw=1.8,color='#25678a',
                zorder=9,solid_capstyle='round')
        ax.text(*(MID - 5.8*ALONG), 'A', color='#7d263c', weight='bold',
                fontsize=7.5, ha='center',va='center',zorder=11,
                bbox=dict(facecolor='white',edgecolor='none',pad=1.6,alpha=.96))
        ax.text(*(MID + 5.8*ACROSS), 'B', color='#25678a', weight='bold',
                fontsize=7.5, ha='center',va='center',zorder=11,
                bbox=dict(facecolor='white',edgecolor='none',pad=1.6,alpha=.96))
    ax.scatter([MID[0]],[MID[1]],marker='D',s=30,
               facecolors='white',edgecolors='#172331',
               linewidths=1.2,zorder=12)
    if saddle_callout:
        ax.annotate(f'Saddle\n$R={R_MID:.3f}$', xy=MID, xytext=(36.4, 34.6),
                    ha='center', va='center', fontsize=7.5, color=TXT,
                    bbox=dict(facecolor='white',edgecolor='#e2e8f0',
                              boxstyle='round,pad=.22',linewidth=.5,alpha=.98),
                    arrowprops=dict(arrowstyle='-',lw=.8,color=TXT,
                                    shrinkA=4,shrinkB=5),
                    zorder=15)
    ax.set_aspect('equal', adjustable='box')
    ax.set(xlim=(0,50),ylim=(0,50),xlabel='East position (ft)',
           ylabel='North position (ft)')
    ax.set_xticks(np.arange(0,51,10)); ax.set_yticks(np.arange(0,51,10))
    ax.tick_params(colors=GRAY, direction='out', length=2.8, width=.65)
    for spine in ax.spines.values(): spine.set_color('#94a3b8')
    ax.grid(False)


def style_3d(ax):
    # Surface shows the same field, including saturation at R=1 inside discs.
    ax.plot_surface(X[::2,::2], Y[::2,::2], Z[::2,::2],
                    cmap=CMAP, norm=NORM,
                    rstride=1,cstride=1,
                    edgecolor='none',linewidth=0,
                    antialiased=True,rasterized=True,
                    shade=False, alpha=1)
    t=np.linspace(0,2*np.pi,170)
    for ob in OBSTACLES:
        ax.plot(ob.x + ob.radius*np.cos(t),
                ob.y + ob.radius*np.sin(t),
                np.full_like(t,1.012),
                color=TXT, alpha=.85,lw=.8, zorder=15)
    ax.scatter([MID[0]],[MID[1]],[R_MID+0.032],marker='D',
               s=20,color=TXT,edgecolor='white',linewidth=.45,
               depthshade=False,zorder=25)
    ax.set(xlim=(0,50),ylim=(0,50),zlim=(0,1.1))
    ax.set_xlabel('East (ft)',labelpad=-1)
    ax.set_ylabel('North (ft)',labelpad=-1)
    ax.set_zlabel('Risk',labelpad=-1)
    ax.set_xticks([0,25,50]);ax.set_yticks([0,25,50]);ax.set_zticks([0,.5,1.0])
    ax.set_box_aspect((1.08,1.08,.60),zoom=1.02)
    ax.view_init(elev=27, azim=-134)
    ax.tick_params(labelsize=7,pad=-2,colors=GRAY)
    # Subtle panes and projected grid; non-intrusive in print
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
        axis.pane.fill=False
        axis.line.set_color('#94a3b8')
        axis._axinfo['grid']['color']=(.78,.81,.85,.35)
        axis._axinfo['grid']['linewidth']=.45


def save_many(fig, stem):
    for ext in ('png','pdf','svg'):
        fig.savefig(OUT / f'{stem}.{ext}', dpi=420,facecolor='white')
        print(f'Wrote {stem}.{ext}')
    plt.close(fig)


def field_and_surface():
    fig=plt.figure(figsize=(7.55,3.54),layout=None)
    gs=fig.add_gridspec(nrows=1,ncols=3,
                        width_ratios=[1.05,1.25,.05],
                        left=.077,right=.932,bottom=.155,top=.855,wspace=.09)
    left=fig.add_subplot(gs[0,0]); style_plan(left)
    left.set_title('(a) Plan-view risk field',fontweight='bold',loc='left',pad=10,color=TXT)
    right=fig.add_subplot(gs[0,1],projection='3d'); style_3d(right)
    right.set_title('(b) Three-dimensional risk surface',fontweight='bold',pad=6,color=TXT)
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=NORM,cmap=CMAP),cax=fig.add_subplot(gs[0,2]))
    cb.set_ticks([0,.25,.5,.75,1])
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cb.set_label('Normalized static risk',labelpad=5)
    cb.outline.set_linewidth(.5)
    cb.ax.tick_params(length=2,width=.55,labelsize=7)
    save_many(fig,'Static_Obstacle_Risk_Field')


def saddle_analysis():
    s=np.linspace(-11.5,11.5,601)
    lin= MID[None,:]+s[:,None]*ALONG[None,:]
    perp= MID[None,:]+s[:,None]*ACROSS[None,:]
    r_along=risk_field(lin[:,0],lin[:,1])
    r_across=risk_field(perp[:,0],perp[:,1])
    fig=plt.figure(figsize=(7.55,3.5))
    gs=fig.add_gridspec(1,2,width_ratios=[1,1.17],left=.073,right=.965,
                        top=.86,bottom=.175,wspace=.23)
    left=fig.add_subplot(gs[0,0]);style_plan(left,transects=True,saddle_callout=False)
    left.set_title('(a) Two orthogonal transects',fontweight='bold',loc='left',pad=8,color=TXT)
    right=fig.add_subplot(gs[0,1])
    right.plot(s,r_along,color='#7d263c',lw=2.0,label='A: Between obstacle centers')
    right.plot(s,r_across,color='#25678a',lw=2.0,label='B: Perpendicular direction')
    right.axvline(0,color='#64748b',alpha=.6,lw=.9,ls='--')
    right.axhline(R_MID,color='#64748b',alpha=.4,lw=.8,ls=':')
    right.scatter([0],[R_MID],marker='D',s=37,
                  facecolor='white',edgecolor=TXT,lw=1.15,zorder=10)
    right.annotate(f'Saddle  $R={R_MID:.3f}$',xy=(0,R_MID),
                   xytext=(3.0,.56),fontsize=7.5,color=TXT,
                   arrowprops=dict(arrowstyle='-',color=TXT,lw=.8),
                   bbox=dict(facecolor='white',edgecolor='none',pad=2.0),zorder=11)
    right.set(xlim=(-11.5,11.5),ylim=(.12,1.045),
              xlabel='Signed distance from saddle (ft)',ylabel='Normalized static risk')
    right.set_xticks([-10,-5,0,5,10]);right.set_yticks([.2,.4,.6,.8,1.0])
    right.grid(axis='y',color='#e2e8f0',linewidth=.65)
    right.tick_params(colors=GRAY,direction='out',length=3,width=.7)
    for s_ in right.spines.values():s_.set_color('#94a3b8')
    right.spines['top'].set_visible(False);right.spines['right'].set_visible(False)
    right.set_title('(b) Saddle-point directional profiles',fontweight='bold',loc='left',pad=8,color=TXT)
    right.text(-3.55,.948,'A  Along obstacle centers',fontsize=7.5,
               color='#7d263c',weight='bold',ha='left',va='center',
               bbox=dict(facecolor='white',edgecolor='none',pad=1.3))
    right.text(.8,.19,'B  Perpendicular',fontsize=7.5,
               color='#25678a',weight='bold',ha='left',va='center',
               bbox=dict(facecolor='white',edgecolor='none',pad=1.3))
    fig.text(.5,.04,'At the marked point, risk is minimal along A and maximal along B.',
             ha='center',color=GRAY,fontsize=7.5)
    save_many(fig,'Static_Obstacle_Saddle_Analysis')


def standalone_3d():
    fig=plt.figure(figsize=(5.55,4.35))
    ax=fig.add_subplot(111,projection='3d');style_3d(ax)
    ax.set_position([.055,.065,.80,.88])
    cax=fig.add_axes([.88,.22,.025,.57])
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=NORM,cmap=CMAP),cax=cax)
    cb.set_ticks([0,.25,.5,.75,1]);cb.set_label('Normalized risk',labelpad=7)
    cb.outline.set_linewidth(.6)
    save_many(fig,'Static_Obstacle_Risk_Surface_3D')


if __name__ == '__main__':
    # Confirm numerical equivalence to supplied formulation at representative points.
    samples=np.array([[2.,2.], [15.,37.5], [37.5,15.],MID,
                      [26.25,35.0], [24.9,27.1]])
    vector_values=risk_field(samples[:,0],samples[:,1])
    scalar_values=np.array([scalar_risk(*xy) for xy in samples])
    assert np.allclose(vector_values,scalar_values,atol=1e-12)
    assert np.isclose(R_MID,0.409009742330268)
    print('Saddle point:', MID, 'risk:', R_MID)
    field_and_surface()
    saddle_analysis()
    standalone_3d()
