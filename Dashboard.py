import streamlit as st
import streamlit.components.v1 as components
import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from pathlib import Path
import numpy as np


# Import your existing simulation pipeline
from Run_Functions import load_World, run_Single
from visualizations.Std_Visual import visualize_world, animate_trajectory_with_traffic
from visualizations.animate_planner_search import animate_planner_search

mpl.rcParams['animation.embed_limit'] = 100.0 # Increase th embeded limit to 100MB for the animations


st.set_page_config(layout="wide", page_title="UAV Path Planning")

st.title("🚁 Risk-Aware UAV Path Planning Simulator")
st.markdown("Select a test scenario to execute the GNC simulation and view the algorithm's search footprint.")

# 1. Sidebar Controls
st.sidebar.header("Simulation Setup")
scenario_dir = Path("scenarios")
scenario_files = [f.name for f in scenario_dir.glob("*.json")] 

if not scenario_files:
    st.sidebar.error("No JSON scenarios found in the 'scenarios' folder.")
else:
    # Scenario Dropdown
    selected_file = st.sidebar.selectbox("Choose Base Scenario", scenario_files)

    # NEW: Planner Configuration Dropdown
    planner_dir = Path("plan_src/Planner_Config")
    planner_files = [f.name for f in planner_dir.glob("*.json")]
    selected_planner = st.sidebar.selectbox("Choose Planner Config", planner_files)

    st.sidebar.markdown("---")
    
    # NEW: Live Parameter Overrides
    st.sidebar.subheader("🛠️ Live Parameter Overrides")
    override_params = st.sidebar.checkbox("Enable Live Tweaks")
    
    if override_params:
        # Create input widgets for the parameters you want to be adjustable
        col1, col2 = st.sidebar.columns(2)
        with col1:
            new_start_x = st.number_input("Start X", value=10.0, step=1.0)
            new_goal_x = st.number_input("Goal X", value=90.0, step=1.0)
        with col2:
            new_start_y = st.number_input("Start Y", value=10.0, step=1.0)
            new_goal_y = st.number_input("Goal Y", value=90.0, step=1.0)
            
        new_resolution = st.sidebar.slider("A* Grid Resolution", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
        new_radius = st.sidebar.number_input("Vehicle Risk Radius", value=1.0, step=0.5)

    # Visualization Layer Toggle
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎥 Render Settings")
    render_throttle = st.sidebar.slider(
        "Animation Throttle (Frame Skip)", 
        min_value=1, max_value=20, value=5, step=1,
        help="Higher values render faster but playback looks slightly choppier."
    )
    viz_mode = st.sidebar.radio("Visualization Layer", [
        "Final Path Only", 
        "Show Search Tree", 
        "Show Traffic Hazards",
        "Animate Search Process",
        "Animate Flight & Traffic"
    ])

    if st.sidebar.button("Run Simulation", type="primary"):
        with st.spinner("Executing Simulation & Pathfinding..."):
            
            # 1. Load the chosen Base Template
            config_path = scenario_dir / selected_file
            with open(config_path, 'r') as f:
                config_Data = json.load(f)
                
            # 2. Apply Live Overrides (if enabled)
            if override_params:
                # Update the Agent dictionaries with the new Streamlit values
                # (Adjust these dictionary keys to match your exact JSON structure)
                if "Agent" in config_Data:
                    config_Data["Agent"]["start"] = [new_start_x, new_start_y]
                    config_Data["Agent"]["goal"] = [new_goal_x, new_goal_y]
                    config_Data["Agent"]["radius"] = new_radius
                    
                if "Planner" in config_Data:
                    config_Data["Planner"]["resolution"] = new_resolution
            
            # We use a temporary folder for the web app logs
            save_dir = Path("gui_logs")
            save_dir.mkdir(exist_ok=True)
            
            # 3. Run the simulation using the modified dictionary and selected planner
            log = run_Single(
                config_Data=config_Data, 
                save_dir=save_dir, 
                planner_Config=selected_planner, # <-- Pass the dropdown selection here
                gui_mode=True
            )
            world = load_World(config_Data)

            # --- NEW: METRICS DASHBOARD SECTION ---
            st.markdown("---")
            st.subheader("📊 Mission Debrief & Metrics")
            
            # 1. High-Level Status Banners
            if log.success:
                st.success("✅ **Mission Accomplished:** UAV reached the target safely.")
            else:
                st.error(f"❌ **Mission Failed:** {log.reason}")

            # 2. Load and Display Detailed Metrics
            metrics_file = save_dir / "run_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                # Check for an outright planner failure (no paths generated)
                if metrics.get('total_replans', 0) == 0 and not log.success:
                    st.warning("⚠️ **Planner Alert:** The algorithmic planner failed to find a valid route.")
                
                # Display core metrics in a clean row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    sep = metrics.get("min_safe_separation_ft")
                    sep_text = f"{sep:.2f} ft" if sep is not None else "N/A"
                    st.metric(label="Min Safe Separation", value=sep_text)
                    
                with col2:
                    static_viol = metrics.get("static_violation", False)
                    # Using delta_color="inverse" makes "Clear" show up green and "Crash" show up red
                    st.metric(label="Static Violations", value="Yes" if static_viol else "None", 
                              delta="- Crash" if static_viol else "Clear", delta_color="inverse")
                    
                with col3:
                    traffic_viol = metrics.get("traffic_violation", False)
                    st.metric(label="Traffic Violations", value="Yes" if traffic_viol else "None",
                              delta="- Conflict" if traffic_viol else "Clear", delta_color="inverse")
                    
                with col4:
                    avg_time = metrics.get("avg_replan_computation_sec", 0.0)
                    st.metric(label="Avg Compute Time", value=f"{avg_time:.4f} s")
                    
            st.markdown("---")
            # --------------------------------------

            # 4. Render the Results
            st.subheader(f"Simulation Results: {viz_mode}")
            
            # Draw the bounds and obstacles
            visualize_world(world)
            fig = plt.gcf() 
            ax = plt.gca()
            
            # --- LAYER 1: Search Tree ---
            if viz_mode == "Show Search Tree":
                try:
                    nodes_df = pd.read_csv(save_dir / "run_sampled_nodes.csv")
                    if not nodes_df.empty:
                        segments = [((row['parent_x'], row['parent_y']), (row['x'], row['y'])) for _, row in nodes_df.iterrows()]
                        lc = LineCollection(segments, colors='lightgray', linewidths=0.5, alpha=0.7, zorder=1)
                        ax.add_collection(lc)
                except FileNotFoundError:
                    st.warning("No planner debug CSV found. Ensure PlannerLogger is running in debug_mode.")

            # --- LAYER 2: Traffic Hazards ---
            elif viz_mode == "Show Traffic Hazards":
                # Safely pull the traffic dictionary directly from the run log in RAM
                traffic_data = getattr(log, 'traffic_positions', getattr(log, 'traffic_telemetry', None))
                
                if traffic_data and isinstance(traffic_data, dict):
                    for traffic_id, positions in traffic_data.items():
                        if positions: # Ensure there is data
                            t_traj = np.array(positions).reshape(-1, 2)
                            # Draw the dashed track
                            ax.plot(t_traj[:, 0], t_traj[:, 1], '--', color='darkred', alpha=0.6, label=f'Intruder {traffic_id}')
                            # Add the red diamond at the final location
                            ax.plot(t_traj[-1, 0], t_traj[-1, 1], 'rD', markersize=6)
                else:
                    st.info("No traffic active in this scenario.")

            # --- LAYER 4: The Animation Player ---
            elif viz_mode == "Animate Search Process":
                with st.spinner("Compiling Animation Video (This takes a few seconds)..."):
                    
                    # Dynamically calculate batch size based on the throttle slider
                    # E.g., Throttle 5 = 250 nodes per frame. Throttle 1 = 50 nodes per frame.
                    batch_size = render_throttle * 50 
                    
                    ani = animate_planner_search(
                        world=world, log_dir=str(save_dir), filename_prefix="run", plan_id=0, 
                        gui_mode=True, nodes_per_frame=batch_size
                    )
                    
                    if ani:
                        components.html(ani.to_jshtml(), height=700)
                    else:
                        st.error("Failed to generate animation.")
                st.stop()

            # --- LAYER 5: The Flight & Traffic Animation Player ---
            elif viz_mode == "Animate Flight & Traffic":
                with st.spinner("Compiling Flight & Traffic Video..."):
                    
                    ani = animate_trajectory_with_traffic(
                        world=world, log=log, playback_speed_ms=50, 
                        gui_mode=True, frame_skip=render_throttle # <--- NEW PARAMETER
                    )
                    
                    if ani:
                        components.html(ani.to_jshtml(), height=700)
                    else:
                        st.error("Failed to generate animation.")
                
                st.stop()

            # --- ALWAYS SHOW: The Flown Trajectory ---
            if log.agent_positions:
                import numpy as np
                traj = np.array(log.agent_positions)
                ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=2.5, label='Flown Trajectory', zorder=3)
                # Add a solid dot at the final drone position
                ax.plot(traj[-1, 0], traj[-1, 1], 'bo', markersize=8)

            # Clean up the legend and render
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            fig.tight_layout()
            
            st.pyplot(fig)
            plt.clf()