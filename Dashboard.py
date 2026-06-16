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

            st.success(f"Simulation Complete! Mission Success: {log.success}")

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
                    # Call the function with gui_mode=True to prevent popups
                    ani = animate_planner_search(world=world, log_dir=str(save_dir), filename_prefix="run", plan_id=0, gui_mode=True, nodes_per_frame=10)
                    
                    if ani:
                        # Convert the Matplotlib animation to an interactive HTML5 player
                        components.html(ani.to_jshtml(), height=700)
                    else:
                        st.error("Failed to generate animation.")
                
                # We use 'continue' or 'return' here so it doesn't draw the static 
                # trajectory from the "ALWAYS SHOW" block over top of the video player
                st.stop()

            # --- LAYER 5: The Flight & Traffic Animation Player ---
            elif viz_mode == "Animate Flight & Traffic":
                with st.spinner("Compiling Flight & Traffic Video..."):
                    # Call the function passing the log object directly
                    ani = animate_trajectory_with_traffic(world=world, log=log, playback_speed_ms=50, gui_mode=True)
                    
                    if ani:
                        # Convert to an interactive HTML5 player
                        components.html(ani.to_jshtml(), height=700)
                    else:
                        st.error("Failed to generate animation.")
                
                st.stop() # Prevent drawing the static trajectory over the video

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