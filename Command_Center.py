import streamlit as st
import streamlit.components.v1 as components
import json, os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.collections import LineCollection
from pathlib import Path
import numpy as np

# Imports from your architecture
from Run_Functions import load_World, run_Single
from visualizations.Std_Visual import visualize_world, animate_trajectory_with_traffic
from visualizations.Risk_Heat_map import plot_risk_heatmap
from visualizations.animate_planner_search import animate_planner_search
from sim_src.simulation.RunSessionManager import RunSessionManager
from plan_src.RiskMonitor import RiskMonitor

mpl.rcParams['animation.embed_limit'] = 100.0 

st.set_page_config(layout="wide", page_title="UAV Path Planning Command Center")

st.title("Risk-Aware UAV Path Planning Simulation Command Center")

# --- CREATE THE TABS ---
tab1, tab2 = st.tabs([
    "Single Run Interface", 
    "Scenario Builder"
])

# ==========================================
# TAB 1: SINGLE RUN INTERFACE
# ==========================================
with tab1:
    st.markdown("Execute a single scenario, apply live overrides, and visualize the algorithmic search footprint.")
    
    col_controls, col_render = st.columns([1, 2])
    
    with col_controls:
        st.subheader("Simulation Setup")
        scenario_dir = Path("scenarios")
        scenario_files = [f.name for f in scenario_dir.glob("*.json")] 
        selected_file = st.selectbox("Choose Base Scenario", scenario_files)
        
        planner_dir = Path("plan_src/Planner_Config")
        planner_files = [f.name for f in planner_dir.glob("*.json")]
        selected_planner = st.selectbox("Choose Planner Config", planner_files)

        # 1. LOAD THE DATA EARLY so the UI can read it
        with open(scenario_dir / selected_file, 'r') as f:
            base_scenario = json.load(f)
            
        with open(planner_dir / selected_planner, 'r') as f:
            base_planner = json.load(f)

        st.markdown("---")
        override_params = st.checkbox("Enable Live Tweaks (Overrides)")
        
        # 2. DYNAMIC SCENARIO OVERRIDES
        if override_params:
            st.markdown("#### Environment Toggles")
            
            # Dynamically check for obstacles
            if "Obstacles" in base_scenario and len(base_scenario["Obstacles"]) > 0:
                if not st.checkbox(f"Enable Obstacles ({len(base_scenario['Obstacles'])})", value=True):
                    base_scenario["Obstacles"] = [] # Clear them out if unchecked
                    
            # Dynamically check for traffic
            if "Intruders" in base_scenario and len(base_scenario["Intruders"]) > 0:
                if not st.checkbox(f"Enable Traffic ({len(base_scenario['Intruders'])})", value=True):
                    base_scenario["Intruders"] = []
                else:
                    # Let the user dynamically change the behavioral model of the traffic
                    new_traffic_type = st.selectbox("Traffic Behavior Model", ["Constant_Velocity", "Static_Hover"])
                    for intruder in base_scenario["Intruders"]:
                        intruder["Type"] = new_traffic_type

            st.markdown("#### Algorithmic Parameters")
            # 3. DYNAMIC PLANNER OVERRIDES
            # Iterate through whatever is inside the planner JSON
            for key, value in base_planner.items():
                if key == "name":
                    continue # Skip the name so we don't break the loader
                
                # Format the key to look nice (e.g., 'vehicle_radius' -> 'Vehicle Radius')
                ui_label = str(key).replace('_', ' ').title()
                
                # Spawn the correct widget based on the JSON data type
                if isinstance(value, bool):
                    base_planner[key] = st.checkbox(ui_label, value=value)
                elif isinstance(value, float):
                    base_planner[key] = st.number_input(ui_label, value=float(value), step=0.1)
                elif isinstance(value, int):
                    base_planner[key] = st.number_input(ui_label, value=int(value), step=1)
                elif isinstance(value, str):
                    base_planner[key] = st.text_input(ui_label, value=str(value))

        st.markdown("---")
        st.subheader("🎥 Render Settings")
        render_throttle = st.slider("Animation Throttle (Frame Skip)", 1, 20, 5)
        
        viz_mode = st.radio("Visualization Layer", [
            "Final Path Only", "Show Search Tree", "Show Traffic Hazards", 
            "Animate Search Process", "Animate Flight & Traffic", 'Static Risk Heat Map'
        ])
        
        save_permanently = st.checkbox("Save to Run_Results (Permanent)")
        run_button = st.button("Run Simulation", type="primary", use_container_width=True)

    with col_render:
        if run_button:
            with st.spinner("Executing Simulation..."):
                
                # 4. ROUTE DIRECTORY & EXECUTE
                if save_permanently:
                    session_manager = RunSessionManager(root_config=base_scenario)
                    save_dir = session_manager.get_save_dir()
                else:
                    save_dir = Path("gui_logs")
                    save_dir.mkdir(exist_ok=True)
                
                # Pass the dynamically modified dictionaries directly to the runner
                log = run_Single(
                    config_Data=base_scenario, 
                    save_dir=save_dir, 
                    planner_Config=base_planner, # Passing the dict instead of string
                    gui_mode=True
                )
                world = load_World(base_scenario)
                
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

            # --- LAYER 6: Static Risk Heat Map ---
            elif viz_mode == 'Static Risk Heat Map':
                with st.spinner("Generating Static Risk Heat Map..."):
                    risk_monitor = RiskMonitor(world=world, d_safe_obs=10)
                    risk_field = risk_monitor.generate_static_risk_field(resolution=0.1)
                    if world.obstacles is not None:
                        obstacles = world.obstacles
                    else:
                        obstacles = []

                    plot_risk_heatmap(risk_field_data=risk_field, obstacles=obstacles, gui_mode = True)

                    fig = plt.gcf() 
                    ax = plt.gca()

                

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


# ==========================================
# TAB 2: SCENARIO BUILDER
# ==========================================
with tab2:
    st.markdown("Design a new environment and save it to the `scenarios/` folder.")
    
    with st.form("scenario_builder_form"):
        scen_name = st.text_input("Scenario Name (e.g., Narrow_Corridor)", "New_Scenario")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("World Bounds")
            x_max = st.number_input("X Max", value=100.0)
            y_max = st.number_input("Y Max", value=100.0)
            
            st.subheader("Agent Mission")
            start_x = st.number_input("Start X", value=10.0)
            start_y = st.number_input("Start Y", value=10.0)
            
        with col2:
            goal_x = st.number_input("Goal X", value=90.0)
            goal_y = st.number_input("Goal Y", value=90.0)
            
            # Future expansion: Add dynamic obstacle entry here
            st.info("Obstacle and Traffic array generation coming soon.")

        submitted = st.form_submit_button("Save Scenario JSON")
        if submitted:
            # Build the dictionary
            new_config = {
                "Project_Name": scen_name,
                "Run_Type": "Single",
                "Sim_Params": {"Time_Step": 0.01, "Max_Time": 100.0},
                "Agent": {"Type": "Holonomic", "Radius": 1.0},
                "Bounds": {"X_Min": 0.0, "X_Max": x_max, "Y_Min": 0.0, "Y_Max": y_max},
                "Start": {"X": start_x, "Y": start_y},
                "Goal": {"X": goal_x, "Y": goal_y},
                "Obstacles": [],
                "Intruders": []
            }
            
            # Save it to the directory
            file_path = Path("scenarios") / f"{scen_name}.json"
            with open(file_path, 'w') as f:
                json.dump(new_config, f, indent=4)
                
            st.success(f"Successfully saved {scen_name}.json! It is now available in the Single Run Playground.")


# ==========================================
# TAB 3: BATCH & MC ANALYTICS (Not Published)
# ==========================================

#with tab3:
#    st.markdown("Review aggregated metrics from high-volume Monte Carlo and Comparison runs.")
#    
#    results_dir = Path("Run_Results")
#    if not results_dir.exists():
#        st.info("No Run_Results folder found. Run some batch simulations from the command line first!")
#    else:
#        # Example of how you would crawl the results folder
#        projects = [d.name for d in results_dir.iterdir() if d.is_dir()]
#        selected_project = st.selectbox("Select Project to Analyze", projects)
#       
#        st.warning("Analytics dashboard wiring is under construction. This will eventually parse MC_Summary.json files to generate Pandas DataFrames and Matplotlib charts.")

