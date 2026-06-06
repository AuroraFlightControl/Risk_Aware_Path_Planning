# Risk_Aware_Path_Planning
Repo for my master's thesis titled "A Risk-Aware Sampling-Based Planning Architecture for Small UAVs in Dynamic and Uncertain Environments".  The framework is developed as a 2D+time simulation that supports simplified kinematic models for quadcopters and fixed-wing aircraft.

# Required Packages
- Numpy
- Matplotlib


# Architecture Summary
The code is structured as a classic object-oriented approach, where the simulation is encapsulated by an object, along with the principal agent, traffic agents, and obstacles. The sim_src file contains all simulation-specific files, and the plan_src file contains the planners and their code. 

Run the simulations from the Run_Simulation file specifying the run scenario file you wish to use. Note the structure for runs has multiple layers used for different analyses. Most scenarios are single-run files that contain the world data, simulation parameters, obstacles, and traffic. These files will have a single planner and agent attached, and will output a single run folder containing telemetry for each involved agent, planner processes, and planner results. Batch run files will run a set of run files and output a layered output folder structure. Monte Carlo runs will take place in an environment and, based on the specified settings, randomize and run the simulations a number of times, generating run statistics for large-scale analysis. The final run type is the comparison run type, where it will run the specified scenario files, such as single runs, batch runs, and or Monte-Carlo simulations, and then output a comparison analysis of the runs. The comparison will run the specified agents and planners through each specified run file for the analysis. 
