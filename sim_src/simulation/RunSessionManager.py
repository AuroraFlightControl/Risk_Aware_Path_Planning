import os, shutil, json
from datetime import datetime
from pathlib import Path

class RunSessionManager:
    def __init__(self, root_config: dict):
        self.project_name = root_config.get("Project_Name", "Unnamed_Project")
        self.run_type = root_config.get("Run_Type", "Unknown_Run")

        self.timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        self.session_dir = Path("Run_Results") / self.project_name / self.run_type / self.timestamp
        os.makedirs(self.session_dir, exist_ok=True)

        config_snapshot_path = self.session_dir / "execution_config.json"
        with open(config_snapshot_path, 'w') as f:
            json.dump(root_config, f, indent=4)
        
    def get_save_dir(self, enviroment_name: str = "", planner_name: str = "", run_id: str = "") -> Path:
        
        current_dir = self.session_dir

        if enviroment_name:
            current_dir = current_dir / enviroment_name
        if planner_name:
            current_dir = current_dir / planner_name
        if run_id:
            current_dir = current_dir / run_id

        os.makedirs(current_dir, exist_ok=True)
        
        return current_dir