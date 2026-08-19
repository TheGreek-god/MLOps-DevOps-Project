import json
import pandas as pd
import os
import sys

def find_project_root():
    current = os.path.dirname(os.path.abspath(__file__))
    while current != os.path.dirname(current):
        if os.path.exists(os.path.join(current, "requirements.txt")):
            return current
        current = os.path.dirname(current)
    raise FileNotFoundError("Could not find project root")

root_path = find_project_root()
src_path = os.path.join(root_path,"src")
if(root_path not in sys.path):
    sys.path.insert(0,root_path)
if(src_path not in sys.path):
    sys.path.insert(0,src_path)

DATAPATH = os.path.join(root_path,"data","Complete_Driver_Data.csv")

df = pd.read_csv(DATAPATH)


df["Driver"] = df["Driver"].replace("Andrea Kimi Antonelli", "Kimi Antonelli")


unique_drivers = sorted(df["Driver"].unique())
mapping_drivers = {"UNKNOWN_DRIVER": 0}
mapping_drivers.update(
    {driver: idx for idx, driver in enumerate(unique_drivers, start=1)}
)

unique_teams = sorted(df["Team"].unique())
mapping_teams = {"UNKNOWN_TEAM": 0}
mapping_teams.update({team: idx for idx, team in enumerate(unique_teams, start=1)})

master_mapping = {"drivers": mapping_drivers, "teams": mapping_teams}


with open("config/mappings.json", "w") as f:
    json.dump(master_mapping, f, indent=4)

print(" mappings.json generated successfully and formatted cleanly!")