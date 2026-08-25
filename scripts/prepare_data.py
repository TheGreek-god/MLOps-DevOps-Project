"""
Data preparation script.
Cleans raw fetched data, applies imputation, creates driver/team mappings.
Input:  data/Complete_Driver_Data.csv (raw, from bootstrap fetch)
Output: data/Complete_Driver_Data.csv (cleaned), config/mappings.json
"""
import os
import sys
import json
import yaml
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
CONFIG_DIR = os.path.join(ROOT, "config")

os.makedirs(CONFIG_DIR, exist_ok=True)

with open(os.path.join(ROOT, "params.yaml")) as f:
    params = yaml.safe_load(f)

prep = params["prepare"]
team_replacements = prep["team_replacements"]
driver_replacements = prep["driver_replacements"]
time_cols = prep["time_cols"]

raw_path = os.path.join(DATA_DIR, "Complete_Driver_Data.csv")
if not os.path.exists(raw_path):
    print(f"ERROR: Raw data not found at {raw_path}")
    print("Run 'python scripts/bootstrap.py' first to fetch data.")
    sys.exit(1)

df = pd.read_csv(raw_path)
print(f"Loaded raw data: {len(df)} rows, {df['Year'].nunique()} years")

df = df.sort_values(by=["Year", "Round", "FinishPos"], ascending=[True, True, True]).reset_index(drop=True)


def smart_impute_grid(group):
    for col in time_cols:
        group[col] = pd.to_numeric(group[col], errors="coerce")
    gaps = group[time_cols].diff()
    mean_gaps = gaps.mean().fillna(0)
    for i in range(len(group)):
        if pd.isna(group.iloc[i]["QualyTime"]) or pd.isna(group.iloc[i]["AvgFPTime"]):
            if i > 0:
                group.iloc[i, group.columns.get_indexer(time_cols)] = (
                    group.iloc[i - 1][time_cols] + mean_gaps
                )
            else:
                group[time_cols] = group[time_cols].bfill()
    return group


df = df.groupby(["Year", "Round"], group_keys=False).apply(smart_impute_grid)
df["IsAccurate"] = df["IsAccurate"].fillna(False).astype(bool)

for col in time_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df[time_cols] = df[time_cols].fillna(0)
df["FinishPos"] = df["FinishPos"].fillna(20)

print("Applying name replacements...")
for old, new in team_replacements.items():
    df["Team"] = df["Team"].replace(old, new)
for old, new in driver_replacements.items():
    df["Driver"] = df["Driver"].replace(old, new)

print("Computing driver form rolling averages...")
df = df.sort_values(by=["Driver", "Year", "Round"]).reset_index(drop=True)
df["Driver_Form_Avg"] = (
    df.groupby("Driver")["FinishPos"]
    .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
)
df["Driver_Form_Avg"] = df["Driver_Form_Avg"].fillna(20.0)
df = df.sort_values(by=["Year", "Round", "FinishPos"]).reset_index(drop=True)

csv_path = os.path.join(DATA_DIR, "Complete_Driver_Data.csv")
df.to_csv(csv_path, index=False)
print(f"Cleaned data saved: {csv_path} ({len(df)} rows)")

print("Creating driver/team mappings...")
unique_drivers = sorted(df["Driver"].unique())
mapping_drivers = {"UNKNOWN_DRIVER": 0}
mapping_drivers.update({driver: idx for idx, driver in enumerate(unique_drivers, start=1)})

unique_teams = sorted(df["Team"].unique())
mapping_teams = {"UNKNOWN_TEAM": 0}
mapping_teams.update({team: idx for idx, team in enumerate(unique_teams, start=1)})

master_mapping = {"drivers": mapping_drivers, "teams": mapping_teams}
mappings_path = os.path.join(CONFIG_DIR, "mappings.json")
with open(mappings_path, "w") as f:
    json.dump(master_mapping, f, indent=4)

print(f"Mappings saved: {mappings_path}")
print(f"  Drivers: {len(mapping_drivers)}, Teams: {len(mapping_teams)}")
print(f"  Years: {sorted(df['Year'].unique())}")
print(f"  Races: {df.groupby(['Year', 'Round']).ngroups}")
print("prepare_data complete!")
