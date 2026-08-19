import json
import pandas as pd
import numpy as np

TEAM_ALIASES = {"Kick Sauber": "Audi"}
DRIVER_ALIASES = {"Andrea Kimi Antonelli": "Kimi Antonelli"}


def normalize_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Team"] = df["Team"].replace(TEAM_ALIASES)
    df["Driver"] = df["Driver"].replace(DRIVER_ALIASES)
    return df


def recompute_rolling_avg(master_df: pd.DataFrame) -> pd.DataFrame:
    df = master_df.copy()
    df = df.sort_values(["Driver", "Year", "Round"]).reset_index(drop=True)
    df["Driver_Form_Avg"] = (
        df.groupby("Driver")["FinishPos"]
        .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
    )
    df["Driver_Form_Avg"] = df["Driver_Form_Avg"].fillna(20.0)
    df = df.sort_values(["Year", "Round", "FinishPos"]).reset_index(drop=True)
    return df


def update_mappings(new_df: pd.DataFrame, mappings_path: str) -> tuple[list[str], list[str]]:
    """Append unseen drivers/teams to mappings.json. Returns (new_drivers, new_teams)."""
    with open(mappings_path, "r") as f:
        mappings = json.load(f)

    new_drivers, new_teams = [], []

    next_driver_id = max(mappings["drivers"].values()) + 1
    for driver in sorted(new_df["Driver"].unique()):
        if driver not in mappings["drivers"]:
            mappings["drivers"][driver] = next_driver_id
            next_driver_id += 1
            new_drivers.append(driver)

    next_team_id = max(mappings["teams"].values()) + 1
    for team in sorted(new_df["Team"].unique()):
        if team not in mappings["teams"]:
            mappings["teams"][team] = next_team_id
            next_team_id += 1
            new_teams.append(team)

    if new_drivers or new_teams:
        with open(mappings_path, "w") as f:
            json.dump(mappings, f, indent=4)

    return new_drivers, new_teams


def clean_single_race_production(race_df):
    """
    Cleans and imputes missing session times for a single race weekend.
    Expects an unsorted or sorted dataframe of the 20 drivers in a single event.
    """
    
    df = race_df.copy()
    
   
  
    if df["QualyTime"].notna().any():
        df = df.sort_values(by="QualyTime", na_position="last").reset_index(drop=True)
    else:
        df = df.sort_values(by="GridPos").reset_index(drop=True)
    
    for col in ['AvgFPTime', 'QualyTime']:
        series = df[col].copy()

        
        known = series.dropna()
        mean_gap = known.diff().mean() if len(known) > 1 else 0.0
        if pd.isna(mean_gap):
            mean_gap = 0.0

        series = series.interpolate(method='linear')

       
        if pd.isna(series.iloc[0]):
            first_valid = series.first_valid_index()
            if first_valid is not None:
                for i in range(first_valid - 1, -1, -1):
                    series.iloc[i] = series.iloc[i + 1] - mean_gap

       
        if pd.isna(series.iloc[-1]):
            last_valid = series.last_valid_index()
            if last_valid is not None:
                for i in range(last_valid + 1, len(series)):
                    series.iloc[i] = series.iloc[i - 1] + mean_gap

        df[col] = series

    
    df['QualTimeDelta'] = df['QualyTime'] - df['QualyTime'].min()

    df['IsAccurate'] = df['IsAccurate'].fillna(False).astype(bool)
    
    return df