import torch
import pandas as pd
import os
import json
from torch.utils.data import Dataset,DataLoader
import math

def find_project_root():
    current = os.path.dirname(os.path.abspath(__file__))
    while current != os.path.dirname(current):
        if os.path.exists(os.path.join(current, "requirements.txt")):
            return current
        current = os.path.dirname(current)
    raise FileNotFoundError("Could not find project root")

root_path = find_project_root()
datapath = os.path.join(root_path, "data", "Complete_Driver_Data.csv")
mappingpath = os.path.join(root_path, "config", "mappings.json")

class F1NetDataset(Dataset):
    def __init__(self,datapath = datapath,mappingpath = mappingpath,years = None,train_size = .8,group = 'train'):
        super().__init__()

        self.df = pd.read_csv(datapath)
        self.train_size = train_size
        self.group = group


        if years is not None:
            self.df = self.df[self.df["Year"].isin(years)].reset_index(
                drop=True
            )
        
    

        with open(mappingpath, "r") as f:
            self.mappings = json.load(f)

        exclude_cols = [
            "Year",
            "Round",
            "Driver",
            "Team",
            "IsAccurate",
            "FinishPos",
        ]
        self.feature_cols = [
            col for col in self.df.columns if col not in exclude_cols
        ]


        #map drivers and teams


        self.df["Driver_Id"] = (self.df["Driver"].map(self.mappings["drivers"]).fillna(0).astype(int))
        self.df["Team_Id"] = (self.df["Team"].map(self.mappings["teams"]).fillna(0).astype(int))


        unified = self.df[(self.df["Year"]>=2022) & (self.df["Year"]<2026)].copy()
        df_2026 = self.df[(self.df["Year"] == 2026)]

        historical_races = sorted(list(unified.groupby(["Year", "Round"])), key=lambda x: x[0])
        races_2026 = sorted(list(df_2026.groupby(["Year", "Round"])), key=lambda x: x[0])

        if not historical_races:
            split_idx = int(len(races_2026) * self.train_size)
            split_idx = max(1, min(split_idx, len(races_2026) - 1))
            train_races = races_2026[:split_idx]
            test_races = races_2026[split_idx:]
        else:
            split_idx = int(len(historical_races)*self.train_size)
            train_races = historical_races[:-2]
            test_races = historical_races[-2:]

        if self.group == 'train':
            self.race_groups = train_races
        elif self.group == 'test':
            self.race_groups = test_races
        elif self.group == '2026':
            self.race_groups = races_2026
        else:
            raise ValueError("Group must be either 'train', 'test', or '2026'")
        

        
        

    def __len__(self):
        return len(self.race_groups)
    

    def __getitem__(self,idx):
        (year,round_num),race_df = self.race_groups[idx]

        race_df = race_df.sort_values("FinishPos").copy()


        if(self.group == 'train'):
            midfield_df = race_df[(race_df["FinishPos"]>=12)&(race_df["FinishPos"]<=20)]
            baseline_feats = None
            baseline_target = None
            if(len(midfield_df)>0):
                baseline_feats = midfield_df[self.feature_cols].mean()
                baseline_target = float(math.ceil(midfield_df["FinishPos"].mean()))
            else:
                baseline_feats = race_df[self.feature_cols].mean()
                baseline_target = 16.0
            

            rookie_row = pd.DataFrame([{col:baseline_feats[col] for col in self.feature_cols}])
            rookie_row["Driver_Id"] = 0
            rookie_row["Team_Id"] = 0
            rookie_row["FinishPos"] = baseline_target

            race_df = pd.concat([race_df,rookie_row],ignore_index = True)

        driver_ids = torch.tensor(race_df["Driver_Id"].values,dtype = torch.long)
        team_ids = torch.tensor(race_df["Team_Id"].values,dtype = torch.long)

        targets = torch.tensor(race_df["FinishPos"].values,dtype = torch.float32)

        numeric_features = torch.tensor(race_df[self.feature_cols].values,dtype = torch.float32)

        num_drivers = len(race_df)
        targets = (num_drivers+1) -  targets
        return {
            "driver_id":driver_ids,
            "team_id":team_ids,
            "numeric_feat":numeric_features,
            "targets":targets
        }
    



def collate(batch):
    return {
        "driver_id" : [item["driver_id"] for item in batch],
        "team_id" : [item["team_id"] for item in batch],
        "numeric_feat": [item["numeric_feat"] for item in batch],
        "targets": [item["targets"] for item in batch],
    }