import torch
import mlflow
import dagshub
from fastapi import FastAPI,HTTPException
import os
import pandas as pd
import json
from pydantic import BaseModel
import sys
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import  numpy as np
import math
load_dotenv()


def find_project_root():
    current = os.path.dirname(os.path.abspath(__file__))
    while current != os.path.dirname(current):
        if os.path.exists(os.path.join(current, "requirements.txt")):
            return current
        current = os.path.dirname(current)
    raise FileNotFoundError("Could not find project root")

root_path = find_project_root()
src_path = os.path.join(root_path, "src")
if root_path not in sys.path:
    sys.path.insert(0, root_path)
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from models.F1Net import F1Net
from utils.dataLoader import F1NetDataset


MAPPINGS_PATH = os.path.join(root_path, "config","mappings.json")
MODEL_CHECKPOINT_PATH = os.path.join(root_path, "checkpoints","f1net_main.pth")
app_state = {}

with open(MAPPINGS_PATH, "r") as f:
        mappings = json.load(f)

class RoundRequest(BaseModel):
    year:int
    round_num :int

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app_state["device"] = device
   
    app_state["mappings"] = mappings

    app_state["id_to_driver"] = {v: k for k, v in mappings['drivers'].items()}
    app_state["id_to_driver"][0] = "Unknown Driver"
    app_state["id_to_team"] = {v: k for k, v in mappings['teams'].items()}
    app_state["id_to_team"][0] = "Unknown Team"

    

    try:
        
        checkpoint_path = MODEL_CHECKPOINT_PATH

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        drivers = len(mappings['drivers'].keys())
        teams = len(mappings['teams'].keys())
        model = F1Net(num_teams=teams,num_drivers=drivers)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        app_state["model"] = model
        app_state["device"] = device
        print(f"Using Local checkpoint at {checkpoint_path}")


    except Exception as e:
        print(f"Error Loading model: {e}")


        try:
            dagshub.auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN"))
            dagshub.init(repo_owner='pranav070904', repo_name='F1Net-V2', mlflow=True)
            print("Loading model artifacts.")
            model_uri = "models:/F1NET/latest"
            model = mlflow.pytorch.load_model(model_uri)
            model.to(device)
            model.eval()
            app_state["model"] = model
            print("Model Loaded")

            

        except Exception as e:
             print("Failed to load model from both MLflow and local checkpoint.")
             raise e
        

    try:
        
        if "mappings" not in app_state:
            with open(MAPPINGS_PATH, "r") as f:
                app_state["mappings"] = json.load(f)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise e
    
    yield

    print("Shutting down FastAPI Server Engine process...")
    app_state.clear()


app = FastAPI(
    title = "F1Net API",
    description = "API for F1Net model predictions",
    lifespan = lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONT_END_URL","http://localhost:5173")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
def predict(request:RoundRequest):
    year = request.year
    round_num = request.round_num

   
    if(year<2026):
        raise HTTPException(status_code=404,detail="Selected Race is Invalid")
    model = app_state.get("model")
    device = app_state.get("device")
    Dataset = app_state["Dataset"] = F1NetDataset(group = '2026')
    mappingss = app_state.get("mappings")
    if model is None or device is None or Dataset is None:
        raise HTTPException(status_code=500, detail="Model or Device not loaded")
    

    total_races = len(Dataset)
    if(total_races == 0):
        raise HTTPException(status_code=404, detail="Dataset not found")
    


    race_idx = None
    for i in range(len(Dataset)):
        (y,r),_ = Dataset.race_groups[i]
        if((y,r) == (year,round_num)):
            race_idx = i
            break

    if(race_idx is None):
        raise HTTPException(status_code=404, detail="Race Data not available")

    

    last_race_df = Dataset[race_idx]

    d_ids = last_race_df["driver_id"].to(device)
    t_ids = last_race_df["team_id"].to(device)
    n_feat = last_race_df["numeric_feat"].to(device)

    y_true = last_race_df["targets"]
    y_true = y_true.cpu().tolist() if y_true is not None else None


    with torch.no_grad():
        preds = model(n_feat, t_ids, d_ids)
    

    id_to_driver = app_state.get("id_to_driver")
    id_to_team = app_state.get("id_to_team")
    driver_names = [id_to_driver.get(int(d_id.item()),f"Unknown Driver {d_id.item()}") for d_id in d_ids]
    team_names = [id_to_team.get(int(t_id.item()),f"Unknow Team {t_id.item()}") for t_id in t_ids]
    grid = []

    num_drivers = len(d_ids)
    for i in range(len(driver_names)):
        true_pos = int((num_drivers + 1) - int(y_true[i])) if y_true is not None and not pd.isna(y_true[i]) else None
        grid.append(
            {
                "driver": driver_names[i],
                "predicted_score": float(preds[i].item()),
                "true_position": true_pos,
                "team": team_names[i]
            }
        )

    predicted_order = sorted(grid,key = lambda x:x["predicted_score"],reverse = True)

    
    response = {
        "race_idx": race_idx+1,
        "total_cars": len(driver_names),
        "ground_truth_available": y_true is not None and not all(pd.isna(v) for v in y_true),
        "predictions": [
            {
                "predicted_p": idx + 1,
                "driver": item["driver"],
                "actual_p": item["true_position"],
                "model_raw_score": item["predicted_score"],
                "exact_match_hit": item["true_position"] == (idx + 1) if item["true_position"] is not None else False,
                "team":item["team"]
            }
            for idx, item in enumerate(predicted_order)
        ]
    }


    return response