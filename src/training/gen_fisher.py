import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import json
import sys
import mlflow
import pickle

src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
project_root = os.path.dirname(src_path)


if src_path not in sys.path:
    sys.path.insert(0, src_path)

from models.F1Net import F1Net
from models.F1Loss import F1Loss
from utils.dataLoader import F1NetDataset,collate
from dotenv import load_dotenv
load_dotenv()


def calc_fisher(model,loader,lossfn,device):
    model.eval()
    fisher = {}
    for name,param in model.named_parameters():
        fisher[name] = torch.zeros_like(param)
    

    for batch in loader:
        model.zero_grad()
        d_id = batch['driver_id'][0].to(device)
        t_id = batch['team_id'][0].to(device)
        n_feat = batch['numeric_feat'][0].to(device)
        y_true = batch['targets'][0].to(device)
        preds = model(n_feat,t_id,d_id)
        loss = lossfn([preds],[y_true])
        loss.backward()

        for name,param in model.named_parameters():
            if(param.grad is not None):
                fisher[name] += param.grad.data.pow(2)
    
    for name in fisher:
        fisher[name] /= len(loader)
    return fisher



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
   
    print(mlflow.get_tracking_uri())
    print(mlflow.MlflowClient().get_registered_model("F1NET"))


    with mlflow.start_run(run_name = 'fisher_computation'):
    
        checkpoint_dir = os.path.join(project_root, "checkpoints", "f1net_main.pth")
        fisher_dir = os.path.join(project_root, "checkpoints", "fisher_info.pkl")
        mappings_path = os.path.join(project_root, "config", "mappings.json")

        with open(mappings_path, 'r') as f:
            mappings = json.load(f)

        num_drivers = len(mappings['drivers'].keys())
        num_teams = len(mappings['teams'].keys())

        train_set = F1NetDataset(group = 'train')
        train_Loader =  DataLoader(train_set,batch_size = 4,shuffle = False,collate_fn=collate)

        lossfn = F1Loss()


        model_uri = "models:/F1NET/latest"
        model = mlflow.pytorch.load_model(model_uri)
        model.to(device)
        model.eval()
        print("Model Loaded")

        fisher = calc_fisher(model,train_Loader,lossfn,device)

        print("Fisher Matrix Computed")
        
        state_payload = {
            'model_state_dict': model.state_dict(),
            'mappings':mappings,
        }

        torch.save(state_payload,checkpoint_dir)
        print(f"\nModel checkpoint saved to :{checkpoint_dir}")
        with open(fisher_dir,"wb") as f:
            pickle.dump(fisher,f)


        


        mlflow.log_artifact(mappings_path, artifact_path="model_metadata")
        mlflow.log_artifact(fisher_dir,artifact_path = "model_metadata")
        print("Fisher Matrix Logged")

        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="f1net_model",
            registered_model_name="F1NET"

        )

        print("\nModel Logged and Saved")

