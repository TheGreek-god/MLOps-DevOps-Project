import torch
import torch.nn as nn
import torch.functional as F
from torch.utils.data import DataLoader
import os
import json
import sys
from hyperopt import STATUS_OK,Trials,fmin,hp,tpe
import mlflow
import dagshub
from scipy.stats import spearmanr
import numpy as np
from functools import partial
import pickle


src_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
project_root = os.path.dirname(src_path)
mappings_path = os.path.join(project_root, "config", "mappings.json")
checkpoint_dir = os.path.join(project_root, "checkpoints", "f1net_main.pth")
fisher_dir = os.path.join(project_root, "checkpoints", "fisher_info.pkl")


if src_path not in sys.path:
    sys.path.insert(0, src_path)

from models.F1Net import F1Net
from models.F1Loss import F1Loss
from utils.dataLoader import F1NetDataset,collate
from dotenv import load_dotenv
load_dotenv()



def calculate_fisher(model,loader,device,lossfn):
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
        fisher[name] /=len(loader)
    
    return fisher

def print_race_ranking_comparison(model, test_loader, mappings, device, num_races_to_show=2):
    model.eval()
    
    
    id_to_driver = {v: k for k, v in mappings["drivers"].items()}
    id_to_driver[0] = "ROOKIE_BASELINE" 
    
    print("\n" + "="*60)
    print("EVALUATION (REAL VS. PREDICTED)")
    print("="*60)
    corrs = []
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            if idx >= num_races_to_show:
                break
                
           
            d_ids = batch["driver_id"][0].to(device)
            t_ids = batch["team_id"][0].to(device)
            n_feat = batch["numeric_feat"][0].to(device)
            y_true = batch["targets"][0].cpu().tolist()
            
           
            predictions = model(n_feat, t_ids, d_ids).squeeze(-1).cpu().tolist()
            
           
            driver_names = [id_to_driver.get(int(d_id.item()), "UNKNOWN") for d_id in d_ids]
            
          
            race_grid = []
            for name, true_pos, pred_score in zip(driver_names, y_true, predictions):
                race_grid.append({
                    "name": name,
                    "true_pos": int(true_pos),
                    "pred_score": pred_score
                })
            
           
            predicted_order = sorted(race_grid, key=lambda x: x["pred_score"],reverse=True)
            actual_order = sorted(race_grid, key=lambda x: x["true_pos"],reverse=True)

            true_ranks = [x["true_pos"] for x in race_grid]
            predicted_scores = [x["pred_score"] for x in race_grid]

            
            corr,_ = spearmanr(predicted_scores,true_ranks)

            if not np.isnan(corr):
                corrs.append(corr)
            
            # --- 5. PRINT THE GRID COMPARISON ---
            print(f"\n🏎️  Race Sample {idx + 1} Evaluation:")
            print(f"{'P.':<4} | {'GROUND TRUTH RESULTS':<25} | {'MODEL PREDICTED ORDER':<25}")
            print("-" * 62)
            
            for p in range(len(race_grid)):
                actual_name = actual_order[p]["name"] if p < len(actual_order) else "DNF/OUT"
                predicted_name = predicted_order[p]["name"] if p < len(predicted_order) else "DNF/OUT"
                
                # Highlight if the model got the exact position dead right
                match_marker = "🎯" if actual_name == predicted_name else "  "
                
                print(f"P{p+1:<2} | {actual_name:<25} | {predicted_name:<25} {match_marker}")
            print("="*60)

            
    
    return sum(corrs)/len(corrs) if corrs else 0.0


def train_stage_2(model,loader,lossfn,device,epochs,lr,mappings):
    print(f"\n Finetuning Phase on 2026 Data ({len(loader)} batches)")

    driver_count = len(mappings['drivers'])
    team_count = len(mappings['teams'])
    arvid_id = mappings['drivers']["Arvid Lindblad"]
    cadillac_id = mappings['teams']["Cadillac"]


    model.driver_embedding.weight.data[arvid_id] = model.driver_embedding.weight.data[0].clone()
    model.team_embedding.weight.data[cadillac_id] = model.team_embedding.weight.data[0].clone()

    model.driver_embedding.requires_grad = True
    model.team_embedding.requires_grad = True

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params,lr = lr)

    total_train_losses = []

    for epoch in range(epochs):
        model.train()
        batch_train_losses = []
        for batch in loader:
            d_ids = batch["driver_id"][0].to(device)
            t_ids = batch["team_id"][0].to(device)  
            n_feat = batch["numeric_feat"][0].to(device)
            y_true = batch["targets"][0].to(device)

            preds = model(n_feat,t_ids,d_ids)
            loss = lossfn([preds],[y_true])
            batch_train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()


            if(model.driver_embedding.weight.grad is not None):
                grad_mask = torch.zeros_like(model.driver_embedding.weight.grad)
                grad_mask[2] = 1.0
                model.driver_embedding.weight.grad *= grad_mask
            
            if(model.team_embedding.weight.grad is not None):
                grad_mask = torch.zeros_like(model.team_embedding.weight.grad)
                grad_mask[cadillac_id] = 1.0
                model.team_embedding.weight.grad *= grad_mask

            optimizer.step()
        avg_batch_train_loss = sum(batch_train_losses)/len(batch_train_losses)
        total_train_losses.append(avg_batch_train_loss)

        print(f"Epoch [{epoch+1:02d}/{epochs} | Train Loss: {avg_batch_train_loss:.4f}]")
    
    mlflow.log_metric(
        'finetune_train_loss',sum(total_train_losses)/len(total_train_losses)
    )

    return model
        

    
def train_model():
    # temperature = float(params["temperature"])

    train_set = F1NetDataset(group = "train")
    test_set = F1NetDataset(group = "test")
    ds26 = F1NetDataset(group = '2026')

    with open(mappings_path, "r") as f:
            mappings = json.load(f)
    drivers = len(mappings['drivers'].keys())
    teams = len(mappings['teams'].keys())


    EPOCHS_P1 = 25 #pretraining
    EPOCHS_P2 = 25 #finetuning
    LR_P1 = .001
    LR_P2 = .003

    model = F1Net(num_teams=teams,num_drivers=drivers)
    lossfn = F1Loss()
    optimizer = torch.optim.Adam(model.parameters(),lr = LR_P1)


    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    model.to(device)

    train_Loader =  DataLoader(train_set,batch_size = 4,shuffle = False,collate_fn=collate)
    test_Loader = DataLoader(test_set,batch_size = 1,shuffle = False,collate_fn=collate)
    loader_2026 = DataLoader(ds26,batch_size = 1,shuffle = False,collate_fn=collate)

    print(f"\n Pretraining Phase ({len(train_Loader)} batches)")

    total_train_losses = []
    total_val_losses = []
    for epoch in range(EPOCHS_P1):
        model.train()
        
        batch_train_losses = []
        batch_val_losses = []

    ###########Training Step#####################

        for batch in train_Loader:
            driver_ids = [d.to(device) for d in batch["driver_id"]]
            team_ids = [t.to(device) for t in batch["team_id"]]
            numeric_feat = [n.to(device) for n in batch["numeric_feat"]]
            targets = [n.to(device) for n in batch["targets"]]


            batch_preds  = []
            for i in range(len(driver_ids)):
                preds = model(numeric_feat[i],team_ids[i],driver_ids[i])
                batch_preds.append(preds)
            

            loss = lossfn(batch_preds,targets)
            batch_train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            

        ##############Validation################################
        #print(f"Total Train Loss: {total_train_loss}")
        model.eval()
        
        with torch.no_grad():
             for batch in test_Loader:
                  d_id = batch["driver_id"][0].to(device)
                  t_id = batch["team_id"][0].to(device)
                  n_feat = batch["numeric_feat"][0].to(device)
                  y_true = batch["targets"][0].to(device)

                
                
                  preds = model(n_feat,t_id,d_id)
                  
                  
                  loss = lossfn([preds],[y_true])
                  batch_val_losses.append(loss.item())
        
        avg_batch_train_loss = sum(batch_train_losses) / len(batch_train_losses)
        avg_batch_val_loss = sum(batch_val_losses) / len(batch_val_losses)
        total_train_losses.append(avg_batch_train_loss)
        total_val_losses.append(avg_batch_val_loss)
        print(f"Epoch [{epoch+1:02d}/{EPOCHS_P1} | Train Loss: {avg_batch_train_loss:.4f} | Test Loss: {avg_batch_val_loss:.4f}]")

    avg_cor = print_race_ranking_comparison(model, test_Loader, mappings, device, num_races_to_show=2)
    avg_train_loss = sum(total_train_losses)/len(total_train_losses)
    avg_test_loss = sum(total_val_losses)/len(total_val_losses)
    mlflow.log_metric(
        "avg_spearman_corr",avg_cor
    )
    mlflow.log_metric(
        'train_loss',avg_train_loss
    )
    mlflow.log_metric(
        'test_loss',avg_test_loss
    )
    
    
    model = train_stage_2(model,loader_2026,lossfn,device,epochs = EPOCHS_P2,lr = LR_P2,mappings = mappings)
    avg_cor_2026 = print_race_ranking_comparison(model, loader_2026, mappings, device, num_races_to_show=6)
    mlflow.log_metric(
        "avg_spearman_corr_2026",avg_cor_2026

    )

    fisher = calculate_fisher(model,loader_2026,device,lossfn)

    return model,mappings,fisher




# def objective(params):
#     with mlflow.start_run(nested=True):
#         avg_cor,final_val_loss = train_model(params)

#         return {
#             "loss" : -avg_cor,
#             "status" : STATUS_OK,
#             "attachments": {'val_loss':final_val_loss}
#         }
    

if __name__ == "__main__":
    dagshub.auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN"))
    dagshub.init(repo_owner='pranav070904', repo_name='F1Net-V2', mlflow=True)
    mlflow.set_experiment("F1Net_Fisher")

    # space = {
    #     'temperature': hp.uniform('temperature', 0.5, 10.0)
    # }
    # trials = Trials()
    # tpe_algo = partial(
    #     tpe.suggest,
    #     n_startup_jobs = 8,
    #     gamma = 0.25
    # )
    # best = fmin(
    #     fn = objective,
    #     space = space,
    #     algo = tpe_algo,
    #     max_evals = 20,
    #     trials = trials
    # )
    # print(f"Best param selection index: {best}")

    with mlflow.start_run(run_name="F1Net_with_Fisher_Gen"):
        model,mappings,fisher = train_model()
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

        mlflow.pytorch.log_model(
            pytorch_model=model,
            name="f1net_model",
            registered_model_name="F1NET"

        )

        print("\nModel Logged and Saved")


