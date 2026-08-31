import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pickle
import os
import sys
import json
from fastapi import FastAPI,BackgroundTasks,HTTPException
from dotenv import load_dotenv
from fastapi.security import APIKeyHeader
from fastapi import Security
import io
import datetime
import time
import structlog
load_dotenv()

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
)
log = structlog.get_logger("finetune")


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)

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

from models.F1Net import F1Net
from models.F1Loss import F1Loss
from utils.dataLoader import F1NetDataset,collate
from utils.metrics import (
    setup_metrics, FINETUNE_JOBS_TOTAL, FINETUNE_DURATION_SECONDS,
    FINETUNE_SPEARMAN_PRE, FINETUNE_SPEARMAN_POST, FINETUNE_EPOCHS,
    FINETUNE_LOSS, MODEL_LOADED,
)
import mlflow
import dagshub
import numpy as np
from scipy.stats import spearmanr
import uuid

API_KEY = os.getenv("ADMIN_API_KEY")
api_key_header = APIKeyHeader(name="X-API-KEY")
MAPPINGS_PATH = os.path.join(root_path,"config","mappings.json")
MODELS_CHECKPOINT = os.path.join(root_path,"checkpoints","f1net_main.pth")
FISHER_PATH = os.path.join(root_path,"checkpoints","fisher_info.pkl")

START_TIME = datetime.datetime.now(datetime.timezone.utc)

job_store : dict[str,dict] = {}


def verify_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

def _set_status(job_id:str,**kwargs):
    job_store[job_id].update(kwargs)
    log.info("job_status", job_id=job_id[:8], **{k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool))})



def EWC(model, fixed_params, fisher_mat):
    l = 0
    for name, param in model.named_parameters():
        if name not in fisher_mat:
            log.warning("missing_fisher_param", name=name)
            continue
        if name not in fixed_params:
            log.warning("missing_fixed_param", name=name)
            continue
        l += (fisher_mat[name] * (param - fixed_params[name]).pow(2)).sum()
    return l

def calc_spearman(model, test_loader, mappings, device, num_races_to_calc=3):
    model.eval()
    id_to_driver = {v: k for k, v in mappings["drivers"].items()}
    id_to_driver[0] = "ROOKIE_BASELINE"  
    corrs = []
    with torch.no_grad():
        for idx,batch in enumerate(test_loader):
            if idx >= num_races_to_calc:
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


    return sum(corrs)/len(corrs) if corrs else 0.0


def finetune(job_id:str):
    start = time.time()
    try:
        _set_status(job_id, status="running", message="Starting finetune...")
        log.info("finetune_started", job_id=job_id[:8])
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
        if tracking_uri.startswith(("http://localhost", "http://127.0.0.1")):
            if os.getenv("DAGSHUB_TOKEN"):
                dagshub.init(repo_owner="TheGreek-god", repo_name="MLOps-DevOps-Project")
                tracking_uri = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("F1Net_Finetune")

        _set_status(job_id, message="Loading checkpoint and mappings...")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(MODELS_CHECKPOINT,map_location = device)
        
        
        with open(MAPPINGS_PATH,"r") as f:
            mappings = json.load(f)
        num_drivers = len(mappings['drivers'])
        num_teams = len(mappings["teams"])

        _set_status(job_id, message="Loading fisher matrix...")
        with open(FISHER_PATH,"rb") as f:
            fisher_mat = CPU_Unpickler(f).load()

        _set_status(job_id, message="Loading Model")
        model = F1Net(num_teams = num_teams,num_drivers=num_drivers)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        MODEL_LOADED.set(1)

        _set_status(job_id, message="Building dataset and dataloader...")
        dataset = F1NetDataset(group = "2026")
        window_size = 6
        start_idx = max(0,len(dataset)-window_size)
        final_ds = [dataset[i] for i in range(start_idx,len(dataset))]
        dl = DataLoader(final_ds,batch_size=1,shuffle=False,collate_fn=collate)

        losfn = F1Loss()
        optimizer = torch.optim.Adam(model.parameters(),lr = .003)

        pre_corrs = calc_spearman(model,dl,mappings,device,num_races_to_calc=window_size)
        FINETUNE_SPEARMAN_PRE.set(pre_corrs)
        log.info("spearman_pre", job_id=job_id[:8], spearman=round(pre_corrs, 4))

        fixed_params = {}
        for name,param in model.named_parameters():
            fixed_params[name] = param.detach().clone()
        lam = 10

        _set_status(job_id, message="Starting training loop...")
        with mlflow.start_run(run_name = "F1Net_Finetune_run"):


            EPOCHS = 10
            for epoch in range(EPOCHS):
                
                for batch in dl:
                    model.train()
                    d_id = batch["driver_id"][0].to(device)
                    t_id = batch["team_id"][0].to(device)
                    n_feat = batch["numeric_feat"][0].to(device)
                    y_true = batch["targets"][0].to(device)

                    preds = model(n_feat,t_id,d_id)
                    loss = losfn([preds],[y_true])
                    full_loss = loss + (lam/2)*EWC(model,fixed_params,fisher_mat)
                    optimizer.zero_grad()
                    full_loss.backward()
                    optimizer.step()

                FINETUNE_EPOCHS.inc()
                FINETUNE_LOSS.set(full_loss.item())
                log.info("epoch_complete", job_id=job_id[:8], epoch=epoch+1, loss=round(full_loss.item(), 4))




            _set_status(job_id, message="Logging metrics")
            post_corrs = calc_spearman(model,dl,mappings,device,num_races_to_calc=window_size)
            FINETUNE_SPEARMAN_POST.set(post_corrs)
            mlflow.log_metric("Pre_Spearman_Corr",pre_corrs)
            mlflow.log_metric("Post_Spearman_Corr",post_corrs)
            mlflow.log_param("EWC_Lam",lam)
            mlflow.log_artifact(MAPPINGS_PATH,artifact_path = "model_metadata")
            mlflow.log_artifact(FISHER_PATH,artifact_path="model_metadata")
            mlflow.pytorch.log_model(
                pytorch_model=model,
                name = "f1net_model",
                registered_model_name="F1NET",
                serialization_format="pickle"
            )

            client = mlflow.MlflowClient()
            latest_version = client.get_latest_versions("F1NET")[-1].version
            client.set_registered_model_alias("F1NET", "prod", version=latest_version)


            state_payload = {
                'model_state_dict': model.state_dict(),
                'mappings':mappings,
            }
            torch.save(state_payload,MODELS_CHECKPOINT)
            _set_status(job_id, message="Checkpoint Saved and model logged")

            elapsed = time.time() - start
            FINETUNE_DURATION_SECONDS.observe(elapsed)
            FINETUNE_JOBS_TOTAL.labels(status="success").inc()
            _set_status(job_id, status="done", message="Finetune complete.")
            log.info("finetune_complete", job_id=job_id[:8], spearman_pre=round(pre_corrs, 4), spearman_post=round(post_corrs, 4), elapsed=round(elapsed, 2))
    except Exception as e:
        FINETUNE_JOBS_TOTAL.labels(status="error").inc()
        log.error("finetune_failed", job_id=job_id[:8], error=str(e))
        _set_status(job_id, status="error", message=str(e))





app = FastAPI(
    title = "Finetune model API",
    description="Finetunes the model on the last couple of races"

)

setup_metrics(app, "finetune")


@app.get("/health")
def health():
    uptime = (datetime.datetime.now(datetime.timezone.utc) - START_TIME).total_seconds()
    return {"status": "healthy", "service": "finetune", "uptime_seconds": round(uptime, 1)}

@app.post("/finetune")
def finetune_endpoint(background_tasks:  BackgroundTasks,_=Security(verify_key)):
    job_id = str(uuid.uuid4())
    job_store[job_id] = {"status":"queued","message":"Job Queued"}
    log.info("finetune_queued", job_id=job_id[:8])
    background_tasks.add_task(finetune,job_id)
    return {"job_id":job_id,"status":"queued"}

@app.get("/finetune/status/{job_id}")
def finetune_status(job_id: str):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
