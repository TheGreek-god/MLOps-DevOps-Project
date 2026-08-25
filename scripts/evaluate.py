"""
DVC pipeline stage: evaluate
Loads trained checkpoint and computes Spearman correlation on test + 2026 data.
Saves results to metrics.json for DVC tracking.
"""
import os
import sys
import json
import yaml
import torch
import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from models.F1Net import F1Net
from models.F1Loss import F1Loss
from utils.dataLoader import F1NetDataset, collate
from torch.utils.data import DataLoader

CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
METRICS_PATH = os.path.join(ROOT, "metrics.json")

with open(os.path.join(ROOT, "params.yaml")) as f:
    params = yaml.safe_load(f)

eval_params = params["evaluate"]
train_params = params["train"]
device = torch.device(train_params["device"])

checkpoint_path = os.path.join(CHECKPOINT_DIR, "f1net_main.pth")
if not os.path.exists(checkpoint_path):
    print(f"ERROR: Checkpoint not found at {checkpoint_path}")
    sys.exit(1)

checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
mappings = checkpoint["mappings"]
num_drivers = len(mappings["drivers"])
num_teams = len(mappings["teams"])

model = F1Net(
    num_teams=num_teams,
    num_drivers=num_drivers,
    embedding_dim=train_params["embedding_dim"],
)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()
print(f"Loaded checkpoint: {checkpoint_path}")
print(f"  Drivers: {num_drivers}, Teams: {num_teams}")

lossfn = F1Loss()

id_to_driver = {v: k for k, v in mappings["drivers"].items()}
id_to_driver[0] = "ROOKIE_BASELINE"


def evaluate_split(model, loader, device, lossfn, split_name):
    corrs = []
    losses = []
    with torch.no_grad():
        for batch in loader:
            d_ids = batch["driver_id"][0].to(device)
            t_ids = batch["team_id"][0].to(device)
            n_feat = batch["numeric_feat"][0].to(device)
            y_true = batch["targets"][0].to(device)

            preds = model(n_feat, t_ids, d_ids)
            loss = lossfn([preds], [y_true])
            if not torch.isnan(loss):
                losses.append(loss.item())

            true_ranks = y_true.cpu().tolist()
            pred_scores = preds.squeeze(-1).cpu().tolist()
            corr, _ = spearmanr(pred_scores, true_ranks)
            if not np.isnan(corr):
                corrs.append(corr)

    avg_corr = np.mean(corrs) if corrs else 0.0
    avg_loss = np.mean(losses) if losses else 0.0
    print(f"  {split_name}: Spearman={avg_corr:.4f}, Loss={avg_loss:.4f}, Races={len(corrs)}")
    return avg_corr, avg_loss


test_set = F1NetDataset(group="test")
ds26 = F1NetDataset(group="2026")

test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate) if len(test_set) > 0 else None
loader_2026 = DataLoader(ds26, batch_size=1, shuffle=False, collate_fn=collate) if len(ds26) > 0 else None

print("Evaluating model...")
metrics = {}

if test_loader:
    test_corr, test_loss = evaluate_split(model, test_loader, device, lossfn, "Test")
    metrics["test_spearman_corr"] = round(test_corr, 4)
    metrics["test_loss"] = round(test_loss, 4)
else:
    print("  WARNING: No test data available")
    metrics["test_spearman_corr"] = 0.0
    metrics["test_loss"] = 0.0

if loader_2026:
    corr_2026, loss_2026 = evaluate_split(model, loader_2026, device, lossfn, "2026")
    metrics["spearman_corr_2026"] = round(corr_2026, 4)
    metrics["loss_2026"] = round(loss_2026, 4)
else:
    print("  WARNING: No 2026 data available")
    metrics["spearman_corr_2026"] = 0.0
    metrics["loss_2026"] = 0.0

metrics["model_drivers"] = num_drivers
metrics["model_teams"] = num_teams
metrics["checkpoint"] = checkpoint_path

with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"\nMetrics saved: {METRICS_PATH}")
print(json.dumps(metrics, indent=2))
