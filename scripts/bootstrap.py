"""
Bootstrap script for lab use.
Fetches minimal F1 data from FastF1, trains model, generates checkpoints.

FastF1 has a 500 calls/hour rate limit. Each session.load() makes ~7-10 API calls.
Strategy:
  - Only load Race + Qualifying (skip FP to stay under rate limit)
  - AvgFPTime will be NaN and imputed by the data cleaner
  - 30s pause between races, 90s between years
  - If rate-limited, the script saves partial progress — just re-run it later

Fetches 2 races/year for 2022-2025 + 1 race for 2026 = 9 races total (~180 API calls).
"""
import fastf1
from fastf1.events import get_event_schedule
import os
import sys
import time
import pandas as pd
import json
import pickle
import subprocess
from dotenv import load_dotenv

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

CACHE = os.path.join(ROOT, "cache_folder")
DATA_DIR = os.path.join(ROOT, "data")
CONFIG_DIR = os.path.join(ROOT, "config")
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
PROGRESS_FILE = os.path.join(ROOT, "scripts", ".bootstrap_progress.json")

os.makedirs(CACHE, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

fastf1.Cache.enable_cache(CACHE)

if os.getenv("DAGSHUB_TOKEN"):
    import dagshub
    dagshub.init(repo_owner="TheGreek-god", repo_name="MLOps-DevOps-Project")
    print("DagsHub initialized.")
else:
    print("WARNING: DAGSHUB_TOKEN not set. Skipping DVC push / MLflow remote.")

RACES = [
    (2026, 1), (2026, 2), (2026, 3), (2026, 4), (2026, 5),
    (2026, 6), (2026, 7), (2026, 8), (2026, 9), (2026, 10),
    (2026, 11), (2026, 12),
]

PAUSE_BETWEEN_RACES = 30
PAUSE_BETWEEN_YEARS = 90


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed": [], "data_rows": []}


def save_progress(progress):
    import numpy as _np

    class _Encoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (_np.bool_,)):
                return bool(o)
            if isinstance(o, (_np.integer,)):
                return int(o)
            if isinstance(o, (_np.floating,)):
                return None if _np.isnan(o) else float(o)
            return super().default(o)

    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, cls=_Encoder)


def load_session_safe(year, round_num, session_type, max_attempts=3, base_wait=20):
    session = fastf1.get_session(year, round_num, session_type)
    for attempt in range(max_attempts):
        try:
            session.load()
            return session
        except fastf1.exceptions.RateLimitExceededError:
            wait = 120
            print(f"      RATE LIMITED. Waiting {wait}s before retry...")
            time.sleep(wait)
        except Exception as e:
            if attempt < max_attempts - 1:
                wait = base_wait * (attempt + 1)
                print(f"      Retry {attempt+1}/{max_attempts} for {session_type} (waiting {wait}s)...")
                time.sleep(wait)
            else:
                raise e
    return None


EVENT_NAMES = {
    (2022, 1): "Bahrain Grand Prix", (2022, 2): "Saudi Arabian Grand Prix",
    (2023, 1): "Bahrain Grand Prix", (2023, 2): "Saudi Arabian Grand Prix",
    (2024, 1): "Bahrain Grand Prix", (2024, 2): "Saudi Arabian Grand Prix",
    (2025, 1): "Australian Grand Prix", (2025, 2): "Chinese Grand Prix",
    (2026, 1): "Australian Grand Prix",
}


def fetch_race(year, round_num):
    event_name = EVENT_NAMES.get((year, round_num), f"Round {round_num}")
    print(f"  {year} R{round_num}: {event_name}...")

    sessionR = load_session_safe(year, round_num, "R")
    if sessionR is None:
        print(f"    SKIP: Could not load Race session")
        return []
    time.sleep(3)

    sessionQ = load_session_safe(year, round_num, "Q")
    if sessionQ is None:
        print(f"    SKIP: Could not load Qualifying session")
        return []
    quali_results = sessionQ.results.set_index("DriverNumber")
    time.sleep(3)

    has_laps = False
    fastest_laps = None
    try:
        q_laps = sessionQ.laps.pick_not_deleted()
        if isinstance(q_laps, pd.Series):
            q_laps = q_laps.to_frame().T
        q_laps = q_laps[q_laps["LapTime"].notna()]
        if len(q_laps) > 0:
            fastest_laps = q_laps.loc[q_laps.groupby("Driver")["LapTime"].idxmin()].copy()
            pole_time = fastest_laps["LapTime"].min()
            fastest_laps["LapDelta"] = fastest_laps["LapTime"].apply(lambda x: x - pole_time).dt.total_seconds()
            fastest_laps["LapTime"] = fastest_laps["LapTime"].dt.total_seconds()
            has_laps = True
    except Exception:
        pass

    pole_time_from_results = None
    if not has_laps:
        q_cols = [c for c in ["Q1", "Q2", "Q3"] if c in quali_results.columns]
        if q_cols:
            best_q_times = quali_results[q_cols].min(axis=1).dropna()
            if len(best_q_times) > 0:
                pole_time_from_results = best_q_times.min()

    results = sessionR.results.set_index("DriverNumber")
    rows = []
    for drv in results.index:
        row = results.loc[drv]
        driver_code = row["Abbreviation"]
        q_position = quali_results.loc[drv, "Position"] if drv in quali_results.index else None
        grid_pos = row["GridPosition"] if (pd.notna(row["GridPosition"]) and row["GridPosition"] != 0 and row["GridPosition"] != -1) else q_position

        if has_laps and fastest_laps is not None:
            q_data = fastest_laps[fastest_laps["Driver"] == driver_code]
            if not q_data.empty:
                qualy_delta = q_data.iloc[0]["LapDelta"]
                qualy_time = q_data.iloc[0]["LapTime"]
                is_accurate = q_data.iloc[0]["IsAccurate"]
            else:
                qualy_delta = None
                qualy_time = None
                is_accurate = None
        elif pole_time_from_results is not None and drv in quali_results.index:
            q_cols = [c for c in ["Q1", "Q2", "Q3"] if c in quali_results.columns]
            if q_cols:
                q_row = quali_results.loc[drv, q_cols]
                q_best_td = q_row.dropna().min() if len(q_row.dropna()) > 0 else None
                if q_best_td is not None and pd.notna(q_best_td):
                    qualy_time = q_best_td.total_seconds()
                    qualy_delta = round(qualy_time - pole_time_from_results.total_seconds(), 3)
                    is_accurate = True
                else:
                    qualy_time = None
                    qualy_delta = None
                    is_accurate = None
            else:
                qualy_time = None
                qualy_delta = None
                is_accurate = None
        else:
            qualy_delta = None
            qualy_time = None
            is_accurate = None

        rows.append({
            "Year": year,
            "Round": round_num,
            "Driver": row["FullName"],
            "Team": row["TeamName"],
            "AvgFPTime": None,
            "QualyTime": qualy_time,
            "QualTimeDelta": qualy_delta,
            "GridPos": grid_pos,
            "FinishPos": row["Position"],
            "IsAccurate": is_accurate,
        })

    print(f"    OK ({len(results)} drivers)")
    return rows


# --- Phase 1: Fetch data (resume-aware) ---
print("=" * 60)
print("Phase 1: Fetching race data from FastF1")
print("=" * 60)

progress = load_progress()
all_data = progress["data_rows"]

completed_keys = set(progress["completed"])
remaining = [(y, r) for y, r in RACES if f"{y}_{r}" not in completed_keys]

if not remaining:
    print("All races already fetched (from previous run). Skipping fetch.")
else:
    print(f"  {len(remaining)} races to fetch ({len(completed_keys)} already cached)")

for i, (year, round_num) in enumerate(remaining):
    try:
        rows = fetch_race(year, round_num)
        all_data.extend(rows)
        progress["completed"].append(f"{year}_{round_num}")
        progress["data_rows"] = all_data
        save_progress(progress)
    except Exception as e:
        print(f"    ERROR: {e}")
        save_progress(progress)

    if i < len(remaining) - 1:
        next_year = remaining[i + 1][0]
        if next_year != year:
            print(f"  --- Pausing {PAUSE_BETWEEN_YEARS}s between years ---")
            time.sleep(PAUSE_BETWEEN_YEARS)
        else:
            print(f"  --- Pausing {PAUSE_BETWEEN_RACES}s between races ---")
            time.sleep(PAUSE_BETWEEN_RACES)

print(f"\nTotal rows collected: {len(all_data)}")

if len(all_data) == 0:
    print("ERROR: No data collected. Check network connection and try again.")
    sys.exit(1)

# --- Phase 2: Clean and save ---
print("\n" + "=" * 60)
print("Phase 2: Cleaning data")
print("=" * 60)

df = pd.DataFrame(all_data)
df = df.sort_values(by=["Year", "Round", "FinishPos"], ascending=[True, True, True]).reset_index(drop=True)

time_cols = ["AvgFPTime", "QualyTime", "QualTimeDelta"]

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

print("\n--- Adding rolling averages ---")
df["Team"] = df["Team"].replace("Kick Sauber", "Audi")
df["Driver"] = df["Driver"].replace("Andrea Kimi Antonelli", "Kimi Antonelli")
df = df.sort_values(by=["Driver", "Year", "Round"]).reset_index(drop=True)
df["Driver_Form_Avg"] = (
    df.groupby("Driver")["FinishPos"]
    .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
)
df["Driver_Form_Avg"] = df["Driver_Form_Avg"].fillna(20.0)
df = df.sort_values(by=["Year", "Round", "FinishPos"]).reset_index(drop=True)

csv_path = os.path.join(DATA_DIR, "Complete_Driver_Data.csv")
df.to_csv(csv_path, index=False)
print(f"Dataset saved: {csv_path} ({len(df)} rows)")

# --- Phase 3: Create mappings ---
print("\n--- Creating mappings ---")
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

# --- Phase 4: Train model ---
print("\n" + "=" * 60)
print("Phase 4: Training model")
print("=" * 60)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
import numpy as np

from models.F1Net import F1Net
from models.F1Loss import F1Loss
from utils.dataLoader import F1NetDataset, collate

train_set = F1NetDataset(group="train")
test_set = F1NetDataset(group="test")

num_drivers = len(mapping_drivers)
num_teams = len(mapping_teams)
print(f"  Drivers: {num_drivers}, Teams: {num_teams}")
print(f"  Train races: {len(train_set)}, Test races: {len(test_set)}")

if len(train_set) == 0 and len(test_set) == 0:
    print("ERROR: Not enough data. Need at least 3 races (2 train + 1 test).")
    sys.exit(1)

if len(train_set) == 0:
    print("WARNING: No train data. Using test set for both train and val (overfitting demo).")

EPOCHS_P1 = 10
EPOCHS_P2 = 10
LR_P1 = 0.001
LR_P2 = 0.003

model = F1Net(num_teams=num_teams, num_drivers=num_drivers)
lossfn = F1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR_P1)
device = torch.device("cpu")
model.to(device)

train_loader = DataLoader(train_set, batch_size=4, shuffle=False, collate_fn=collate) if len(train_set) > 0 else None
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, collate_fn=collate)

print(f"\n--- Phase 1: Pretraining ({EPOCHS_P1} epochs) ---")
for epoch in range(EPOCHS_P1):
    model.train()
    batch_losses = []

    source = train_loader if train_loader else test_loader
    for batch in source:
        driver_ids = [d.to(device) for d in batch["driver_id"]]
        team_ids = [t.to(device) for t in batch["team_id"]]
        numeric_feat = [n.to(device) for n in batch["numeric_feat"]]
        targets = [n.to(device) for n in batch["targets"]]
        batch_preds = []
        for i in range(len(driver_ids)):
            preds = model(numeric_feat[i], team_ids[i], driver_ids[i])
            batch_preds.append(preds)
        loss = lossfn(batch_preds, targets)
        if torch.isnan(loss):
            continue
        batch_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in test_loader:
            d_id = batch["driver_id"][0].to(device)
            t_id = batch["team_id"][0].to(device)
            n_feat = batch["numeric_feat"][0].to(device)
            y_true = batch["targets"][0].to(device)
            preds = model(n_feat, t_id, d_id)
            loss = lossfn([preds], [y_true])
            if not torch.isnan(loss):
                val_losses.append(loss.item())

    avg_train = sum(batch_losses) / len(batch_losses) if batch_losses else 0
    avg_val = sum(val_losses) / len(val_losses) if val_losses else 0
    print(f"  Epoch [{epoch+1:02d}/{EPOCHS_P1}] Train: {avg_train:.4f} | Val: {avg_val:.4f}")

print(f"\n--- Phase 2: Finetuning ({EPOCHS_P2} epochs) ---")
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer2 = torch.optim.Adam(trainable_params, lr=LR_P2)

for epoch in range(EPOCHS_P2):
    model.train()
    batch_losses = []
    for batch in test_loader:
        d_id = batch["driver_id"][0].to(device)
        t_id = batch["team_id"][0].to(device)
        n_feat = batch["numeric_feat"][0].to(device)
        y_true = batch["targets"][0].to(device)
        preds = model(n_feat, t_id, d_id)
        loss = lossfn([preds], [y_true])
        if torch.isnan(loss):
            continue
        batch_losses.append(loss.item())
        optimizer2.zero_grad()
        loss.backward()
        optimizer2.step()
    avg_loss = sum(batch_losses) / len(batch_losses) if batch_losses else 0
    print(f"  Epoch [{epoch+1:02d}/{EPOCHS_P2}] Loss: {avg_loss:.4f}")

print("\n--- Computing Fisher matrix ---")
fisher = {}
for name, param in model.named_parameters():
    fisher[name] = torch.zeros_like(param)

model.eval()
fisher_count = 0
for batch in test_loader:
    model.zero_grad()
    d_id = batch["driver_id"][0].to(device)
    t_id = batch["team_id"][0].to(device)
    n_feat = batch["numeric_feat"][0].to(device)
    y_true = batch["targets"][0].to(device)
    preds = model(n_feat, t_id, d_id)
    loss = lossfn([preds], [y_true])
    if torch.isnan(loss):
        continue
    loss.backward()
    for name, param in model.named_parameters():
        if param.grad is not None:
            fisher[name] += param.grad.data.pow(2)
    fisher_count += 1

if fisher_count > 0:
    for name in fisher:
        fisher[name] /= fisher_count

print("Fisher matrix computed.")

# --- Phase 5: Save checkpoints ---
print("\n" + "=" * 60)
print("Phase 5: Saving checkpoints")
print("=" * 60)

state_payload = {
    "model_state_dict": model.state_dict(),
    "mappings": master_mapping,
}
checkpoint_path = os.path.join(CHECKPOINT_DIR, "f1net_main.pth")
torch.save(state_payload, checkpoint_path)
print(f"  Model saved: {checkpoint_path}")

fisher_path = os.path.join(CHECKPOINT_DIR, "fisher_info.pkl")
with open(fisher_path, "wb") as f:
    pickle.dump(fisher, f)
print(f"  Fisher saved: {fisher_path}")

# --- Summary ---
races_by_year = df.groupby(["Year", "Round"]).ngroups
print(f"\n{'=' * 60}")
print(f"Bootstrap complete!")
print(f"{'=' * 60}")
print(f"  Total rows: {len(df)}")
print(f"  Total races: {races_by_year}")
print(f"  Years covered: {sorted(df['Year'].unique())}")
print(f"  Data: {csv_path}")
print(f"  Checkpoints: {CHECKPOINT_DIR}")
print(f"  Mappings: {mappings_path}")
print(f"\n  Train races: {len(train_set)} | Test races: {len(test_set)}")
print(f"  Model ready for prediction!")

# --- Phase 6: DVC + Git push ---
if os.getenv("DAGSHUB_TOKEN"):
    print("\n" + "=" * 60)
    print("Phase 6: Pushing to DVC remote & Git")
    print("=" * 60)

    def run_cmd(cmd):
        print(f"  $ {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=ROOT)
        if result.stdout.strip():
            print(f"    {result.stdout.strip()}")
        if result.returncode != 0 and result.stderr.strip():
            print(f"    stderr: {result.stderr.strip()}")
        return result.returncode == 0

    run_cmd("git config user.email bootstrap@f1net.local")
    run_cmd('git config user.name "Bootstrap Script"')

    run_cmd("dvc add data/Complete_Driver_Data.csv")
    run_cmd("dvc add checkpoints/f1net_main.pth checkpoints/fisher_info.pkl")

    run_cmd("git add data/Complete_Driver_Data.csv.dvc data/.gitignore")
    run_cmd("git add checkpoints/f1net_main.pth.dvc checkpoints/fisher_info.pkl.dvc")
    run_cmd("git add config/mappings.json")
    run_cmd('git commit -m "bootstrap: update data, model, and fisher matrix"')

    if run_cmd("git push origin HEAD"):
        print("  Git push succeeded.")
    else:
        print("  WARNING: git push failed. Check remote and credentials.")

    if run_cmd("dvc push"):
        print("  DVC push succeeded.")
    else:
        print("  WARNING: dvc push failed. Check DagsHub remote and token.")
else:
    print("\nSkipping DVC/Git push (no DAGSHUB_TOKEN).")
