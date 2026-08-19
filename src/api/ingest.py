import fastf1
import logging
import os
import subprocess
import sys
import time
import uuid

logging.getLogger("fastf1").setLevel(logging.WARNING)
from dotenv import load_dotenv
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException,Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import httpx
from fastapi.middleware.cors import CORSMiddleware
load_dotenv()


def find_project_root():
    current = os.path.dirname(os.path.abspath(__file__))
    while current != os.path.dirname(current):
        if os.path.exists(os.path.join(current, "requirements.txt")):
            return current
        current = os.path.dirname(current)
    raise FileNotFoundError("Could not find project root")


root_path = find_project_root()
sys.path.insert(0, os.path.join(root_path, "src"))

from utils.prep_data import clean_single_race_production, normalize_names, recompute_rolling_avg, update_mappings
github_token = os.getenv("GITHUB_TOKEN")
github_user = os.getenv("GITHUB_USER")
if github_token:
    subprocess.run(["git", "config", "--global", "user.email", "pranav070904@users.noreply.github.com"], cwd=root_path)
    subprocess.run(["git", "config", "--global", "user.name", "pranav070904"], cwd=root_path)
    subprocess.run(
        ["git", "remote", "set-url", "origin", 
         f"https://{github_user}:{github_token}@github.com/{github_user}/F1Net-V2.git"],
        cwd=root_path, capture_output=True
    )

FINETUNE_URL = os.getenv("FINETUNE_API_URL", "http://localhost:8002/finetune")
CACHE_FOLDER = os.path.join(root_path, "cache_folder")
DATA_PATH = os.path.join(root_path, "data", "Complete_Driver_Data.csv")
MAPPINGS_PATH = os.path.join(root_path, "config", "mappings.json")
API_KEY = os.getenv("ADMIN_API_KEY")
api_key_header = APIKeyHeader(name = "X-API-KEY")
os.makedirs(CACHE_FOLDER, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_FOLDER)

job_store: dict[str, dict] = {}

def verify_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403,detail="Invalid API key")

def _set_status(job_id: str, **kwargs):
    job_store[job_id].update(kwargs)
    print(f"[Job {job_id[:8]}] {kwargs.get('message', kwargs.get('status', ''))}")


def _dvc_add(job_id: str):
    dvc = subprocess.run(
        ["dvc", "add", DATA_PATH],
        capture_output=True, text=True, cwd=root_path,
    )
    git_add = subprocess.run(
        ["git", "add", f"{DATA_PATH}.dvc"],
        capture_output=True, text=True, cwd=root_path
    )
    git_commit = subprocess.run(
        ["git", "commit", "-m", "ingest: dataset update"],
        capture_output=True, text=True, cwd=root_path
    )
    git_push = subprocess.run(
        ["git", "push", "origin", "main"],
        capture_output=True, text=True, cwd=root_path
    )
    dvc_push = subprocess.run(
        ["dvc", "push"],
        capture_output=True, text=True, cwd=root_path
    )
    _set_status(job_id, dvc_add_output=(dvc.stdout + dvc.stderr).strip())
    _set_status(job_id, git_add_output=(git_add.stdout + git_add.stderr).strip())
    _set_status(job_id, git_commit_output=(git_commit.stdout + git_commit.stderr).strip())
    _set_status(job_id, git_push_output=(git_push.stdout + git_push.stderr).strip())
    if dvc_push.returncode != 0:
        _set_status(job_id, dvc_push_warning=f"DVC push failed: {dvc_push.stderr.strip()}")
    else:
        _set_status(job_id, message="DVC push complete.")


def fetch_and_process(year: int, round_num: int, job_id: str):
    _set_status(job_id, status="running", message="Loading Qualifying session...")
    try:
        sessionQ = fastf1.get_session(year, round_num, "Q")
        sessionQ.load()
        quali_results = sessionQ.results.set_index("DriverNumber")

        time.sleep(3)

        _set_status(job_id, message="Loading Free Practice sessions...")
        all_fp_laps = []
        fp1 = fastf1.get_session(year, round_num, "FP1")
        fp1.load()
        time.sleep(3)
        all_fp_laps.append(fp1.laps)

        for fp_label in ["FP2", "FP3"]:
            try:
                fp = fastf1.get_session(year, round_num, fp_label)
                fp.load()
                time.sleep(3)
                all_fp_laps.append(fp.laps)
            except Exception:
                pass

        avg_fp = pd.concat(all_fp_laps, ignore_index=True).groupby("Driver")["LapTime"].mean().dt.total_seconds()

        pole_time = sessionQ.laps.pick_fastest()["LapTime"]
        valid_q = sessionQ.laps.pick_not_deleted()
        if isinstance(valid_q, pd.Series):
            valid_q = valid_q.to_frame().T
        valid_q = valid_q[valid_q["LapTime"].notna()]
        fastest_per_driver = valid_q.loc[valid_q.groupby("Driver")["LapTime"].idxmin()].copy()
        pole_time = fastest_per_driver["LapTime"].min()
        fastest_per_driver["LapDelta"] = fastest_per_driver["LapTime"].apply(lambda x: x - pole_time).dt.total_seconds()
        fastest_per_driver["LapTime"] = fastest_per_driver["LapTime"].dt.total_seconds()
        

       
        _set_status(job_id, message="Attempting to load Race session...")
        race_results = None
        try:
            sessionR = fastf1.get_session(year, round_num, "R")
            sessionR.load()
            time.sleep(3)
            temp_results = sessionR.results.set_index("DriverNumber")
            if "TeamName" in temp_results.columns and not temp_results.empty and temp_results["TeamName"].notna().any():
                race_results = temp_results
        except Exception:
            pass

        driver_source = race_results if race_results is not None else sessionQ.results.set_index("DriverNumber")

        rows = []
        for drv in driver_source.index:
            row = driver_source.loc[drv]
            driver_code = row["Abbreviation"]
            q_row = fastest_per_driver[fastest_per_driver["Driver"] == driver_code]
            q_position = quali_results.loc[drv, "Position"] if drv in quali_results.index else None
            rows.append({
                "Year": year,
                "Round": round_num,
                "Driver": row["FullName"],
                "Team": row["TeamName"],
                "AvgFPTime": avg_fp.get(driver_code),
                "QualyTime": q_row.iloc[0]["LapTime"] if not q_row.empty else None,
                "QualTimeDelta": q_row.iloc[0]["LapDelta"] if not q_row.empty else None,
                "GridPos": row["GridPosition"] if (race_results is not None and pd.notna(row["GridPosition"]) and row["GridPosition"] != 0 and row["GridPosition"] != -1) else q_position,
                "FinishPos": row["Position"] if race_results is not None else None,
                "IsAccurate": q_row.iloc[0]["IsAccurate"] if not q_row.empty else None,
            })

        new_df = normalize_names(pd.DataFrame(rows))
        new_df = clean_single_race_production(new_df)
        new_df = new_df.sort_values("FinishPos").reset_index(drop=True)

        master_df = pd.read_csv(DATA_PATH)
        if ((master_df["Year"] == year) & (master_df["Round"] == round_num)).any():
            _set_status(job_id, status="error", message=f"Year {year} Round {round_num} already exists. Use /ingest/finish to fill missing results.")
            return

        new_drivers, new_teams = update_mappings(new_df, MAPPINGS_PATH)

        _set_status(job_id, message="Appending data and recomputing rolling averages...")
        master_df = recompute_rolling_avg(normalize_names(pd.concat([master_df, new_df], ignore_index=True)))
        master_df.to_csv(DATA_PATH, index=False)
        print("New Data Added")
        _set_status(job_id, message="Updating DVC tracking...")
        _dvc_add(job_id)

        _set_status(
            job_id,
            status="done",
            message="Ingestion complete." if race_results is not None else "Ingestion complete. FinishPos is null — call /ingest/finish after the race.",
            rows_added=len(new_df),
            race_results_available=race_results is not None,
            new_drivers=new_drivers,
            new_teams=new_teams,
        )

    except Exception as e:
        _set_status(job_id, status="error", message=str(e))


def fill_finish_positions(year: int, round_num: int, job_id: str):
    _set_status(job_id, status="running", message="Loading Race session...")
    try:
        sessionR = fastf1.get_session(year, round_num, "R")
        sessionR.load()

        master_df = pd.read_csv(DATA_PATH)
        round_mask = (master_df["Year"] == year) & (master_df["Round"] == round_num)
        if not round_mask.any():
            _set_status(job_id, status="error", message=f"No rows found for Year {year} Round {round_num}. Run /ingest first.")
            return

        results = sessionR.results
        for _, row in results.iterrows():
            driver_name = row["FullName"]
            # Apply same normalization used during ingest so names match
            normalized = normalize_names(pd.DataFrame([{"Driver": driver_name, "Team": row["TeamName"]}]))
            driver_name = normalized.iloc[0]["Driver"]

            driver_mask = round_mask & (master_df["Driver"] == driver_name)
            if driver_mask.any():
                master_df.loc[driver_mask, "FinishPos"] = row["Position"]
                grid_pos  = row["GridPosition"]
                if pd.notna(grid_pos) and grid_pos != 0 and grid_pos != -1:
                    master_df.loc[driver_mask, "GridPos"] = grid_pos

        _set_status(job_id, message="Recomputing rolling averages...")
        master_df = recompute_rolling_avg(master_df)
        master_df.to_csv(DATA_PATH, index=False)

        _set_status(job_id, message="Updating DVC tracking...")
        _dvc_add(job_id)

        _set_status(job_id, status="done", message="Finish positions and grid positions updated.")

        try:
            response = httpx.post(FINETUNE_URL, headers={"X-API-KEY": os.getenv("ADMIN_API_KEY")})
            finetune_job_id = response.json().get("job_id")
            _set_status(job_id, finetune_job_id=finetune_job_id, message="Finetune triggered.")
        except Exception as e:
            _set_status(job_id, finetune_warning=f"Finetune trigger failed: {str(e)}")



    except Exception as e:
        _set_status(job_id, status="error", message=str(e))


app = FastAPI(
    title="F1Net Ingest API",
    description="Fetches FastF1 race data, preprocesses it, and appends to the DVC-tracked dataset.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONT_END_URL","http://localhost:5173")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RoundRequest(BaseModel):
    year: int
    round: int


@app.post("/ingest")
def ingest(request: RoundRequest, background_tasks: BackgroundTasks,_=Security(verify_key)):
    job_id = str(uuid.uuid4())
    job_store[job_id] = {"status": "queued", "message": "Job queued.", "year": request.year, "round": request.round}
    background_tasks.add_task(fetch_and_process, request.year, request.round, job_id)
    return {"job_id": job_id, "status": "queued"}


@app.post("/ingest/finish")
def ingest_finish(request: RoundRequest, background_tasks: BackgroundTasks,_=Security(verify_key)):
    job_id = str(uuid.uuid4())
    job_store[job_id] = {"status": "queued", "message": "Job queued.", "year": request.year, "round": request.round}
    background_tasks.add_task(fill_finish_positions, request.year, request.round, job_id)
    return {"job_id": job_id, "status": "queued"}


@app.get("/ingest/status/{job_id}")
def ingest_status(job_id: str,_=Security(verify_key)):
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/ingest/jobs")
def list_jobs(_=Security(verify_key)):
    return {jid: {"status": j["status"], "year": j.get("year"), "round": j.get("round")} for jid, j in job_store.items()}
