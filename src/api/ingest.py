import fastf1
import logging
import os
import sys
import time
import uuid
import datetime

logging.getLogger("fastf1").setLevel(logging.WARNING)
from dotenv import load_dotenv
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException,Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
import httpx
from fastapi.middleware.cors import CORSMiddleware
import structlog
load_dotenv()

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
)
log = structlog.get_logger("ingest")


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
from utils.metrics import (
    setup_metrics, INGEST_JOBS_TOTAL, INGEST_ROWS_ADDED,
    INGEST_DURATION_SECONDS, MODEL_LOADED,
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
START_TIME = datetime.datetime.now(datetime.timezone.utc)

def verify_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=403,detail="Invalid API key")

def _set_status(job_id: str, **kwargs):
    job_store[job_id].update(kwargs)
    msg = kwargs.get('message', kwargs.get('status', ''))
    log.info("job_status", job_id=job_id[:8], **{k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool))})



def fetch_and_process(year: int, round_num: int, job_id: str):
    start = time.time()
    _set_status(job_id, status="running", message="Loading Qualifying session...")
    log.info("ingest_started", job_id=job_id[:8], year=year, round=round_num)
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
            INGEST_JOBS_TOTAL.labels(status="error").inc()
            return

        new_drivers, new_teams = update_mappings(new_df, MAPPINGS_PATH)

        _set_status(job_id, message="Appending data and recomputing rolling averages...")
        master_df = recompute_rolling_avg(normalize_names(pd.concat([master_df, new_df], ignore_index=True)))
        master_df.to_csv(DATA_PATH, index=False)
        INGEST_ROWS_ADDED.inc(len(new_df))
        log.info("data_appended", job_id=job_id[:8], rows=len(new_df))

        elapsed = time.time() - start
        INGEST_DURATION_SECONDS.observe(elapsed)
        INGEST_JOBS_TOTAL.labels(status="success").inc()
        _set_status(
            job_id,
            status="done",
            message="Ingestion complete." if race_results is not None else "Ingestion complete. FinishPos is null — call /ingest/finish after the race.",
            rows_added=len(new_df),
            race_results_available=race_results is not None,
            new_drivers=new_drivers,
            new_teams=new_teams,
        )
        log.info("ingest_complete", job_id=job_id[:8], rows=len(new_df), elapsed=round(elapsed, 2))

    except Exception as e:
        INGEST_JOBS_TOTAL.labels(status="error").inc()
        log.error("ingest_failed", job_id=job_id[:8], error=str(e))
        _set_status(job_id, status="error", message=str(e))


def fill_finish_positions(year: int, round_num: int, job_id: str):
    start = time.time()
    _set_status(job_id, status="running", message="Loading Race session...")
    log.info("finish_started", job_id=job_id[:8], year=year, round=round_num)
    try:
        sessionR = fastf1.get_session(year, round_num, "R")
        sessionR.load()

        master_df = pd.read_csv(DATA_PATH)
        round_mask = (master_df["Year"] == year) & (master_df["Round"] == round_num)
        if not round_mask.any():
            _set_status(job_id, status="error", message=f"No rows found for Year {year} Round {round_num}. Run /ingest first.")
            INGEST_JOBS_TOTAL.labels(status="error").inc()
            return

        results = sessionR.results
        for _, row in results.iterrows():
            driver_name = row["FullName"]
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

        elapsed = time.time() - start
        INGEST_DURATION_SECONDS.observe(elapsed)
        _set_status(job_id, status="done", message="Finish positions and grid positions updated.")
        log.info("finish_complete", job_id=job_id[:8], elapsed=round(elapsed, 2))

        try:
            response = httpx.post(FINETUNE_URL, headers={"X-API-KEY": os.getenv("ADMIN_API_KEY")})
            finetune_job_id = response.json().get("job_id")
            _set_status(job_id, finetune_job_id=finetune_job_id, message="Finetune triggered.")
            log.info("finetune_triggered", job_id=job_id[:8], finetune_job_id=finetune_job_id)
        except Exception as e:
            _set_status(job_id, finetune_warning=f"Finetune trigger failed: {str(e)}")
            log.error("finetune_trigger_failed", job_id=job_id[:8], error=str(e))



    except Exception as e:
        INGEST_JOBS_TOTAL.labels(status="error").inc()
        log.error("finish_failed", job_id=job_id[:8], error=str(e))
        _set_status(job_id, status="error", message=str(e))


app = FastAPI(
    title="F1Net Ingest API",
    description="Fetches FastF1 race data, preprocesses it, and appends to the dataset.",
)

setup_metrics(app, "ingest")

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


@app.get("/health")
def health():
    uptime = (datetime.datetime.now(datetime.timezone.utc) - START_TIME).total_seconds()
    return {"status": "healthy", "service": "ingest", "uptime_seconds": round(uptime, 1)}


@app.post("/ingest")
def ingest(request: RoundRequest, background_tasks: BackgroundTasks,_=Security(verify_key)):
    job_id = str(uuid.uuid4())
    job_store[job_id] = {"status": "queued", "message": "Job queued.", "year": request.year, "round": request.round}
    log.info("ingest_queued", job_id=job_id[:8], year=request.year, round=request.round)
    background_tasks.add_task(fetch_and_process, request.year, request.round, job_id)
    return {"job_id": job_id, "status": "queued"}


@app.post("/ingest/finish")
def ingest_finish(request: RoundRequest, background_tasks: BackgroundTasks,_=Security(verify_key)):
    job_id = str(uuid.uuid4())
    job_store[job_id] = {"status": "queued", "message": "Job queued.", "year": request.year, "round": request.round}
    log.info("finish_queued", job_id=job_id[:8], year=request.year, round=request.round)
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
