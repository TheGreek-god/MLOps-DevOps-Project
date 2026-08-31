# F1Net-V2

A neural learning-to-rank system for predicting Formula 1 race results, with a fully automated continuous training pipeline that retrains after every race weekend — deployed as a Kubernetes cluster with full MLOps observability.

---

## Overview

F1Net-V2 predicts the finishing order of all 22 drivers in an F1 race weekend using pre-race data — qualifying times, free practice pace, grid positions, and historical driver/team form. The model outputs a ranked list of drivers by predicted finishing position.

The 2026 season introduced a major regulation change (new power units, four new teams, new driver line-ups), making prior years of historical data partially obsolete. F1Net-V2 therefore retrains exclusively on live 2026 race weekends, using Elastic Weight Consolidation (EWC) with a rolling window so the model adapts to the new competitive order without catastrophic forgetting.

After each race weekend the pipeline ingests new data, finetunes the model, and logs the result to MLflow — no manual intervention required.

**Current dataset:** 2026 rounds 1–12 (264 rows, 22 drivers per race). Rounds 13+ are added automatically as they happen. A `Pre/Post Spearman` of **0.76 → 0.80** is logged to MLflow after each finetune run.

### Predict
![Predict Page](assets/predict.png)

### Compare
![Compare Page](assets/compare.png)
---

## Architecture

```
FastF1 API
    │
    ▼
Ingest Service (FastAPI)          ← pre-race: FP + Quali data
    │                             ← post-race: FinishPos + GridPos
    │  writes CSV
    ▼
Dataset (2026 rounds 1–12) ──────► shared volume /mnt/data
    │
    ▼
Finetune Service (FastAPI)        ← triggered automatically post-race
    │   EWC + rolling window (last 6 races)
    │   logs to MLflow (in-cluster)
    ▼
Updated Model Checkpoint ─────────► shared volume /mnt/checkpoints + MLflow Model Registry
    │
    ▼
Predict Service (FastAPI)         ← reads latest checkpoint + dataset
    │
    ▼
React Frontend (nginx)            ← Predict, Compare, Admin pages
    │
    └── reverse-proxies /predict, /ingest → services (same-origin, single tunnel)

Monitoring (Prometheus + Grafana + MLflow)
```

### Deployment topology (Kubernetes)

```
             ┌─────────────────────────────────────────────┐
             │            Minikube cluster (f1net ns)      │
             │                                             │
  ┌──────────▼────────┐      ┌────────────────────────┐    │
  │ frontend (nginx)  │      │  prometheus (scrapes)  │    │
  │ :30083            │      │  :30090                │    │
  │  /predict ────────┼──►   │  ▲                     │    │
  │  /ingest ─────────┼──►   │  │ metrics             │    │
  └──────────┬────────┘      │  └─────┬───────────────┘    │
             │               │        │                    │
  ┌──────────▼────────┐  ┌───▼────────▼────┐  ┌──────────┐ │
  │ predict :30081    │  │ ingest :30080   │  │ finetune │ │
  │                   │  │                 │  │ :30082   │ │
  └──────────┬────────┘  └────────┬────────┘  └────┬─────┘ │
             │               ┌────▼────┐           │       │
             ▼               │  MLflow │◄──────────┘       │
        /mnt/checkpoints     │ :30050  │  logs + model     │
        /mnt/data (hostPath) └─────────┘                   │
             └────── Grafana :30030 ◄──── Prometheus ────  ┘
```

The browser only needs **one** tunnel (the frontend). The nginx sidecar reverse-proxies `/predict` and `/ingest` to the upstream services using in-cluster DNS (`predict-svc`, `ingest-svc`), so no cross-origin `localhost` calls are needed.

---

## Project Structure

```
F1Net-V2/
├── src/
│   ├── api/          # FastAPI services (predict, ingest, finetune)
│   ├── models/       # F1Net architecture + ListNet loss
│   ├── utils/        # Dataset loader, preprocessing
│   └── training/     # Pretraining + Fisher matrix computation
├── f1net-frontend/   # React + Vite frontend
├── docker/           # Dockerfiles (base + shared) for all services
│   ├── base.Dockerfile      # shared deps image (torch, mlflow, fastf1, …)
│   ├── ingest.Dockerfile    # FROM f1net-base + service code
│   ├── predict.Dockerfile   # FROM f1net-base + service code
│   ├── finetune.Dockerfile  # FROM f1net-base + service code
│   ├── frontend.Dockerfile  # Vite build → nginx
│   ├── nginx.conf           # static + config.js + reverse-proxy
│   └── frontend-start.sh    # envsubst → config.js at runtime
├── config/           # Driver/team ID mappings
├── data/             # 2026 dataset (264 rows)
├── checkpoints/      # model weights + Fisher matrix
├── scripts/
│   ├── bootstrap.py  # dataset fetch + model pretraining
│   └── deploy.py     # applies k8s manifests
├── k8s/
│   ├── monitoring/   # prometheus.yml, grafana.yml, mlflow.yml
│   ├── ingest/       # deploy.yml + svc
│   ├── predict/      # deploy.yml + svc
│   ├── finetune/     # deploy.yml + svc
│   └── frontend/     # deploy.yml + svc
├── assets/           # README screenshots
├── docker-compose.yml
└── requirements.txt
```

---

## Model

F1Net is a learning-to-rank model trained with ListNet loss (KL divergence between predicted and target softmax distributions).

**Input features per driver:**
- Driver embedding (learned, 16-dim)
- Team embedding (learned, 16-dim)
- Numeric features: qualifying time, qualifying delta to pole, average FP pace, grid position, rolling 3-race finish average

**Architecture:**
- Embedding layers for driver and team IDs
- Feature projection MLP
- Scoring head → single scalar per driver
- Rankings derived by sorting predicted scores

**Loss:** KL divergence between `softmax(scores / T)` and `softmax(targets / T)` where temperature `T = 3.4` (found via Hyperopt). Higher temperature flattens the target distribution, providing better gradient signal for mid-field predictions rather than only optimizing the top positions.

**Evaluation:** Spearman rank correlation across all 22 positions. The 2026 model reaches ~0.79–0.80 on live rounds after finetuning (see MLflow, `F1Net_Finetune` experiment).

**2026 adaptation:** Because no 2022–2025 in-season data is loaded anymore, training runs a two-phase strategy scoped to the live season:
- Phase 1: Pretrain on the available 2026 rounds (train split = earliest 9 rounds)
- Phase 2: EWC-regularized finetune on a rolling window of the last 6 rounds, adapting to the latest form
- New drivers/teams are initialized from a "rookie"/"new-team" baseline index so the model can score unseen entities

---

## Continuous Training Pipeline

After each race weekend, the pipeline runs end-to-end automatically:

1. **Pre-race** — call `/ingest` with year and round number. Fetches FP1/FP2/FP3 + qualifying data from FastF1, preprocesses, and appends to the master dataset with `FinishPos = null`
2. **Post-race** — call `/ingest/finish`. Fills in actual finishing positions from the race session, recomputes rolling averages, triggers finetune automatically
3. **Finetune** — EWC-regularized finetuning on a rolling window of the last 6 races. EWC lambda = 10, 10 epochs, Adam lr = 0.003. Pre/Post Spearman logged to MLflow (in-cluster), updated checkpoint saved to the shared volume

**EWC (Elastic Weight Consolidation):** Adds a regularization term to the loss penalizing parameter drift from the pretrained weights, weighted by the Fisher information matrix (computed once on the full dataset). This lets the model learn new race patterns without forgetting the established driver/team order.

```
L_total = L_new + (λ/2) * Σ F_i * (θ_i - θ_pretrained_i)²
```

**Experiment tracking:** Every finetune run logs `Pre_Spearman_Corr`, `Post_Spearman_Corr`, and the `EWC_Lam` parameter to the in-cluster MLflow server (`F1Net_Finetune` experiment), and registers the model (`F1NET`, `prod` alias) in the MLflow Model Registry. Runtime training metrics are also exported to Prometheus for the Grafana dashboard.

---

## API Services

Three independent FastAPI services, each containerized separately:

**Ingest (port 8000)**
Fetches race weekend data from FastF1. Split into two endpoints because pre-race data (FP + Quali) is available Friday/Saturday while race results only exist Sunday evening. Running them separately avoids blocking predictions on race data availability.
- `POST /ingest` — pre-race ingestion (FP + Quali)
- `POST /ingest/finish` — post-race: fills FinishPos, triggers finetune
- `GET /ingest/status/{job_id}` — async job polling
- All write operations protected by `X-API-KEY` header

**Predict (port 8001)**
Loads model checkpoint and serves ranking predictions. Reads dataset fresh on each request to pick up new races ingested since startup. Falls back to the MLflow `F1NET/prod` model if the local checkpoint is unavailable.
- `POST /predict` — takes `{year, round_num}`, returns ranked predictions with driver, team, predicted position, actual position (if available), and exact match flag. Returns `404 {"detail":"Race Data not available"}` for rounds not yet run.

**Finetune (port 8002)**
Runs EWC finetuning on the rolling window dataset. Triggered automatically by ingest/finish but can also be called manually.
- `POST /finetune` — queues async finetune job
- `GET /finetune/status/{job_id}` — job status polling

**MLflow (port 5000 / NodePort 30050)**
In-cluster experiment tracker + model registry. Services point at `MLFLOW_TRACKING_URI=http://mlflow-svc:5000`. Note: if a `DAGSHUB_TOKEN` is set and no explicit cluster URI is provided, the services will log to DagsHub instead — the cluster starts with `MLFLOW_TRACKING_URI` pinned to the local server.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | PyTorch, custom ListNet loss |
| Hyperparameter tuning | Hyperopt |
| Data fetching | FastF1 |
| APIs | FastAPI, Uvicorn |
| Experiment tracking | MLflow (in-cluster) |
| Monitoring | Prometheus + Grafana |
| Orchestration | Kubernetes (Minikube) |
| Containerization | Docker, shared base image |
| Frontend | React, Vite, Axios |
| Serving | Nginx reverse proxy |
| Continuous training | EWC, rolling window finetuning |

---

## Local Development (Docker Compose)

The original Docker Compose layout is preserved for local experimentation:

```
docker compose up --build
```

Services: `ingest:8000`, `predict:8001`, `finetune:8002`, `frontend:3000`.

---

## Kubernetes Deployment

Requirements: [Minikube](https://minikube.sigs.k8s.io/docs/start/) + Docker driver, `kubectl`.

### 1. Build and load images into Minikube

Build from the repo root against Minikube's Docker daemon so images are directly available (the shared base image avoids re-transferring ~2.6 GB per service):

```powershell
& minikube docker-env --shell powershell | Invoke-Expression
docker build -f docker/base.Dockerfile    -t f1net-base:latest     .
docker build -f docker/ingest.Dockerfile  -t f1net-ingest:latest   .
docker build -f docker/predict.Dockerfile -t f1net-predict:latest  .
docker build -f docker/finetune.Dockerfile -t f1net-finetune:latest .
docker build -f docker/frontend.Dockerfile -t f1net-frontend:latest .
```

If built with Docker Desktop instead, load into Minikube:

```powershell
minikube image load f1net-base:latest f1net-ingest:latest f1net-predict:latest f1net-finetune:latest f1net-frontend:latest
```

### 2. Provision shared volumes and secrets

```powershell
minikube ssh -- "sudo mkdir -p /mnt/data /mnt/checkpoints /mnt/mlflow && sudo chmod 777 /mnt/data /mnt/checkpoints /mnt/mlflow"

kubectl create secret generic f1net-secrets -n f1net --from-literal=ADMIN_API_KEY=<key> --from-literal=DAGSHUB_TOKEN=<token> ...
```

### 3. Apply manifests

```powershell
kubectl apply -f k8s/monitoring/   # mlflow, prometheus, grafana
kubectl apply -f k8s/ingest/       # + finetune, predict, frontend
kubectl apply -f k8s/finetune/
kubectl apply -f k8s/predict/
kubectl apply -f k8s/frontend/
```

### 4. Verify

```powershell
kubectl get pods -n f1net          # all 7 pods 1/1 Running
kubectl get svc -n f1net
```

### 5. Access the UIs (Docker driver needs a tunnel)

The frontend reverse-proxies `/predict` and `/ingest`, so **one** tunnel reaches the whole app:

```powershell
minikube service -n f1net frontend-svc   # app (Predict / Compare / Admin)
minikube service -n f1net grafana-svc    # dashboards — admin / f1net
minikube service -n f1net mlflow-svc     # experiment tracker + model registry
minikube service -n f1net prometheus-svc # metrics
```

| Service | NodePort | Notes |
|---|---|---|
| frontend | 30083 | React app (nginx) |
| predict | 30081 | `/predict` ranking |
| ingest | 30080 | `/ingest`, `/ingest/finish` |
| finetune | 30082 | `/finetune` |
| mlflow | 30050 | experiments + model registry |
| grafana | 30030 | dashboards (`admin` / `f1net`) |
| prometheus | 30090 | `/metrics` scraping |

### Example: trigger a finetune run (creates an MLflow run)

```powershell
$key = (kubectl get secret f1net-secrets -n f1net -o jsonpath='{.data.ADMIN_API_KEY}' | ForEach-Object { [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($_)) }).Trim()
curl -X POST -H "X-API-KEY: $key" -d '{}' http://localhost:30082/finetune
```

---

## Operational Notes (things we hit)

- **Shared base image:** `docker/base.Dockerfile` holds all pip deps; the three service Dockerfiles are thin (`FROM f1net-base`) so rebuilds and image loads are fast.
- **Grafana / MLflow OOM:** `grafana:latest` and `mlflow:latest` boot many plugin/uvicorn sub-processes; keep limits ≥ 1 Gi (Grafana 1 Gi) / 2 Gi (MLflow 1 Gi request / 2 Gi limit) or they silently hang or `OOMKilled`.
- **Prometheus RBAC:** the Prometheus pod must have a Role to `get/list/watch pods` or it can't discover targets (`pods is forbidden`); a `ServiceAccount` + `RoleBinding` is included in `k8s/monitoring/prometheus.yml`.
- **MLflow client/server version pin:** the services bundle `mlflow` 3.x; the server must also be 3.x (`ghcr.io/mlflow/mlflow:latest`). A 3.x client against a 2.x server 404s on `/api/2.0/mlflow/logged-models`.
- **MLflow 3.x model logging:** `mlflow.pytorch.log_model` defaults to the `pt2` (traced) serialization format, which requires `TensorSpec`/`input_example`; the services set `serialization_format="pickle"` to avoid tracing multi-tensor inputs.
- **MLflow 3.x reserved alias:** the alias name `latest` is reserved; the pipeline registers the model under the `prod` alias instead.
- **MLflow security middleware:** enable it only with explicit hosts, or disable it for a local cluster (`MLFLOW_SERVER_DISABLE_SECURITY_MIDDLEWARE=true`) — otherwise non-localhost probes are blocked.
- **FastF1 rate limits:** the bootstrap script pauses between races to respect the 500 calls/hour limit.
- **Persistent data:** datasets, checkpoints, and the MLflow sqlite store live on Minikube host paths (`/mnt/data`, `/mnt/checkpoints`, `/mnt/mlflow`) so they survive pod restarts; MLflow's store was originally on an `emptyDir` and lost on every OOM restart.