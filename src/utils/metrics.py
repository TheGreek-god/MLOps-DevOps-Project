from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator
import time
import structlog

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
)
logger = structlog.get_logger()

INGEST_JOBS_TOTAL = Counter(
    "f1net_ingest_jobs_total",
    "Total number of ingest jobs",
    ["status"],
)
INGEST_ROWS_ADDED = Counter(
    "f1net_ingest_rows_added_total",
    "Total rows added by ingest",
)
INGEST_DURATION_SECONDS = Histogram(
    "f1net_ingest_duration_seconds",
    "Time spent processing ingest job",
    buckets=[5, 10, 30, 60, 120, 300, 600],
)

FINETUNE_JOBS_TOTAL = Counter(
    "f1net_finetune_jobs_total",
    "Total number of finetune jobs",
    ["status"],
)
FINETUNE_DURATION_SECONDS = Histogram(
    "f1net_finetune_duration_seconds",
    "Time spent finetuning",
    buckets=[10, 30, 60, 120, 300, 600],
)
FINETUNE_SPEARMAN_PRE = Gauge(
    "f1net_finetune_spearman_pre",
    "Spearman correlation before finetune",
)
FINETUNE_SPEARMAN_POST = Gauge(
    "f1net_finetune_spearman_post",
    "Spearman correlation after finetune",
)
FINETUNE_EPOCHS = Counter(
    "f1net_finetune_epochs_total",
    "Total finetune epochs completed",
)
FINETUNE_LOSS = Gauge(
    "f1net_finetune_loss",
    "Latest finetune loss value",
)

PREDICT_REQUESTS_TOTAL = Counter(
    "f1net_predict_requests_total",
    "Total prediction requests",
)
PREDICT_DURATION_SECONDS = Histogram(
    "f1net_predict_duration_seconds",
    "Time spent on prediction",
    buckets=[0.1, 0.5, 1, 2, 5],
)

MODEL_LOADED = Gauge(
    "f1net_model_loaded",
    "Whether the model is currently loaded (1=yes, 0=no)",
)
MODEL_CHECKPOINT_AGE_INFO = Info(
    "f1net_model_checkpoint",
    "Info about the loaded model checkpoint",
)

def setup_metrics(app: FastAPI, service_name: str):
    instrumentator = Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        excluded_handlers=["/metrics"],
    )
    instrumentator.instrument(app)
    instrumentator.expose(app, endpoint="/metrics", include_in_schema=True)

    SERVICE_INFO = Info(
        f"f1net_{service_name}",
        f"Info about the {service_name} service",
    )
    SERVICE_INFO.info({"service": service_name, "version": "2.0.0"})
