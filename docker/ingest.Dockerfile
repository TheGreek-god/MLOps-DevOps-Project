FROM f1net-base:latest

WORKDIR /app

COPY src/api/ingest.py ./src/api/
COPY src/utils/ ./src/utils/
COPY config/ ./config/

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.ingest:app", "--host", "0.0.0.0", "--port", "8000"]