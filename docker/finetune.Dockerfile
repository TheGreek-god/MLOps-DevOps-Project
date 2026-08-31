FROM f1net-base:latest

WORKDIR /app

COPY src/api/finetune.py ./src/api/
COPY src/models/ ./src/models/
COPY src/utils/ ./src/utils/
COPY config/ ./config/

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8002/health || exit 1

CMD ["uvicorn", "src.api.finetune:app", "--host", "0.0.0.0", "--port", "8002"]