FROM f1net-base:latest

WORKDIR /app

COPY src/api/predict.py ./src/api/
COPY src/models/ ./src/models/
COPY src/utils/dataLoader.py ./src/utils/
COPY src/utils/metrics.py ./src/utils/
COPY config/ ./config/

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

CMD ["uvicorn", "src.api.predict:app", "--host", "0.0.0.0", "--port", "8001"]