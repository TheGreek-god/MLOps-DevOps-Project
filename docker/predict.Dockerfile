FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir typing-extensions --upgrade
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/api/predict.py ./src/api/
COPY src/models/ ./src/models/
COPY src/utils/dataLoader.py ./src/utils/
COPY src/utils/metrics.py ./src/utils/
COPY config/ ./config/

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

CMD ["uvicorn", "src.api.predict:app", "--host", "0.0.0.0", "--port", "8001"]
