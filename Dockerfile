# Dockerfile - CPU-only, robust for this project
FROM python:3.10-slim

WORKDIR /app

# system deps some Python packages need
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy app
COPY . .

# logs dir
RUN mkdir -p /app/logs

EXPOSE 8000

# run uvicorn (single worker so model loads once)
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]

