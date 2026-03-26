FROM nvidia/cuda:12.8.0-base-ubuntu24.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip python-is-python3 \
        git ffmpeg pandoc && \
    rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app package (core library)
COPY app/ /app/app/

# Copy backend code
COPY backend/ /app/backend/

# Create data dirs
RUN mkdir -p /data/uploads /data/outputs /data/voice_samples

WORKDIR /app/backend
ENV PYTHONPATH="/app:${PYTHONPATH}"

CMD ["bash", "-c", "celery -A app.tasks.app worker --loglevel=info -P solo & uvicorn main:app --host 0.0.0.0 --port 8000"]
