FROM nvidia/cuda:12.8.0-base-ubuntu24.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip python-is-python3 \
        git ffmpeg pandoc && \
    rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip
ENV PATH="/opt/venv/bin:${PATH}"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
WORKDIR /app
RUN chmod +x /app/processing_scripts/documenting.sh

CMD ["bash", "-c", "celery -A tasks.app worker --loglevel=info -P solo & uvicorn main:app --host 0.0.0.0 --port 8000"]
