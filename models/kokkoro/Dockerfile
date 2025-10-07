FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY kokkoro_server.py .
COPY handler.py .

# Pre-download model
RUN python3 -c "from transformers import AutoModel; \
    AutoModel.from_pretrained('hexgrad/Kokoro-82M', trust_remote_code=True)" || true

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/health')"

CMD ["python3", "-m", "uvicorn", "kokkoro_server:app", "--host", "0.0.0.0", "--port", "8000"]
