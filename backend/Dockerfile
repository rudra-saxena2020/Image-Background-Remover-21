FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Pre-download model (optional: set BIREFNET_MODEL env at build time)
# This is done at runtime if not cached; set HF_HOME to a mounted volume
# to avoid re-downloading on every container restart.

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8080

ENV PORT=8080
ENV DEVICE=auto
ENV USE_FP16=true
ENV MODEL_WARMUP=true
ENV MAX_GPU_CONCURRENCY=1
ENV HF_HOME=/model-cache
# Mount or copy the open-source HiDream repository and weights separately:
# HIDREAM_REPO=/models/HiDream-O1-Image
# HIDREAM_MODEL_PATH=/models/HiDream-O1-Image-Dev

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1"]
