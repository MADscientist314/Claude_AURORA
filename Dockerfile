# AURORA — PAS Detection Dashboard
# Builds a self-contained image with all code and model weights.
# No GPU required; CPU inference is sufficient for research use.

FROM python:3.10-slim

# System libraries required by Pillow and TensorFlow
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first — separate layer for cache efficiency.
# Re-runs only when requirements.txt changes, not on every code edit.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY pipeline/ pipeline/
COPY static/ static/

# Copy trained model weights.
# These are gitignored (large binaries) but must be present in the build
# context (i.e. the local models/ folder) before running docker compose build.
COPY models/ models/

# TensorFlow must use its own backend when loaded via standalone Keras 3.x
ENV KERAS_BACKEND=tensorflow

EXPOSE 8000

# Single worker: TensorFlow/Keras models are not fork-safe.
# Keep-alive timeout accommodates long GradCAM streams (several minutes).
CMD ["uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "600"]
