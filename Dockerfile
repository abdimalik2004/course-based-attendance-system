FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────────────
# CPU-only PyTorch first (saves ~1.8 GB vs CUDA wheel)
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir \
    torch==2.1.2+cpu \
    torchvision==0.16.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    --prefer-binary \
    -r /app/backend/requirements.txt

# ── Pre-download InsightFace buffalo_l model ───────────────────────────────────
RUN python -c "\
from insightface.app import FaceAnalysis; \
fa = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); \
fa.prepare(ctx_id=0, det_size=(640, 640)); \
print('InsightFace buffalo_l model ready.')" \
    || echo "InsightFace pre-download skipped — will download on first start."

# ── Application code ───────────────────────────────────────────────────────────
# Copies the full tree: backend/ + face_recognition/ + utils/ + dataset/ + models/
COPY . /app

# ── Entrypoint script ──────────────────────────────────────────────────────────
# On first boot it migrates baked-in dataset/models → Railway Volume (/data/).
# On subsequent boots /data/ is already populated so the copy is skipped.
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# ── Redirect all mutable data to /data (Railway persistent volume) ─────────────
ENV ATTENDANCE_DATASET_DIR=/data/dataset
ENV ATTENDANCE_EMBEDDING_PATH=/data/models/face_embeddings.npz
ENV ATTENDANCE_MODEL_PATH=/data/models/lbph_trainer.yml
ENV ATTENDANCE_LABEL_MAP_PATH=/data/models/label_map.json
ENV STATIC_DIR=/data/uploads
ENV STATIC_UPLOAD_DIR=/data/uploads

WORKDIR /app/backend
EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
