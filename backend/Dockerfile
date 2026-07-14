FROM python:3.11-slim

# System dependencies required by OpenCV, InsightFace, and ONNX
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    gcc \
    g++ \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Step 1: Install CPU-only PyTorch first (saves ~1.8 GB vs CUDA build) ──────
# Railway has no GPU, so we never need CUDA. CPU-only torch is ~200 MB.
RUN pip install --no-cache-dir \
    torch==2.1.2+cpu \
    torchvision==0.16.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# ── Step 2: Install all other dependencies ─────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi "uvicorn[standard]" sqlalchemy alembic "pydantic>=2.7.0" \
    "python-jose[cryptography]" "passlib[bcrypt]" "bcrypt==4.1.2" \
    python-multipart pymysql opencv-contrib-python numpy \
    facenet-pytorch Pillow onnxruntime insightface python-dotenv \
    resend

# ── Step 3: Pre-download InsightFace buffalo_l model during build ──────────────
# This avoids a slow first-request download at runtime.
RUN python -c "\
from insightface.app import FaceAnalysis; \
fa = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); \
fa.prepare(ctx_id=0, det_size=(640, 640)); \
print('InsightFace models ready.')" || echo "Model pre-download skipped — will download on first start."

# ── Step 4: Copy application code ─────────────────────────────────────────────
COPY . .

# Create the uploads directory (face images, profile pictures)
RUN mkdir -p static/uploads

EXPOSE 8000

# Run Alembic migrations then start the server.
# Railway injects $PORT automatically.
CMD ["sh", "-c", "python -m alembic upgrade head && python -m uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
