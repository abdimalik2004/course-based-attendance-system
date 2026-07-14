#!/bin/sh
# ─────────────────────────────────────────────────────────────────────────────
#  Heegan Attendance — Railway container entrypoint
#
#  On FIRST boot (Railway Volume is empty):
#    • Copies baked-in dataset/ and models/ to /data/ so existing students
#      and trained embeddings carry over without re-enrolling anyone.
#
#  On SUBSEQUENT boots:
#    • /data/ already has everything — copy is skipped instantly.
#
#  Then runs Alembic migrations and starts the FastAPI server.
# ─────────────────────────────────────────────────────────────────────────────
set -e

DATA_DIR="/data"
DATASET_SRC="/app/dataset"
MODELS_SRC="/app/models"
UPLOADS_SRC="/app/backend/backend/static/uploads"

# Create volume subdirs if they don't exist
mkdir -p "$DATA_DIR/dataset" "$DATA_DIR/models" "$DATA_DIR/uploads"

# ── Seed dataset from baked-in images (first boot only) ───────────────────────
if [ -d "$DATASET_SRC" ] && [ -z "$(ls -A $DATA_DIR/dataset 2>/dev/null)" ]; then
    echo "[entrypoint] First boot — seeding /data/dataset from baked-in dataset..."
    cp -r "$DATASET_SRC/." "$DATA_DIR/dataset/"
    echo "[entrypoint] Dataset seeded: $(find $DATA_DIR/dataset -type f | wc -l) image(s)."
else
    echo "[entrypoint] /data/dataset already populated — skipping seed."
fi

# ── Seed models from baked-in files (first boot only) ─────────────────────────
if [ -d "$MODELS_SRC" ] && [ -z "$(ls -A $DATA_DIR/models 2>/dev/null)" ]; then
    echo "[entrypoint] First boot — seeding /data/models from baked-in models..."
    cp -r "$MODELS_SRC/." "$DATA_DIR/models/"
    echo "[entrypoint] Models seeded."
else
    echo "[entrypoint] /data/models already populated — skipping seed."
fi

# ── Seed uploads (profile images) if any exist ────────────────────────────────
if [ -d "$UPLOADS_SRC" ] && [ "$(ls -A $UPLOADS_SRC 2>/dev/null)" ] && [ -z "$(ls -A $DATA_DIR/uploads 2>/dev/null)" ]; then
    echo "[entrypoint] First boot — seeding /data/uploads..."
    cp -r "$UPLOADS_SRC/." "$DATA_DIR/uploads/"
fi

# ── Alembic migrations ─────────────────────────────────────────────────────────
echo "[entrypoint] Running Alembic migrations..."
python -m alembic upgrade head
echo "[entrypoint] Migrations complete."

# ── Start FastAPI server ───────────────────────────────────────────────────────
echo "[entrypoint] Starting server on port ${PORT:-8000}..."
exec python -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
