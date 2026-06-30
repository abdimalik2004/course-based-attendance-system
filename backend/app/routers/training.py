from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pathlib import Path

from app.core.security import require_roles
from app.services.training_manager import enqueue_embeddings_training, get_job, list_jobs


router = APIRouter(prefix="/training", tags=["training"])


@router.get("", dependencies=[Depends(require_roles("SUPER_ADMIN", "ADMISSIONS"))])
def list_training_jobs():
    """Return all training jobs (queued, running, succeeded, failed)."""
    jobs = list_jobs()
    # Most recent first
    jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
    return {"total": len(jobs), "jobs": jobs}


@router.get("/{job_id}", dependencies=[Depends(require_roles("SUPER_ADMIN", "ADMISSIONS"))])
def get_training_status(job_id: str):
    """Return the status of a single training job by ID."""
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@router.post(
    "/retrain",
    dependencies=[Depends(require_roles("SUPER_ADMIN"))],
    summary="Trigger a full model retrain from the existing dataset",
)
def trigger_retrain():
    """
    Enqueue a full embedding retrain from whatever images are already on disk.
    Useful after manually adding or editing dataset images without going through /students/capture.
    Only SUPER_ADMIN can call this.
    """
    from utils.config import load_config
    from face_recognition.train import train_embeddings_from_dataset
    from app.services.face_service import face_service

    cfg = load_config()
    dataset_root = Path(cfg.dataset_dir)
    if not dataset_root.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Dataset directory does not exist: {dataset_root}",
        )

    def _retrain():
        from app.services.face_service import face_service as _fs

        # Borrow already-loaded models to avoid ONNX hang (see students.py for details).
        # _retrain() is already executing inside a background job thread, so there's no
        # need for a nested thread + timeout — that just adds unnecessary complexity.
        borrowed_model = None
        borrowed_detector = None
        try:
            with _fs._lock:
                rec = _fs._recognizer
                if rec is not None:
                    borrowed_model = rec.model
                    borrowed_detector = rec.detector
        except Exception:
            pass

        if borrowed_model is None:
            try:
                det_rec = _fs._ensure_detect_recognizer()
                if det_rec is not None:
                    borrowed_model = det_rec.model
                    borrowed_detector = det_rec.detector
            except Exception:
                pass

        # Run training directly in this background thread.
        train_embeddings_from_dataset(
            cfg,
            model=borrowed_model,
            detector=borrowed_detector,
        )

        try:
            _fs.reload_models()
        except Exception:
            pass

    job_id = enqueue_embeddings_training(
        _retrain,
        meta={"trigger": "manual_retrain", "dataset_dir": str(dataset_root)},
    )
    return {"job_id": job_id, "status": "queued"}
