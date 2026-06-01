from __future__ import annotations

import threading
import uuid
import logging
from datetime import datetime
from typing import Any, Callable

logger = logging.getLogger(__name__)

_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _run_job(job_id: str, fn: Callable[[], Any]) -> None:
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
        _jobs[job_id]["started_at"] = datetime.utcnow().isoformat()
    try:
        fn()
        with _jobs_lock:
            _jobs[job_id]["status"] = "succeeded"
            _jobs[job_id]["finished_at"] = datetime.utcnow().isoformat()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Training job failed", exc_info=exc)
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(exc)
            _jobs[job_id]["finished_at"] = datetime.utcnow().isoformat()


def enqueue_job(fn: Callable[[], Any], meta: dict | None = None) -> str:
    job_id = uuid.uuid4().hex
    job = {
        "id": job_id,
        "status": "queued",
        "meta": meta or {},
        "created_at": datetime.utcnow().isoformat(),
    }
    with _jobs_lock:
        _jobs[job_id] = job

    thread = threading.Thread(target=_run_job, args=(job_id, fn), daemon=True)
    thread.start()
    return job_id


def get_job(job_id: str) -> dict | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        # Return a copy so callers cannot accidentally mutate internal state
        return dict(job) if job is not None else None


def list_jobs() -> list[dict]:
    with _jobs_lock:
        return [dict(job) for job in _jobs.values()]


# Convenience wrapper for embedding training
def enqueue_embeddings_training(fn: Callable[[], Any], meta: dict | None = None) -> str:
    return enqueue_job(fn, meta=meta)
