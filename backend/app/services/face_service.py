from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

from app.core.config import settings


_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))


class FaceService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._recognizer = None

    def load_models(self) -> None:
        with self._lock:
            if self._recognizer is not None:
                return
            from face_recognition.embedding_recognizer import FaceEmbeddingRecognizer  # noqa: WPS433
            from utils.config import load_config  # noqa: WPS433

            cfg = load_config(base_dir=_ROOT)
            recognizer = FaceEmbeddingRecognizer(cfg)
            recognizer.load_model()
            self._recognizer = recognizer

    def recognize_student(self, frame):
        if self._recognizer is None:
            raise RuntimeError("Face models not loaded")

        start = time.perf_counter()
        results = self._recognizer.recognize_frame(frame)
        duration = time.perf_counter() - start

        known = [r for r in results if r.get("is_known")]
        if not known:
            return {"matched": False, "processing_time": duration}

        best = max(known, key=lambda x: x.get("confidence", 0.0))
        if duration > settings.face_timeout_seconds:
            return {"matched": False, "reason": "timeout", "processing_time": duration}

        if best["confidence"] < settings.face_confidence_threshold:
            return {
                "matched": False,
                "reason": "below_threshold",
                "confidence": float(best["confidence"]),
                "processing_time": duration,
            }

        return {
            "matched": True,
            "student_number": str(best["student_id"]),
            "confidence": float(best["confidence"]),
            "processing_time": duration,
        }


face_service = FaceService()
