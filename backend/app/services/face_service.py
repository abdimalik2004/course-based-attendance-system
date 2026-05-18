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
        self._model_signature: tuple[int | None, int | None, int | None] | None = None
        self._failed_model_signature: tuple[int | None, int | None, int | None] | None = None
        self._load_error: str | None = None

    def _current_model_signature(self) -> tuple[int | None, int | None, int | None]:
        from utils.config import load_config  # noqa: WPS433

        cfg = load_config(base_dir=_ROOT)

        def _mtime(path: Path) -> int | None:
            try:
                return int(path.stat().st_mtime_ns)
            except FileNotFoundError:
                return None

        return (_mtime(Path(cfg.model_path)), _mtime(Path(cfg.label_map_path)), _mtime(Path(cfg.embedding_path)))

    def _load_models_locked(self) -> None:
        from face_recognition.embedding_recognizer import FaceEmbeddingRecognizer  # noqa: WPS433
        from utils.config import load_config  # noqa: WPS433

        cfg = load_config(base_dir=_ROOT)
        current_signature = self._current_model_signature()
        try:
            recognizer = FaceEmbeddingRecognizer(cfg)
            recognizer.load_model()
        except Exception as exc:
            self._recognizer = None
            self._model_signature = None
            self._failed_model_signature = current_signature
            self._load_error = str(exc)
            return

        self._recognizer = recognizer
        self._model_signature = current_signature
        self._failed_model_signature = None
        self._load_error = None

    def load_models(self, force: bool = False) -> None:
        with self._lock:
            current_signature = self._current_model_signature()
            if not force and self._recognizer is not None and self._model_signature == current_signature:
                return
            if not force and self._recognizer is None and self._failed_model_signature == current_signature:
                return
            self._load_models_locked()

    def reload_models(self) -> None:
        self.load_models(force=True)

    def _ensure_current_models(self) -> None:
        current_signature = self._current_model_signature()
        if self._recognizer is None and self._failed_model_signature == current_signature:
            return
        if self._recognizer is None or self._model_signature != current_signature:
            self._load_models_locked()

    def recognize_student(self, frame):
        with self._lock:
            self._ensure_current_models()
            recognizer = self._recognizer

        if recognizer is None:
            raise RuntimeError(self._load_error or "Face models not loaded")

        start = time.perf_counter()
        results = recognizer.recognize_frame(frame)
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
