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

    def _ensure_detect_recognizer(self):
        """
        Return a recognizer suitable for detection + embedding extraction only.
        Deliberately skips load_model() so it works even before any student is trained.
        Uses a module-level cache so we don't re-initialise on every call.
        """
        if getattr(self, "_detect_recognizer", None) is not None:
            return self._detect_recognizer
        try:
            from face_recognition.embedding_recognizer import FaceEmbeddingRecognizer  # noqa: WPS433
            from utils.config import load_config  # noqa: WPS433

            cfg = load_config(base_dir=_ROOT)
            rec = FaceEmbeddingRecognizer(cfg)
            # Do NOT call rec.load_model() — we only need the detector + facenet backbone
            self._detect_recognizer = rec
        except Exception:
            self._detect_recognizer = None
        return self._detect_recognizer

    def detect_faces(self, frame) -> dict:
        """
        Detect faces in a frame and extract an embedding for the largest face.
        Works without trained student data — used during image capture for live
        face-count validation and person-identity tracking.

        Returns:
            {
                "face_count": int,
                "embedding": list[float] | None  # 512-d unit vector of the largest face
            }
        """
        import numpy as np
        import torch

        # Prefer the main recognizer (already loaded from attendance sessions)
        with self._lock:
            recognizer = self._recognizer

        # Fallback to the detect-only recognizer when no model is trained yet
        if recognizer is None:
            recognizer = self._ensure_detect_recognizer()

        if recognizer is None:
            return {"face_count": 0, "embedding": None}

        try:
            _, detections = recognizer.detector.detect_with_keypoints(frame)
            face_count = len(detections)

            embedding = None
            if face_count > 0:
                # Use the largest face (most likely the main subject)
                largest = max(detections, key=lambda d: d["bbox"][2] * d["bbox"][3])
                face_tensor = recognizer._extract_face_tensor(frame, largest["bbox"])
                if face_tensor is not None:
                    with torch.no_grad():
                        emb = recognizer.model(face_tensor.to(recognizer.device)).cpu().numpy()[0]
                    emb = emb / (np.linalg.norm(emb) + 1e-12)
                    embedding = emb.tolist()

            return {"face_count": face_count, "embedding": embedding}
        except Exception:
            return {"face_count": 0, "embedding": None}

    def recognize_student(self, frame):
        with self._lock:
            self._ensure_current_models()
            recognizer = self._recognizer

        if recognizer is None:
            raise RuntimeError(self._load_error or "Face models not loaded")

        start = time.perf_counter()
        results = recognizer.recognize_frame(frame)
        duration = time.perf_counter() - start

        # No face at all detected in the frame
        if not results:
            return {"matched": False, "reason": "no_face", "processing_time": duration}

        known = [r for r in results if r.get("is_known")]
        if not known:
            # At least one face was detected by the detector but it was not matched
            # to any enrolled student. This commonly happens when the face is partially
            # occluded (mask, sunglasses, hand over face).
            return {"matched": False, "reason": "not_recognized", "processing_time": duration}

        best = max(known, key=lambda x: x.get("confidence", 0.0))
        if duration > settings.face_timeout_seconds:
            return {"matched": False, "reason": "timeout", "processing_time": duration}

        if best["confidence"] < settings.face_confidence_threshold:
            # Face matched but confidence too low — often means the face is partially
            # occluded or the image quality is too poor for a reliable embedding.
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
