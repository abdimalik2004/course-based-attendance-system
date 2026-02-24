import json
import platform
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2

from face_recognition.anti_spoof import AntiSpoofModel
from face_recognition.detector import FaceDetector
from utils.logging import get_logger


logger = get_logger(__name__)


class FaceRecognizer:
    def __init__(self, config):
        self.config = config
        self.detector = FaceDetector(config.haar_cascade_path)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.label_map = {}
        self.anti_spoof = AntiSpoofModel(config)
        self.spoof_windows = defaultdict(lambda: deque(maxlen=self.config.anti_spoof_required_frames))

    def load_model(self):
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise RuntimeError("Model file not found. Train the model first.")
        self.recognizer.read(str(model_path))

        map_path = Path(self.config.label_map_path)
        if not map_path.exists():
            raise RuntimeError("Label map not found. Train the model first.")
        with open(map_path, "r", encoding="utf-8") as f:
            self.label_map = json.load(f)

    def recognize_frame(self, frame):
        gray, faces = self.detector.detect(frame)
        results = []
        for (x, y, w, h) in faces:
            roi = gray[y : y + h, x : x + w]
            label, confidence = self.recognizer.predict(roi)
            # LBPH returns a distance; lower is a better match.
            is_known = confidence <= self.config.confidence_threshold
            student_id = self.label_map.get("labels", {}).get(str(label)) if is_known else None
            results.append(
                {
                    "bbox": (x, y, w, h),
                    "label": label,
                    "student_id": student_id,
                    "confidence": float(confidence),
                    "is_known": is_known,
                }
            )
        return results

    def run_webcam(self, course_id, on_recognized, camera_index=None, stop_event=None):
        # Ethical note: use explicit consent for biometric processing and limit data retention.
        index = self.config.camera_index if camera_index is None else camera_index
        cap = self._open_camera(index)
        if not cap.isOpened():
            raise RuntimeError("Camera not available")
        self._apply_resolution(cap)

        match_counts = {}
        start_time = time.monotonic()
        max_runtime = self.config.max_runtime_seconds

        while True:
            if stop_event is not None and stop_event.is_set():
                break
            if max_runtime > 0 and (time.monotonic() - start_time) >= max_runtime:
                logger.info("Max runtime reached (%ds). Stopping recognition.", max_runtime)
                break
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read from camera")
                break

            results = self.recognize_frame(frame)
            seen_ids = set()
            for result in results:
                x, y, w, h = result["bbox"]
                if w < self.config.min_face_size or h < self.config.min_face_size:
                    logger.info(
                        "Face too small for reliable match (size=%dx%d)",
                        w,
                        h,
                    )
                    continue

                if result["is_known"]:
                    is_live, spoof_score = self.anti_spoof.check(frame, (x, y, w, h))
                    student_id = result["student_id"]
                    window = self.spoof_windows[student_id]
                    if spoof_score > 0.0:
                        window.append(spoof_score)
                    window_len = len(window)
                    avg_score = (sum(window) / window_len) if window_len else 0.0
                    pass_count = sum(1 for s in window if s >= self.config.anti_spoof_threshold)
                    pass_ratio = (pass_count / window_len) if window_len else 0.0
                    stable_live = (
                        window_len >= self.config.anti_spoof_required_frames
                        and pass_ratio >= self.config.anti_spoof_min_pass_ratio
                        and avg_score >= (self.config.anti_spoof_threshold + self.config.anti_spoof_margin)
                    )

                    if not is_live or not stable_live:
                        logger.info(
                            "Anti-spoof failed (score=%.2f avg=%.2f threshold=%.2f margin=%.2f pass_ratio=%.2f need_ratio=%.2f need_frames=%d got=%d)",
                            spoof_score,
                            avg_score,
                            self.config.anti_spoof_threshold,
                            self.config.anti_spoof_margin,
                            pass_ratio,
                            self.config.anti_spoof_min_pass_ratio,
                            self.config.anti_spoof_required_frames,
                            window_len,
                        )
                        cv2.putText(
                            frame,
                            f"Spoof suspected {spoof_score:.2f}/{avg_score:.2f}",
                            (x, y + h + 16),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 165, 255),
                            1,
                        )
                        continue
                    else:
                        logger.info(
                            "Anti-spoof passed (score=%.2f avg=%.2f)",
                            spoof_score,
                            avg_score,
                        )

                if result["is_known"]:
                    student_id = result["student_id"]
                    seen_ids.add(student_id)
                    match_counts[student_id] = match_counts.get(student_id, 0) + 1
                    if match_counts[student_id] >= self.config.required_matches:
                        on_recognized(result)
                else:
                    logger.info("Unknown face detected (confidence=%.2f)", result["confidence"])

                label = result["student_id"] if result["is_known"] else "Unknown"
                text = f"{label} {result['confidence']:.2f}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
                cv2.putText(frame, text, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Reset counts for faces that left the frame.
            for student_id in list(match_counts.keys()):
                if student_id not in seen_ids:
                    match_counts.pop(student_id, None)

            cv2.imshow(f"Attendance - {course_id}", frame)
            if self.config.preview_width and self.config.preview_height:
                preview = cv2.resize(
                    frame,
                    (self.config.preview_width, self.config.preview_height),
                    interpolation=cv2.INTER_AREA,
                )
                cv2.imshow(f"Attendance - {course_id}", preview)
            else:
                cv2.imshow(f"Attendance - {course_id}", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    def _open_camera(self, index):
        if platform.system().lower().startswith("win"):
            for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    return cap
                cap.release()
        return cv2.VideoCapture(index)

    def _apply_resolution(self, cap):
        candidates = []
        if self.config.camera_width and self.config.camera_height:
            candidates.append((self.config.camera_width, self.config.camera_height))
        candidates.extend([(1280, 720), (640, 480)])

        for width, height in candidates:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            ok, _ = cap.read()
            if ok:
                return
