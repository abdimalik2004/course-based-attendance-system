from pathlib import Path
import platform
from collections import defaultdict, deque

import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

from face_recognition.anti_spoof import AntiSpoofModel
from face_recognition.detector import FaceDetector
from face_recognition.occlusion import OcclusionChecker
from utils.logging import get_logger


logger = get_logger(__name__)


class FaceEmbeddingRecognizer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cpu")
        self.detector = FaceDetector(config)
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self.mean_embeddings = None
        self.mean_student_ids = None
        self.anti_spoof = AntiSpoofModel(config)
        self.occlusion = OcclusionChecker(config)
        self.spoof_windows = defaultdict(lambda: deque(maxlen=self.config.anti_spoof_required_frames))
        self.occlusion_windows = defaultdict(lambda: deque(maxlen=self.config.occlusion_required_frames))
        self.display_available = True

    def load_model(self):
        embed_path = Path(self.config.embedding_path)
        if not embed_path.exists():
            raise RuntimeError("Embedding file not found. Train the model first.")

        data = np.load(embed_path, allow_pickle=True)
        embeddings = data["embeddings"]
        student_ids = data["student_ids"]
        self._build_index(embeddings, student_ids)

    def _build_index(self, embeddings, student_ids):
        by_student = {}
        for emb, student_id in zip(embeddings, student_ids):
            by_student.setdefault(student_id, []).append(emb)

        mean_embeddings = []
        mean_student_ids = []
        for student_id, embs in sorted(by_student.items()):
            stacked = np.stack(embs, axis=0)
            mean = stacked.mean(axis=0)
            mean = mean / (np.linalg.norm(mean) + 1e-12)
            mean_embeddings.append(mean)
            mean_student_ids.append(student_id)

        self.mean_embeddings = np.stack(mean_embeddings, axis=0)
        self.mean_student_ids = mean_student_ids

    def recognize_frame(self, frame):
        _, detections = self.detector.detect_with_keypoints(frame)
        results = []
        if detections is None or len(detections) == 0:
            return results

        for detection in detections:
            x1, y1, w, h = detection["bbox"]
            if w < self.config.min_face_size or h < self.config.min_face_size:
                results.append(
                    {
                        "bbox": (x1, y1, w, h),
                        "student_id": None,
                        "confidence": 0.0,
                        "is_known": False,
                    }
                )
                continue

            quality_ok, _ = self.detector.passes_quality(frame, (x1, y1, w, h))
            if not quality_ok:
                results.append(
                    {
                        "bbox": (x1, y1, w, h),
                        "student_id": None,
                        "confidence": 0.0,
                        "is_known": False,
                    }
                )
                continue

            face_tensor = self._extract_face_tensor(frame, (x1, y1, w, h))
            if face_tensor is None:
                results.append(
                    {
                        "bbox": (x1, y1, w, h),
                        "student_id": None,
                        "confidence": 0.0,
                        "is_known": False,
                    }
                )
                continue

            with torch.no_grad():
                emb = self.model(face_tensor.to(self.device)).cpu().numpy()[0]

            emb = emb / (np.linalg.norm(emb) + 1e-12)
            scores = np.dot(self.mean_embeddings, emb)
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            is_known = best_score >= self.config.embedding_min_similarity
            student_id = self.mean_student_ids[best_idx] if is_known else None
            results.append(
                {
                    "bbox": (x1, y1, w, h),
                    "student_id": student_id,
                    "confidence": best_score,
                    "is_known": is_known,
                }
            )
        return results

    def _extract_face_tensor(self, frame, bbox):
        x, y, w, h = bbox
        h_img, w_img = frame.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)
        if x2 <= x1 or y2 <= y1:
            return None

        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            return None

        resized = cv2.resize(face, (160, 160), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        norm = (rgb - 127.5) / 128.0
        tensor = torch.from_numpy(np.transpose(norm, (2, 0, 1))).unsqueeze(0)
        return tensor

    def run_webcam(self, course_id, on_recognized, camera_index=None, stop_event=None):
        index = self.config.camera_index if camera_index is None else camera_index
        cap = self._open_camera(index)
        if not cap.isOpened():
            raise RuntimeError("Camera not available")
        self._apply_resolution(cap)
        self.display_available = self._ensure_display_available()

        anti_spoof_status = self.anti_spoof.get_status()
        logger.info(
            "Startup status | anti_spoof: enabled=%s backend=%s configured=%s model=%s input=%s live_index=%s threshold=%.2f",
            anti_spoof_status["enabled"],
            anti_spoof_status["backend"],
            anti_spoof_status["configured_backend"],
            anti_spoof_status["model_path"],
            anti_spoof_status["input_size"],
            anti_spoof_status["live_index"],
            anti_spoof_status["threshold"],
        )
        logger.info(
            "Startup status | occlusion: enabled=%s backend=%s requested=%s min_eyes=%d min_eye_variance=%.1f min_lap_var=%.1f min_edge_density=%.2f max_dark_ratio=%.2f frames=%d pass_ratio=%.2f",
            self.occlusion.enabled,
            self.occlusion.backend,
            self.occlusion.requested_backend,
            self.occlusion.min_eyes_visible,
            self.occlusion.min_eye_variance,
            self.occlusion.min_laplacian_variance,
            self.occlusion.min_edge_density,
            self.occlusion.max_dark_ratio,
            self.config.occlusion_required_frames,
            self.config.occlusion_min_pass_ratio,
        )

        match_counts = {}
        process_every_n = max(1, int(getattr(self.config, "process_every_n_frames", 1)))
        frame_count = 0
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read from camera")
                break

            frame_count += 1
            if process_every_n > 1 and (frame_count % process_every_n) != 0:
                if self.display_available:
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
                continue

            results = self.recognize_frame(frame)
            seen_ids = set()
            seen_known_ids = set()
            for result in results:
                if result["is_known"]:
                    x, y, w, h = result["bbox"]
                    student_id = result["student_id"]
                    seen_known_ids.add(student_id)

                    visible, reason = self.occlusion.check(frame, (x, y, w, h))
                    occ_window = self.occlusion_windows[student_id]
                    occ_window.append(1 if visible else 0)
                    occ_len = len(occ_window)
                    occ_ratio = (sum(occ_window) / occ_len) if occ_len else 0.0
                    stable_visible = (
                        occ_len >= self.config.occlusion_required_frames
                        and occ_ratio >= self.config.occlusion_min_pass_ratio
                    )

                    # Recovery behavior:
                    # - If current frame is visible, allow recognition immediately.
                    # - If current frame is not visible, block only when occlusion is persistent.
                    should_block_occlusion = False
                    if not visible:
                        if occ_len < self.config.occlusion_required_frames:
                            should_block_occlusion = True
                        else:
                            should_block_occlusion = not stable_visible

                    if should_block_occlusion:
                        logger.info("Occlusion check failed (%s)", reason)
                        cv2.putText(
                            frame,
                            "Face uncovered required",
                            (x, y + h + 16),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 165, 255),
                            1,
                        )
                        logger.info(
                            "Occlusion gate blocked (student=%s reason=%s pass_ratio=%.2f need_ratio=%.2f need_frames=%d got=%d)",
                            student_id,
                            reason,
                            occ_ratio,
                            self.config.occlusion_min_pass_ratio,
                            self.config.occlusion_required_frames,
                            occ_len,
                        )
                        continue

                    is_live, spoof_score = self.anti_spoof.check(frame, (x, y, w, h))
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
                    seen_ids.add(student_id)
                    match_counts[student_id] = match_counts.get(student_id, 0) + 1
                    if match_counts[student_id] >= self.config.required_matches:
                        on_recognized(result)
                else:
                    logger.info("Unknown face detected (similarity=%.2f)", result["confidence"])

                x, y, w, h = result["bbox"]
                label = result["student_id"] if result["is_known"] else "Unknown"
                text = f"{label} {result['confidence']:.2f}"
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
                cv2.putText(frame, text, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            for student_id in list(match_counts.keys()):
                if student_id not in seen_ids:
                    match_counts.pop(student_id, None)

            for student_id in list(self.spoof_windows.keys()):
                if student_id not in seen_known_ids:
                    self.spoof_windows.pop(student_id, None)

            for student_id in list(self.occlusion_windows.keys()):
                if student_id not in seen_known_ids:
                    self.occlusion_windows.pop(student_id, None)

            if self.display_available:
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
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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

    def _ensure_display_available(self):
        try:
            cv2.namedWindow("__attendance_display_test__", cv2.WINDOW_NORMAL)
            cv2.destroyWindow("__attendance_display_test__")
            return True
        except cv2.error as exc:
            logger.warning(
                "OpenCV GUI backend is unavailable; running in headless mode (no preview window). "
                "If you want preview UI, install GUI-enabled OpenCV in this venv: "
                "pip uninstall -y opencv-python-headless opencv-python opencv-contrib-python-headless && "
                "pip install opencv-contrib-python"
            )
            logger.debug("Display check failure: %s", exc)
            return False
