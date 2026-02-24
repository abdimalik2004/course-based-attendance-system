from pathlib import Path
import platform
from collections import defaultdict, deque

import cv2
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN

from face_recognition.anti_spoof import AntiSpoofModel
from utils.logging import get_logger


logger = get_logger(__name__)


class FaceEmbeddingRecognizer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cpu")
        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=max(20, int(config.min_face_size)),
            device=self.device,
        )
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        self.mean_embeddings = None
        self.mean_student_ids = None
        self.anti_spoof = AntiSpoofModel(config)
        self.spoof_windows = defaultdict(lambda: deque(maxlen=self.config.anti_spoof_required_frames))

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
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        boxes, _ = self.mtcnn.detect(image)
        results = []
        if boxes is None or len(boxes) == 0:
            return results

        faces = self.mtcnn.extract(image, boxes, save_path=None)
        if faces is None or len(faces) == 0:
            return results

        if isinstance(faces, list):
            faces = torch.stack(faces, dim=0)
        if faces.dim() == 3:
            faces = faces.unsqueeze(0)

        with torch.no_grad():
            embeddings = self.model(faces.to(self.device)).cpu().numpy()

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        for box, emb in zip(boxes, embeddings):
            x1, y1, x2, y2 = box.astype(int).tolist()
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
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

    def run_webcam(self, course_id, on_recognized, camera_index=None, stop_event=None):
        index = self.config.camera_index if camera_index is None else camera_index
        cap = self._open_camera(index)
        if not cap.isOpened():
            raise RuntimeError("Camera not available")
        self._apply_resolution(cap)

        match_counts = {}
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read from camera")
                break

            results = self.recognize_frame(frame)
            seen_ids = set()
            for result in results:
                if result["is_known"]:
                    x, y, w, h = result["bbox"]
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
