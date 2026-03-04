import cv2
import numpy as np

from utils.logging import get_logger

try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None


logger = get_logger(__name__)


class FaceDetector:
    def __init__(self, config):
        self.config = config
        self.det_size = int(getattr(config, "scrfd_det_size", 640))
        self.det_threshold = float(getattr(config, "scrfd_threshold", 0.5))
        self.max_faces = int(getattr(config, "scrfd_max_faces", 20))

        self.quality_check_enabled = bool(getattr(config, "quality_check_enabled", False))
        self.min_blur_variance = float(getattr(config, "quality_min_blur_variance", 60.0))
        self.min_brightness = float(getattr(config, "quality_min_brightness", 45.0))
        self.max_brightness = float(getattr(config, "quality_max_brightness", 210.0))

        if FaceAnalysis is None:
            raise RuntimeError(
                "SCRFD detector requires insightface. Install it with: pip install insightface"
            )

        providers = ["CPUExecutionProvider"]
        self.app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.app.prepare(ctx_id=-1, det_size=(self.det_size, self.det_size), det_thresh=self.det_threshold)
        logger.info(
            "SCRFD detector initialized (det_size=%d threshold=%.2f max_faces=%d)",
            self.det_size,
            self.det_threshold,
            self.max_faces,
        )

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.app.get(frame)
        boxes = []
        for face in faces[: self.max_faces]:
            x1, y1, x2, y2 = [int(v) for v in face.bbox.tolist()]
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            boxes.append((x1, y1, w, h))
        return gray, boxes

    def detect_with_keypoints(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected = []
        faces = self.app.get(frame)
        for face in faces[: self.max_faces]:
            x1, y1, x2, y2 = [int(v) for v in face.bbox.tolist()]
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            item = {
                "bbox": (x1, y1, w, h),
                "score": float(getattr(face, "det_score", 0.0)),
                "kps": np.array(getattr(face, "kps", None), dtype=np.float32)
                if getattr(face, "kps", None) is not None
                else None,
            }
            detected.append(item)
        return gray, detected

    def passes_quality(self, frame, bbox):
        if not self.quality_check_enabled:
            return True, "disabled"

        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return False, "invalid_bbox"

        h_img, w_img = frame.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w_img, x + w)
        y2 = min(h_img, y + h)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False, "empty_roi"

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))

        if blur_var < self.min_blur_variance:
            return False, "too_blurry"
        if brightness < self.min_brightness:
            return False, "too_dark"
        if brightness > self.max_brightness:
            return False, "too_bright"

        return True, "ok"
