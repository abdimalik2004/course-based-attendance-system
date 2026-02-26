import cv2
import numpy as np


class OcclusionChecker:
    def __init__(self, config):
        self.enabled = bool(getattr(config, "occlusion_check_enabled", True))
        self.min_eyes_visible = int(getattr(config, "occlusion_min_eyes_visible", 2))
        self.min_eye_variance = float(getattr(config, "occlusion_min_eye_variance", 120.0))
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

    def check(self, frame, bbox):
        if not self.enabled:
            return True, "disabled"

        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return False, "invalid_bbox"

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_roi = gray[y : y + h, x : x + w]
        if face_roi.size == 0:
            return False, "empty_face"

        min_eye = max(10, int(min(w, h) * 0.12))
        eyes = self.eye_cascade.detectMultiScale(
            face_roi,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(min_eye, min_eye),
        )

        valid_eyes = []
        upper_limit = int(h * 0.65)
        for (ex, ey, ew, eh) in eyes:
            if ey + eh > upper_limit:
                continue
            eye_roi = face_roi[ey : ey + eh, ex : ex + ew]
            if eye_roi.size == 0:
                continue
            variance = float(np.var(eye_roi))
            if variance < self.min_eye_variance:
                continue
            valid_eyes.append((ex, ey, ew, eh))

        if len(valid_eyes) < self.min_eyes_visible:
            return False, "eyes_covered"

        return True, "ok"
