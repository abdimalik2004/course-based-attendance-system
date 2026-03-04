import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None


class OcclusionChecker:
    LEFT_EYE_IDX = [33, 133, 159, 145, 153, 154, 155, 157, 158, 160, 161, 246]
    RIGHT_EYE_IDX = [362, 263, 386, 374, 380, 381, 382, 384, 385, 387, 388, 466]

    def __init__(self, config):
        self.enabled = bool(getattr(config, "occlusion_check_enabled", True))
        self.requested_backend = str(getattr(config, "occlusion_backend", "auto")).lower()
        self.min_eyes_visible = int(getattr(config, "occlusion_min_eyes_visible", 2))
        self.min_eye_variance = float(getattr(config, "occlusion_min_eye_variance", 120.0))
        self.min_laplacian_variance = float(getattr(config, "occlusion_min_laplacian_variance", 35.0))
        self.min_edge_density = float(getattr(config, "occlusion_min_edge_density", 0.05))
        self.max_dark_ratio = float(getattr(config, "occlusion_max_dark_ratio", 0.70))
        self.face_mesh = None

        if self.requested_backend in {"auto", "mediapipe"} and mp is not None:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=5,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

        self.backend = "mediapipe" if self.face_mesh is not None else "heuristic"

    def check(self, frame, bbox):
        if not self.enabled:
            return True, "disabled"

        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return False, "invalid_bbox"

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.face_mesh is not None:
            mp_result = self._check_mediapipe(frame, gray, bbox)
            if mp_result is not None:
                if mp_result[0]:
                    return mp_result

                # If MediaPipe flags occlusion, try a relaxed uncovered-face hint
                # to avoid false rejection after glasses removal.
                if self._uncovered_face_hint(gray, bbox):
                    return True, "ok_relaxed_uncovered"
                return mp_result

        # When MediaPipe is unavailable, use lightweight heuristic fallback.
        if self._uncovered_face_hint(gray, bbox):
            return True, "ok_heuristic"
        return False, "eyes_covered_heuristic"

    def _uncovered_face_hint(self, gray, bbox):
        x, y, w, h = bbox
        face_roi = gray[y : y + h, x : x + w]
        if face_roi.size == 0:
            return False

        top = int(h * 0.20)
        bottom = int(h * 0.58)
        left = int(w * 0.15)
        right = int(w * 0.85)
        eye_band = face_roi[top:bottom, left:right]
        if eye_band.size == 0:
            return False

        lap = cv2.Laplacian(eye_band, cv2.CV_64F)
        lap_var = float(np.var(lap))
        dark_ratio = float(np.mean(eye_band < 50))

        # Heuristic: uncovered eyes usually keep moderate texture and are not heavily dark.
        return lap_var >= 12.0 and dark_ratio <= 0.60

    def _check_mediapipe(self, frame, gray, bbox):
        x, y, w, h = bbox
        frame_h, frame_w = gray.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mesh_result = self.face_mesh.process(rgb)
        if not mesh_result.multi_face_landmarks:
            return None

        target = self._select_matching_face(mesh_result.multi_face_landmarks, bbox, frame_w, frame_h)
        if target is None:
            return None

        left_roi = self._extract_eye_roi(gray, target.landmark, self.LEFT_EYE_IDX, frame_w, frame_h)
        right_roi = self._extract_eye_roi(gray, target.landmark, self.RIGHT_EYE_IDX, frame_w, frame_h)
        if left_roi is None or right_roi is None:
            return None

        visible_eyes = 0
        if self._eye_quality_ok(left_roi):
            visible_eyes += 1
        if self._eye_quality_ok(right_roi):
            visible_eyes += 1

        if visible_eyes < self.min_eyes_visible:
            return False, "eyes_covered_mediapipe"

        return True, "ok_mediapipe"

    def _select_matching_face(self, faces, bbox, frame_w, frame_h):
        x, y, w, h = bbox
        x2, y2 = x + w, y + h
        best_face = None
        best_iou = 0.0

        for face in faces:
            xs = []
            ys = []
            for lm in face.landmark:
                px = int(lm.x * frame_w)
                py = int(lm.y * frame_h)
                xs.append(px)
                ys.append(py)

            if not xs or not ys:
                continue

            fx1 = max(0, min(xs))
            fy1 = max(0, min(ys))
            fx2 = min(frame_w, max(xs))
            fy2 = min(frame_h, max(ys))
            iou = self._iou((x, y, x2, y2), (fx1, fy1, fx2, fy2))
            if iou > best_iou:
                best_iou = iou
                best_face = face

        if best_iou < 0.10:
            return None
        return best_face

    def _extract_eye_roi(self, gray, landmarks, indices, frame_w, frame_h):
        points = []
        for idx in indices:
            if idx >= len(landmarks):
                continue
            lm = landmarks[idx]
            px = int(lm.x * frame_w)
            py = int(lm.y * frame_h)
            points.append((px, py))

        if len(points) < 4:
            return None

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x1 = max(0, min(xs))
        y1 = max(0, min(ys))
        x2 = min(frame_w, max(xs))
        y2 = min(frame_h, max(ys))

        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            return None

        pad_x = max(2, int(width * 0.25))
        pad_y = max(2, int(height * 0.5))
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(frame_w, x2 + pad_x)
        y2 = min(frame_h, y2 + pad_y)

        roi = gray[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        return roi

    def _eye_quality_ok(self, eye_roi):
        if eye_roi is None or eye_roi.size == 0:
            return False

        variance = float(np.var(eye_roi))
        if variance < self.min_eye_variance:
            return False

        lap = cv2.Laplacian(eye_roi, cv2.CV_64F)
        lap_var = float(np.var(lap))
        if lap_var < self.min_laplacian_variance:
            return False

        edges = cv2.Canny(eye_roi, 50, 150)
        edge_density = float(np.mean(edges > 0))
        if edge_density < self.min_edge_density:
            return False

        dark_ratio = float(np.mean(eye_roi < 50))
        if dark_ratio > self.max_dark_ratio:
            return False

        return True

    def _iou(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return float(inter) / float(union)
