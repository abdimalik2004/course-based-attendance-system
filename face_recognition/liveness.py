from collections import deque
from dataclasses import dataclass
from time import monotonic

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None


@dataclass
class LivenessResult:
    passed: bool
    motion_score: float
    blinked: bool
    head_turn: bool
    prompt: str


class MotionLivenessChecker:
    def __init__(
        self,
        history_size: int = 5,
        min_motion_frames: int = 2,
        min_mean_delta: float = 12.0,
        min_bbox_shift: float = 6.0,
        roi_size: int = 96,
    ):
        self.history = deque(maxlen=history_size)
        self.min_motion_frames = min_motion_frames
        self.min_mean_delta = min_mean_delta
        self.min_bbox_shift = min_bbox_shift
        self.roi_size = roi_size
        self._prev_roi = None
        self._prev_bbox = None

    def update(self, gray_frame, bbox):
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            self._reset()
            return LivenessResult(False, 0.0, False, False, "")

        roi = gray_frame[y : y + h, x : x + w]
        if roi.size == 0:
            self._reset()
            return LivenessResult(False, 0.0, False, False, "")

        roi = cv2.resize(roi, (self.roi_size, self.roi_size), interpolation=cv2.INTER_AREA)
        motion_score = 0.0
        bbox_shift = 0.0

        if self._prev_roi is not None:
            diff = cv2.absdiff(roi, self._prev_roi)
            motion_score = float(np.mean(diff))

        if self._prev_bbox is not None:
            prev_x, prev_y, prev_w, prev_h = self._prev_bbox
            cx, cy = x + w / 2.0, y + h / 2.0
            prev_cx, prev_cy = prev_x + prev_w / 2.0, prev_y + prev_h / 2.0
            bbox_shift = float(((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5)

        motion_detected = motion_score >= self.min_mean_delta or bbox_shift >= self.min_bbox_shift
        self.history.append(1 if motion_detected else 0)
        self._prev_roi = roi
        self._prev_bbox = (x, y, w, h)

        passed = sum(self.history) >= self.min_motion_frames
        return LivenessResult(passed, motion_score, False, False, "")

    def _reset(self):
        self.history.clear()
        self._prev_roi = None
        self._prev_bbox = None


class ChallengeLivenessChecker:
    def __init__(
        self,
        history_size: int = 5,
        min_motion_frames: int = 2,
        min_mean_delta: float = 12.0,
        min_bbox_shift: float = 6.0,
        roi_size: int = 96,
        blink_ear_threshold: float = 0.21,
        blink_consec_frames: int = 1,
        step_timeout_seconds: float | None = None,
        pass_window_seconds: float = 10.0,
    ):
        self.motion = MotionLivenessChecker(
            history_size=history_size,
            min_motion_frames=min_motion_frames,
            min_mean_delta=min_mean_delta,
            min_bbox_shift=min_bbox_shift,
            roi_size=roi_size,
        )
        if mp is None or not hasattr(mp, "solutions"):
            raise RuntimeError(
                "mediapipe with solutions is required. Install mediapipe==0.10.14 in your venv."
            )
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.blink_ear_threshold = blink_ear_threshold
        self.blink_consec_frames = blink_consec_frames
        self.step_timeout_seconds = step_timeout_seconds
        self.pass_window_seconds = pass_window_seconds
        self._blink_frames = 0
        self._blinked = False
        self._step = "blink"
        self._step_started = monotonic()
        self._passed_until = 0.0

    def update(self, frame, bbox):
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            self.reset()
            return LivenessResult(False, 0.0, False, False, "")

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_result = self.motion.update(gray_frame, bbox)
        face_roi = frame[y : y + h, x : x + w]
        if face_roi.size == 0:
            self.reset()
            return LivenessResult(False, 0.0, False, False, "")

        rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return LivenessResult(False, motion_result.motion_score, self._blinked, False, "")

        landmarks = results.multi_face_landmarks[0].landmark
        ear = self._eye_aspect_ratio(landmarks, w, h)
        if ear < self.blink_ear_threshold:
            self._blink_frames += 1
        else:
            if self._blink_frames >= self.blink_consec_frames:
                self._blinked = True
            self._blink_frames = 0

        prompt = ""
        now = monotonic()
        if now <= self._passed_until:
            return LivenessResult(True, motion_result.motion_score, True, False, "")
        if self.step_timeout_seconds is not None:
            if now - self._step_started > self.step_timeout_seconds:
                self._step = "blink"
                self._step_started = now
                self._blinked = False

        if self._step == "blink":
            prompt = "Blink now"

        passed = motion_result.passed and self._blinked
        if passed:
            self._passed_until = now + self.pass_window_seconds
        return LivenessResult(passed, motion_result.motion_score, self._blinked, False, prompt)

    def reset(self):
        self.motion._reset()
        self._blink_frames = 0
        self._blinked = False
        self._step = "blink"
        self._step_started = monotonic()
        self._passed_until = 0.0

    def _eye_aspect_ratio(self, landmarks, width, height):
        left = [33, 160, 158, 133, 153, 144]
        right = [362, 385, 387, 263, 373, 380]
        left_ear = self._ear_for_eye(landmarks, left, width, height)
        right_ear = self._ear_for_eye(landmarks, right, width, height)
        return (left_ear + right_ear) / 2.0

    def _ear_for_eye(self, landmarks, idxs, width, height):
        points = [
            (landmarks[i].x * width, landmarks[i].y * height) for i in idxs
        ]
        p1, p2, p3, p4, p5, p6 = points
        vertical1 = np.linalg.norm(np.array(p2) - np.array(p6))
        vertical2 = np.linalg.norm(np.array(p3) - np.array(p5))
        horizontal = np.linalg.norm(np.array(p1) - np.array(p4)) + 1e-6
        return (vertical1 + vertical2) / (2.0 * horizontal)

