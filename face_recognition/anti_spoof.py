import json
from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

from utils.logging import get_logger


logger = get_logger(__name__)


class AntiSpoofModel:
    def __init__(self, config):
        self.config = config
        self.threshold = float(config.anti_spoof_threshold)
        self.enabled = bool(config.anti_spoof_enabled)
        self.backend = str(config.anti_spoof_backend).lower()
        self.input_size = int(config.anti_spoof_input_size)
        self.live_index = int(config.anti_spoof_live_index)
        self.use_onnxruntime = bool(config.anti_spoof_use_onnxruntime)
        self.model_path = str(config.anti_spoof_model_path)
        self.onnx_net = None
        self.ort_session = None
        self.ort_input_name = None
        self.model_input_h = None
        self.model_input_w = None
        self.active_backend = "disabled" if not self.enabled else "heuristic"
        self.weights = {
            "bias": -0.15,
            "laplacian": 0.95,
            "texture": 0.85,
            "colorfulness": 0.35,
            "specular_penalty": 1.25,
            "moire_penalty": 1.45,
            "screen_penalty": 1.55,
            "banding_penalty": 0.85,
        }

        self._load_model_artifacts()

    def _load_model_artifacts(self):
        model_path = Path(self.config.anti_spoof_model_path)
        onnx_loaded = False
        if model_path.exists() and model_path.suffix.lower() == ".onnx":
            if self.use_onnxruntime and ort is not None:
                try:
                    providers = ["CPUExecutionProvider"]
                    session_options = ort.SessionOptions()
                    session_options.log_severity_level = 3
                    session_options.log_verbosity_level = 0
                    self.ort_session = ort.InferenceSession(
                        str(model_path),
                        sess_options=session_options,
                        providers=providers,
                    )
                    input_meta = self.ort_session.get_inputs()[0]
                    self.ort_input_name = input_meta.name
                    shape = input_meta.shape
                    if len(shape) >= 4:
                        h = shape[2]
                        w = shape[3]
                        if isinstance(h, int) and isinstance(w, int):
                            self.model_input_h = h
                            self.model_input_w = w
                    onnx_loaded = True
                    self.active_backend = "onnxruntime"
                    logger.info("Anti-spoof ONNX model loaded via onnxruntime: %s", model_path)
                except Exception as exc:
                    logger.warning("Failed to load anti-spoof ONNX model via onnxruntime: %s", exc)

            if not onnx_loaded:
                try:
                    self.onnx_net = cv2.dnn.readNetFromONNX(str(model_path))
                    onnx_loaded = True
                    self.active_backend = "opencv_dnn"
                    logger.info("Anti-spoof ONNX model loaded via OpenCV DNN: %s", model_path)
                except Exception as exc:
                    logger.warning("Failed to load anti-spoof ONNX model via OpenCV DNN: %s", exc)

        if self.backend in {"onnx", "minifasnet"} and not onnx_loaded:
            logger.warning(
                "Configured anti-spoof backend '%s' but ONNX model not loaded. Falling back to heuristic.",
                self.backend,
            )

        coeff_path = model_path.with_suffix(".json")
        if coeff_path.exists():
            try:
                with open(coeff_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.weights.update(data)
                logger.info("Anti-spoof coefficient model loaded: %s", coeff_path)
            except Exception as exc:
                logger.warning("Failed to load anti-spoof coefficient model: %s", exc)

    def get_status(self):
        if not self.enabled:
            return {
                "enabled": False,
                "backend": "disabled",
                "configured_backend": self.backend,
                "model_path": self.model_path,
                "input_size": self.input_size,
                "live_index": self.live_index,
                "threshold": self.threshold,
            }

        return {
            "enabled": True,
            "backend": self.active_backend,
            "configured_backend": self.backend,
            "model_path": self.model_path,
            "input_size": self.model_input_h if self.model_input_h else self.input_size,
            "live_index": self.live_index,
            "threshold": self.threshold,
        }

    def check(self, frame, bbox):
        if not self.enabled:
            return True, 1.0

        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return False, 0.0

        face = frame[y : y + h, x : x + w]
        if face.size == 0:
            return False, 0.0

        use_model = self.backend in {"auto", "onnx", "minifasnet"} and (
            self.ort_session is not None or self.onnx_net is not None
        )
        if use_model:
            score = self._check_with_onnx(face)
        else:
            score = self._check_with_coefficients(frame, face, bbox)

        return score >= self.threshold, float(score)

    def _check_with_onnx(self, face):
        inp = self._prepare_onnx_input(face)
        out = None

        if self.ort_session is not None and self.ort_input_name is not None:
            try:
                pred = self.ort_session.run(None, {self.ort_input_name: inp})
                out = np.array(pred[0]).squeeze()
            except Exception as exc:
                logger.warning("onnxruntime anti-spoof inference failed: %s", exc)

        if out is None and self.onnx_net is not None:
            try:
                self.onnx_net.setInput(inp)
                out = self.onnx_net.forward().squeeze()
            except Exception as exc:
                logger.warning("OpenCV DNN anti-spoof inference failed: %s", exc)

        if out is None:
            return 0.0

        out = np.array(out).flatten()
        if out.size == 1:
            return float(1.0 / (1.0 + np.exp(-out[0])))
        if out.size >= 2:
            exp = np.exp(out - np.max(out))
            probs = exp / (np.sum(exp) + 1e-9)
            idx = int(np.clip(self.live_index, 0, len(probs) - 1))
            return float(probs[idx])
        return 0.0

    def _prepare_onnx_input(self, face):
        if self.model_input_h and self.model_input_w:
            target_h = max(32, int(self.model_input_h))
            target_w = max(32, int(self.model_input_w))
        else:
            target_h = target_w = max(32, int(self.input_size))

        resized = cv2.resize(face, (target_w, target_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        rgb = (rgb - 127.5) / 128.0
        chw = np.transpose(rgb, (2, 0, 1))
        return np.expand_dims(chw, axis=0).astype(np.float32)

    def _check_with_coefficients(self, frame, face, bbox):
        face = cv2.resize(face, (128, 128), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        lap_norm = np.clip(lap_var / 320.0, 0.0, 1.0)

        texture = self._lbp_texture_score(gray)
        colorfulness = self._colorfulness(face)
        color_norm = np.clip(colorfulness / 60.0, 0.0, 1.0)

        specular = self._specular_ratio(face)
        moire = self._moire_score(gray)
        screen = self._screen_artifact_score(frame, bbox)
        banding = self._banding_score(gray)

        linear = (
            self.weights["bias"]
            + self.weights["laplacian"] * lap_norm
            + self.weights["texture"] * texture
            + self.weights["colorfulness"] * color_norm
            - self.weights["specular_penalty"] * specular
            - self.weights["moire_penalty"] * moire
            - self.weights["screen_penalty"] * screen
            - self.weights["banding_penalty"] * banding
        )

        return float(1.0 / (1.0 + np.exp(-linear)))

    def _lbp_texture_score(self, gray):
        h, w = gray.shape
        center = gray[1 : h - 1, 1 : w - 1]
        neighbors = [
            gray[0 : h - 2, 0 : w - 2],
            gray[0 : h - 2, 1 : w - 1],
            gray[0 : h - 2, 2:w],
            gray[1 : h - 1, 2:w],
            gray[2:h, 2:w],
            gray[2:h, 1 : w - 1],
            gray[2:h, 0 : w - 2],
            gray[1 : h - 1, 0 : w - 2],
        ]

        lbp = np.zeros_like(center, dtype=np.uint8)
        for idx, n in enumerate(neighbors):
            lbp |= ((n >= center).astype(np.uint8) << idx)

        hist = cv2.calcHist([lbp], [0], None, [256], [0, 256]).flatten()
        hist_sum = np.sum(hist) + 1e-6
        hist /= hist_sum
        entropy = -np.sum(hist * np.log2(hist + 1e-9))
        return float(np.clip(entropy / 8.0, 0.0, 1.0))

    def _colorfulness(self, image):
        b, g, r = cv2.split(image.astype(np.float32))
        rg = np.abs(r - g)
        yb = np.abs(0.5 * (r + g) - b)
        std_rg, std_yb = np.std(rg), np.std(yb)
        mean_rg, mean_yb = np.mean(rg), np.mean(yb)
        return float(np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2))

    def _specular_ratio(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2]
        s = hsv[:, :, 1]
        bright = (v > 240) & (s < 40)
        return float(np.mean(bright))

    def _moire_score(self, gray):
        f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
        mag = np.log1p(np.abs(f))
        h, w = mag.shape
        cy, cx = h // 2, w // 2

        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        high_band = (r > min(h, w) * 0.25) & (r < min(h, w) * 0.45)

        band_vals = mag[high_band]
        if band_vals.size == 0:
            return 0.0

        p95 = np.percentile(band_vals, 95)
        p50 = np.percentile(band_vals, 50)
        score = (p95 - p50) / (p95 + 1e-6)
        return float(np.clip(score, 0.0, 1.0))

    def _screen_artifact_score(self, frame, bbox):
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return 0.0

        fh, fw = frame.shape[:2]
        pad_w = int(w * 0.45)
        pad_h = int(h * 0.45)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(fw, x + w + pad_w)
        y2 = min(fh, y + h + pad_h)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return 0.0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 180)
        edge_ratio = float(np.mean(edges > 0))

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=40,
            minLineLength=max(20, int(min(roi.shape[:2]) * 0.35)),
            maxLineGap=8,
        )
        line_score = 0.0
        if lines is not None and len(lines) > 0:
            angles = []
            for line in lines[:, 0, :]:
                x1l, y1l, x2l, y2l = line
                angle = abs(np.degrees(np.arctan2(y2l - y1l, x2l - x1l)))
                angle = min(angle, abs(180 - angle))
                angles.append(angle)
            if angles:
                hv_lines = sum(1 for a in angles if a <= 12 or abs(a - 90) <= 12)
                line_score = hv_lines / len(angles)

        score = 0.45 * np.clip(edge_ratio / 0.20, 0.0, 1.0) + 0.55 * np.clip(line_score, 0.0, 1.0)
        return float(np.clip(score, 0.0, 1.0))

    def _banding_score(self, gray):
        if gray.size == 0:
            return 0.0
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        mean_grad = float(np.mean(mag) + 1e-6)
        q = np.round(gray / 8.0) * 8.0
        quant_noise = float(np.mean(np.abs(gray.astype(np.float32) - q.astype(np.float32))))
        score = (8.0 - quant_noise) / 8.0
        score = score * np.clip(1.0 / (mean_grad / 25.0), 0.0, 1.0)
        return float(np.clip(score, 0.0, 1.0))
