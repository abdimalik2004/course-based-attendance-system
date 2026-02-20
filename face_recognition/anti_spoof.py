import json
from pathlib import Path

import cv2
import numpy as np

from utils.logging import get_logger


logger = get_logger(__name__)


class AntiSpoofModel:
    def __init__(self, config):
        self.config = config
        self.threshold = float(config.anti_spoof_threshold)
        self.enabled = bool(config.anti_spoof_enabled)
        self.onnx_net = None
        self.weights = {
            "bias": -0.15,
            "laplacian": 0.95,
            "texture": 0.85,
            "colorfulness": 0.35,
            "specular_penalty": 1.25,
            "moire_penalty": 1.45,
        }

        self._load_model_artifacts()

    def _load_model_artifacts(self):
        model_path = Path(self.config.anti_spoof_model_path)
        if model_path.exists() and model_path.suffix.lower() == ".onnx":
            try:
                self.onnx_net = cv2.dnn.readNetFromONNX(str(model_path))
                logger.info("Anti-spoof ONNX model loaded: %s", model_path)
                return
            except Exception as exc:
                logger.warning("Failed to load anti-spoof ONNX model: %s", exc)

        coeff_path = model_path.with_suffix(".json")
        if coeff_path.exists():
            try:
                with open(coeff_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.weights.update(data)
                logger.info("Anti-spoof coefficient model loaded: %s", coeff_path)
            except Exception as exc:
                logger.warning("Failed to load anti-spoof coefficient model: %s", exc)

    def check(self, frame, bbox):
        if not self.enabled:
            return True, 1.0

        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            return False, 0.0

        face = frame[y : y + h, x : x + w]
        if face.size == 0:
            return False, 0.0

        if self.onnx_net is not None:
            score = self._check_with_onnx(face)
        else:
            score = self._check_with_coefficients(face)

        return score >= self.threshold, float(score)

    def _check_with_onnx(self, face):
        resized = cv2.resize(face, (128, 128), interpolation=cv2.INTER_AREA)
        blob = cv2.dnn.blobFromImage(
            resized,
            scalefactor=1.0 / 255.0,
            size=(128, 128),
            mean=(0.0, 0.0, 0.0),
            swapRB=True,
            crop=False,
        )
        self.onnx_net.setInput(blob)
        out = self.onnx_net.forward().flatten()
        if out.size == 1:
            return float(1.0 / (1.0 + np.exp(-out[0])))
        if out.size >= 2:
            exp = np.exp(out - np.max(out))
            probs = exp / np.sum(exp)
            return float(probs[-1])
        return 0.0

    def _check_with_coefficients(self, face):
        face = cv2.resize(face, (128, 128), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        lap_norm = np.clip(lap_var / 320.0, 0.0, 1.0)

        texture = self._lbp_texture_score(gray)
        colorfulness = self._colorfulness(face)
        color_norm = np.clip(colorfulness / 60.0, 0.0, 1.0)

        specular = self._specular_ratio(face)
        moire = self._moire_score(gray)

        linear = (
            self.weights["bias"]
            + self.weights["laplacian"] * lap_norm
            + self.weights["texture"] * texture
            + self.weights["colorfulness"] * color_norm
            - self.weights["specular_penalty"] * specular
            - self.weights["moire_penalty"] * moire
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
