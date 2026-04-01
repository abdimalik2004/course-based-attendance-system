from __future__ import annotations

import base64

import cv2
import numpy as np


def decode_base64_image(image_b64: str) -> np.ndarray:
    payload = image_b64
    if "," in image_b64:
        payload = image_b64.split(",", 1)[1]
    image_bytes = base64.b64decode(payload)
    nparr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Invalid image payload")
    return frame
