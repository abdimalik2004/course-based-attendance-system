import json
from pathlib import Path

import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

from face_recognition.detector import FaceDetector
from utils.logging import get_logger


logger = get_logger(__name__)


def _largest_face(faces):
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def train_from_dataset(config):
    detector = FaceDetector(config)
    label_map = {"labels": {}, "students": {}}
    faces_data = []
    labels = []

    dataset_dir = Path(config.dataset_dir)
    if not dataset_dir.exists():
        raise RuntimeError(f"Dataset folder not found: {dataset_dir}")

    next_label = 0
    for student_dir in sorted(dataset_dir.iterdir()):
        if not student_dir.is_dir():
            continue
        student_id = student_dir.name
        if student_id not in label_map["students"]:
            label_map["students"][student_id] = next_label
            label_map["labels"][str(next_label)] = student_id
            next_label += 1

        for img_path in list(student_dir.glob("*.jpg")) + list(student_dir.glob("*.jpeg")) + list(student_dir.glob("*.png")):
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            gray, faces = detector.detect(image)
            face_box = _largest_face(faces)
            if face_box is None:
                continue
            x, y, w, h = face_box
            face_roi = gray[y : y + h, x : x + w]
            faces_data.append(face_roi)
            labels.append(label_map["students"][student_id])

    if not faces_data:
        raise RuntimeError("No face samples found for training")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # LBPH encodes local texture patterns into histograms for each face ROI.
    recognizer.train(faces_data, np.array(labels))

    Path(config.model_path).parent.mkdir(parents=True, exist_ok=True)
    recognizer.save(str(config.model_path))

    with open(config.label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    logger.info("Trained LBPH model with %d samples", len(faces_data))
    # TODO: add incremental update strategy for new students without full retraining.


def train_embeddings_from_dataset(config):
    device = torch.device("cpu")
    detector = FaceDetector(config)
    model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    dataset_dir = Path(config.dataset_dir)
    if not dataset_dir.exists():
        raise RuntimeError(f"Dataset folder not found: {dataset_dir}")

    embeddings = []
    student_ids = []

    for student_dir in sorted(dataset_dir.iterdir()):
        if not student_dir.is_dir():
            continue
        student_id = student_dir.name
        for img_path in list(student_dir.glob("*.jpg")) + list(student_dir.glob("*.jpeg")) + list(student_dir.glob("*.png")):
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            _, detections = detector.detect_with_keypoints(image)
            if detections is None or len(detections) == 0:
                continue

            det = max(detections, key=lambda d: d["bbox"][2] * d["bbox"][3])
            x, y, w, h = det["bbox"]
            quality_ok, _ = detector.passes_quality(image, (x, y, w, h))
            if not quality_ok:
                continue

            h_img, w_img = image.shape[:2]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w_img, x + w)
            y2 = min(h_img, y + h)
            if x2 <= x1 or y2 <= y1:
                continue

            face = image[y1:y2, x1:x2]
            if face.size == 0:
                continue

            resized = cv2.resize(face, (160, 160), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
            norm = (rgb - 127.5) / 128.0
            face_tensor = torch.from_numpy(np.transpose(norm, (2, 0, 1))).unsqueeze(0)

            with torch.no_grad():
                emb = model(face_tensor.to(device)).cpu().numpy()[0]
            emb = emb / (np.linalg.norm(emb) + 1e-12)
            embeddings.append(emb)
            student_ids.append(student_id)

    if not embeddings:
        raise RuntimeError("No face samples found for embedding training")

    Path(config.embedding_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(config.embedding_path),
        embeddings=np.array(embeddings),
        student_ids=np.array(student_ids, dtype=object),
    )

    logger.info("Trained FaceNet embeddings with %d samples", len(embeddings))
