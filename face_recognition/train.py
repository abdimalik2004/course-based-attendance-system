import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN

from face_recognition.detector import FaceDetector
from utils.logging import get_logger


logger = get_logger(__name__)


def _largest_face(faces):
    if len(faces) == 0:
        return None
    return max(faces, key=lambda f: f[2] * f[3])


def train_from_dataset(config):
    detector = FaceDetector(config.haar_cascade_path)
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
    mtcnn = MTCNN(
        image_size=160,
        margin=0,
        min_face_size=max(20, int(config.min_face_size)),
        device=device,
    )
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
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            boxes, _ = mtcnn.detect(pil_img)
            if boxes is None or len(boxes) == 0:
                continue
            box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            box = np.expand_dims(box, axis=0)
            faces = mtcnn.extract(pil_img, box, save_path=None)
            if faces is None or len(faces) == 0:
                continue
            with torch.no_grad():
                if isinstance(faces, list):
                    faces = faces[0]
                if faces.dim() == 3:
                    faces = faces.unsqueeze(0)
                emb = model(faces.to(device)).cpu().numpy()[0]
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
