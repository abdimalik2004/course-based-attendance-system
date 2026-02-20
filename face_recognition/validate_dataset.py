from pathlib import Path

import cv2

from face_recognition.detector import FaceDetector
from utils.logging import get_logger


logger = get_logger(__name__)


def validate_dataset(config):
    detector = FaceDetector(config.haar_cascade_path)
    dataset_dir = Path(config.dataset_dir)
    if not dataset_dir.exists():
        raise RuntimeError(f"Dataset folder not found: {dataset_dir}")

    total_images = 0
    face_found = 0
    missing = []

    for student_dir in sorted(dataset_dir.iterdir()):
        if not student_dir.is_dir():
            continue
        for img_path in list(student_dir.glob("*.jpg")) + list(student_dir.glob("*.jpeg")) + list(student_dir.glob("*.png")):
            total_images += 1
            image = cv2.imread(str(img_path))
            if image is None:
                missing.append(str(img_path))
                continue
            _, faces = detector.detect(image)
            if len(faces) == 0:
                missing.append(str(img_path))
                continue
            face_found += 1

    logger.info("Dataset validation: total=%d with_faces=%d without_faces=%d", total_images, face_found, total_images - face_found)
    if missing:
        logger.warning("Images without detectable faces or unreadable:")
        for path in missing:
            logger.warning("  %s", path)

    return {
        "total": total_images,
        "with_faces": face_found,
        "without_faces": total_images - face_found,
        "missing": missing,
    }
