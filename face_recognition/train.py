import json
from pathlib import Path

import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

from face_recognition.detector import FaceDetector
from utils.dataset_paths import iter_student_dataset_dirs, normalize_student_id
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
    for student_dir in iter_student_dataset_dirs(dataset_dir):
        student_id = normalize_student_id(student_dir.name)
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


def train_embeddings_from_dataset(config, *, model=None, detector=None):
    """
    FaceNet embedding training pipeline.

    Parameters
    ----------
    config  : AppConfig
    model   : optional pre-loaded InceptionResnetV1 instance.
              Pass the one already held by the recognition service to avoid a
              second expensive model-load (and to prevent Windows ONNX-Runtime
              load serialisation, which causes the training thread to hang).
    detector: optional pre-loaded FaceDetector instance (same reason).

    Pipeline:
      1. Auto-select CUDA when available (GPU gives 5-10× speed-up).
      2. Parallel image loading via ThreadPoolExecutor (IO-bound).
      3. SCRFD face detection per image — same crop geometry as live recognition,
         so training embeddings match what the recognizer sees at attendance time.
         Falls back to a centre-crop when SCRFD finds no face.
      4. Batched FaceNet inference.
      5. Cap images per student (default 20) to avoid runaway training times.

    Using SCRFD in training is the key accuracy improvement over the old
    centre-crop approach: previously, training saw a rough 75 %-of-frame crop
    while recognition saw a tight SCRFD bbox — the mismatch degraded accuracy.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    t0 = time.perf_counter()

    # ── 1. Device ────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training embeddings on device: %s", device)

    # ── 2. FaceNet model ─────────────────────────────────────────────────────
    # Prefer the pre-loaded model passed in by the caller (avoids a second
    # heavyweight model initialisation that can hang on Windows due to ONNX
    # Runtime serialising concurrent loads).
    _own_model = model is None
    if model is None:
        logger.info("No pre-loaded FaceNet model provided — loading fresh copy")
        model = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    else:
        logger.info("Re-using pre-loaded FaceNet model for training")
        model = model.eval()

    # ── 3. Tunable parameters ────────────────────────────────────────────────
    max_imgs_per_student = int(getattr(config, "train_max_images_per_student", 20))
    batch_size = int(getattr(config, "train_batch_size", 32))

    dataset_dir = Path(config.dataset_dir)
    if not dataset_dir.exists():
        raise RuntimeError(f"Dataset folder not found: {dataset_dir}")

    # ── 4. Gather image paths ────────────────────────────────────────────────
    all_pairs: list[tuple[Path, str]] = []
    for student_dir in iter_student_dataset_dirs(dataset_dir):
        student_id = normalize_student_id(student_dir.name)
        imgs = (
            list(student_dir.glob("*.jpg"))
            + list(student_dir.glob("*.jpeg"))
            + list(student_dir.glob("*.png"))
        )
        if len(imgs) > max_imgs_per_student:
            imgs = imgs[:max_imgs_per_student]
        for p in imgs:
            all_pairs.append((p, student_id))

    if not all_pairs:
        raise RuntimeError("No images found in dataset directory")

    # ── 5. Parallel image loading (IO-bound) ─────────────────────────────────
    def _load_image(pair: tuple[Path, str]) -> tuple:
        img_path, student_id = pair
        img = cv2.imread(str(img_path))
        return img, student_id

    n_workers = min(8, max(1, len(all_pairs)))
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        loaded = list(pool.map(_load_image, all_pairs))

    # ── 6. SCRFD detection + preprocessing (serial — SCRFD is not thread-safe) ─
    # IMPORTANT: only SCRFD-detected crops are used. The centre-crop fallback
    # was removed because it produces a large background-heavy patch (360×360 of
    # a 640×480 frame) that FaceNet encodes as "room background", making every
    # student's embedding look nearly identical. Live recognition uses a tight
    # SCRFD bbox — training must use the same geometry or embeddings won't match.
    # Images where SCRFD finds no face are SKIPPED; they are not usable.
    _own_detector = detector is None
    if detector is None:
        logger.info("No pre-loaded detector provided — loading fresh SCRFD")
        detector = FaceDetector(config)
    face_tensors: list[torch.Tensor] = []
    student_ids: list[str] = []
    scrfd_hits = 0
    skipped_no_face = 0

    for img, student_id in loaded:
        if img is None:
            continue

        img_h, img_w = img.shape[:2]
        face = None

        # SCRFD detection — identical crop to live recognition.
        try:
            _, detections = detector.detect_with_keypoints(img)
            if detections:
                largest = max(detections, key=lambda d: d["bbox"][2] * d["bbox"][3])
                x, y, w, h = largest["bbox"]
                # Add 10 % padding on each side so FaceNet sees a little context,
                # matching what live recognition gets after SCRFD crops.
                pad_x = int(w * 0.10)
                pad_y = int(h * 0.10)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(img_w, x + w + pad_x)
                y2 = min(img_h, y + h + pad_y)
                if x2 > x1 and y2 > y1:
                    face = img[y1:y2, x1:x2]
                    scrfd_hits += 1
        except Exception as exc:
            logger.debug("SCRFD detection failed for %s: %s", student_id, exc)

        if face is None or face.size == 0:
            # Image unusable — SCRFD found no face.
            # Skipping rather than using a background crop keeps training
            # consistent with inference (both always use SCRFD-detected regions).
            skipped_no_face += 1
            logger.debug(
                "Skipping image for %s — no face detected by SCRFD", student_id
            )
            continue

        resized = cv2.resize(face, (160, 160), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        norm = (rgb - 127.5) / 128.0
        tensor = torch.from_numpy(np.transpose(norm, (2, 0, 1)))
        face_tensors.append(tensor)
        student_ids.append(student_id)

    logger.info(
        "Face crop: %d SCRFD detections, %d skipped (no face detected)",
        scrfd_hits,
        skipped_no_face,
    )

    if not face_tensors:
        raise RuntimeError("No face samples found for embedding training")

    # ── 7. Batched FaceNet inference ─────────────────────────────────────────
    all_embeddings: list[np.ndarray] = []
    for i in range(0, len(face_tensors), batch_size):
        batch = torch.stack(face_tensors[i : i + batch_size]).to(device)
        with torch.no_grad():
            embs = model(batch).cpu().numpy()
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
        all_embeddings.extend(embs / norms)

    # ── 8. Persist ───────────────────────────────────────────────────────────
    Path(config.embedding_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(config.embedding_path),
        embeddings=np.array(all_embeddings),
        student_ids=np.array(student_ids, dtype=object),
    )

    elapsed = time.perf_counter() - t0
    logger.info(
        "Trained FaceNet embeddings: %d samples across %d students | device=%s | %.1fs",
        len(all_embeddings),
        len(set(student_ids)),
        device,
        elapsed,
    )
