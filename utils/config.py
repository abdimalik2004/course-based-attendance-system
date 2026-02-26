from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()


def _env_int(key, default):
    value = os.getenv(key)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(key, default):
    value = os.getenv(key)
    if value is None or value == "":
        return default
    return float(value)


def _env_bool(key, default):
    value = os.getenv(key)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class AppConfig:
    base_dir: Path
    dataset_dir: Path
    model_path: Path
    label_map_path: Path
    db_type: str
    db_path: Path
    db_init_schema: bool
    auto_schedule: bool
    embedding_path: Path
    recognizer_type: str
    anti_spoof_enabled: bool
    anti_spoof_threshold: float
    anti_spoof_model_path: Path
    anti_spoof_backend: str
    anti_spoof_input_size: int
    anti_spoof_live_index: int
    anti_spoof_use_onnxruntime: bool
    anti_spoof_required_frames: int
    anti_spoof_margin: float
    anti_spoof_min_pass_ratio: float
    occlusion_check_enabled: bool
    occlusion_min_eyes_visible: int
    occlusion_min_eye_variance: float
    db_host: str
    db_port: int
    db_name: str
    db_user: str
    db_password: str
    haar_cascade_path: str
    confidence_threshold: float
    embedding_min_similarity: float
    min_face_size: int
    required_matches: int
    camera_width: int
    camera_height: int
    camera_index: int
    preview_width: int
    preview_height: int
    max_runtime_seconds: int
    capture_pad_top: float
    capture_pad_bottom: float
    capture_pad_left: float
    capture_pad_right: float
    log_dir: Path
    log_file: Path


def load_config(base_dir=None):
    base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parents[1]
    dataset_dir = Path(os.getenv("ATTENDANCE_DATASET_DIR", base_dir / "dataset"))
    model_path = Path(os.getenv("ATTENDANCE_MODEL_PATH", base_dir / "models" / "lbph_trainer.yml"))
    label_map_path = Path(os.getenv("ATTENDANCE_LABEL_MAP_PATH", base_dir / "models" / "label_map.json"))
    embedding_path = Path(os.getenv("ATTENDANCE_EMBEDDING_PATH", base_dir / "models" / "face_embeddings.npz"))
    db_type = os.getenv("ATTENDANCE_DB_TYPE", "sqlite")
    db_path = Path(os.getenv("ATTENDANCE_DB_PATH", base_dir / "database" / "attendance.db"))

    return AppConfig(
        base_dir=base_dir,
        dataset_dir=dataset_dir,
        model_path=model_path,
        label_map_path=label_map_path,
        embedding_path=embedding_path,
        db_type=db_type,
        db_path=db_path,
        db_init_schema=_env_bool("ATTENDANCE_DB_INIT_SCHEMA", db_type.lower() == "sqlite"),
        auto_schedule=_env_bool("ATTENDANCE_AUTO_SCHEDULE", False),
        recognizer_type=os.getenv("ATTENDANCE_RECOGNIZER", "lbph").lower(),
        anti_spoof_enabled=_env_bool("ATTENDANCE_ANTI_SPOOF_ENABLED", True),
        anti_spoof_threshold=_env_float("ATTENDANCE_ANTI_SPOOF_THRESHOLD", 0.62),
        anti_spoof_model_path=Path(
            os.getenv("ATTENDANCE_ANTI_SPOOF_MODEL_PATH", base_dir / "models" / "anti_spoof.onnx")
        ),
        anti_spoof_backend=os.getenv("ATTENDANCE_ANTI_SPOOF_BACKEND", "auto").lower(),
        anti_spoof_input_size=_env_int("ATTENDANCE_ANTI_SPOOF_INPUT_SIZE", 80),
        anti_spoof_live_index=_env_int("ATTENDANCE_ANTI_SPOOF_LIVE_INDEX", 1),
        anti_spoof_use_onnxruntime=_env_bool("ATTENDANCE_ANTI_SPOOF_USE_ONNXRUNTIME", True),
        anti_spoof_required_frames=_env_int("ATTENDANCE_ANTI_SPOOF_REQUIRED_FRAMES", 3),
        anti_spoof_margin=_env_float("ATTENDANCE_ANTI_SPOOF_MARGIN", 0.0),
        anti_spoof_min_pass_ratio=_env_float("ATTENDANCE_ANTI_SPOOF_MIN_PASS_RATIO", 0.67),
        occlusion_check_enabled=_env_bool("ATTENDANCE_OCCLUSION_CHECK_ENABLED", True),
        occlusion_min_eyes_visible=_env_int("ATTENDANCE_OCCLUSION_MIN_EYES_VISIBLE", 2),
        occlusion_min_eye_variance=_env_float("ATTENDANCE_OCCLUSION_MIN_EYE_VARIANCE", 120.0),
        db_host=os.getenv("ATTENDANCE_DB_HOST", "localhost"),
        db_port=_env_int("ATTENDANCE_DB_PORT", 3306),
        db_name=os.getenv("ATTENDANCE_DB_NAME", "attendance"),
        db_user=os.getenv("ATTENDANCE_DB_USER", "root"),
        db_password=os.getenv("ATTENDANCE_DB_PASSWORD", ""),
        haar_cascade_path=os.getenv("ATTENDANCE_HAAR_CASCADE_PATH", ""),
        confidence_threshold=_env_float("ATTENDANCE_CONFIDENCE_THRESHOLD", 60.0),
        embedding_min_similarity=_env_float("ATTENDANCE_EMBEDDING_MIN_SIMILARITY", 0.6),
        min_face_size=_env_int("ATTENDANCE_MIN_FACE_SIZE", 60),
        required_matches=_env_int("ATTENDANCE_REQUIRED_MATCHES", 5),
        camera_width=_env_int("ATTENDANCE_CAMERA_WIDTH", 1920),
        camera_height=_env_int("ATTENDANCE_CAMERA_HEIGHT", 1080),
        camera_index=_env_int("ATTENDANCE_CAMERA_INDEX", 0),
        preview_width=_env_int("ATTENDANCE_PREVIEW_WIDTH", 960),
        preview_height=_env_int("ATTENDANCE_PREVIEW_HEIGHT", 540),
        max_runtime_seconds=_env_int("ATTENDANCE_MAX_RUNTIME_SECONDS", 0),
        capture_pad_top=_env_float("ATTENDANCE_CAPTURE_PAD_TOP", 0.3),
        capture_pad_bottom=_env_float("ATTENDANCE_CAPTURE_PAD_BOTTOM", 0.2),
        capture_pad_left=_env_float("ATTENDANCE_CAPTURE_PAD_LEFT", 0.15),
        capture_pad_right=_env_float("ATTENDANCE_CAPTURE_PAD_RIGHT", 0.15),
        log_dir=Path(os.getenv("ATTENDANCE_LOG_DIR", base_dir / "logs")),
        log_file=Path(os.getenv("ATTENDANCE_LOG_FILE", base_dir / "logs" / "attendance.log")),
    )
