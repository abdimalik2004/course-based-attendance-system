from __future__ import annotations

from pathlib import Path

from sqlalchemy import inspect
from sqlalchemy import text

from app.core.config import settings
from app.db.session import engine


def assert_db_reachable() -> None:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))


def assert_required_model_files_exist() -> None:
    # Backend integrates with project-level face models under attendance_system/models.
    root = Path(__file__).resolve().parents[3]
    required_files = [
        root / "models" / "face_embeddings.npz",
        root / "models" / "anti_spoof_minifasnet.onnx",
    ]

    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        raise RuntimeError(f"Required model files missing: {missing}")


def assert_secret_key_is_strong() -> None:
    bad_values = {
        "",
        "change-me-in-production",
        "replace_with_secure_secret",
        "replace-with-strong-production-secret",
        "dev-only-secret",
    }
    key = settings.secret_key.strip()
    if key in bad_values or len(key) < 32:
        raise RuntimeError(
            "SECRET_KEY is weak/default. Configure a strong key (>=32 chars) in your environment profile."
        )


def assert_weekday_storage_schema_is_ready() -> None:
    # Prevent silent MySQL coercion of weekday strings to 0 when old INT schema is still active.
    with engine.connect() as conn:
        weekday_col = next(
            (c for c in inspect(conn).get_columns("course_schedules") if c.get("name") == "weekday"),
            None,
        )

    if weekday_col is None:
        raise RuntimeError("course_schedules.weekday column not found")

    col_type_name = str(weekday_col.get("type", "")).upper()
    if "CHAR" in col_type_name or "TEXT" in col_type_name or "STRING" in col_type_name:
        return

    raise RuntimeError(
        "Outdated schema detected: course_schedules.weekday must be VARCHAR/TEXT day-codes. "
        "Run: python -m alembic upgrade head"
    )


def assert_department_schema_is_ready() -> None:
    with engine.connect() as conn:
        inspector = inspect(conn)
        table_names = set(inspector.get_table_names())

        if "departments" not in table_names:
            raise RuntimeError("Outdated schema detected: departments table not found. Run: python -m alembic upgrade head")

        required_columns = {
            "class_batches": "department_id",
            "students": "department_id",
            "teachers": "department_id",
        }
        for table_name, column_name in required_columns.items():
            columns = {column.get("name") for column in inspector.get_columns(table_name)}
            if column_name not in columns:
                raise RuntimeError(
                    f"Outdated schema detected: {table_name}.{column_name} column not found. "
                    "Run: python -m alembic upgrade head"
                )


