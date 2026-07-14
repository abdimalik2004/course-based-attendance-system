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
        inspector = inspect(conn)
        table_names = set(inspector.get_table_names())
        weekday_col = next((c for c in inspector.get_columns("course_schedules") if c.get("name") == "weekday"), None)

    if weekday_col is None:
        raise RuntimeError("course_schedules.weekday column not found")

    if "course_schedule_weekdays" not in table_names:
        raise RuntimeError("course_schedule_weekdays table not found")

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


def assert_activity_logs_schema_is_ready() -> None:
    with engine.connect() as conn:
        inspector = inspect(conn)
        if "activity_logs" not in inspector.get_table_names():
            raise RuntimeError(
                "Outdated schema detected: activity_logs table not found. "
                "Run: python -m alembic upgrade head"
            )
        columns = {column.get("name") for column in inspector.get_columns("activity_logs")}
        required_columns = {"id", "user_id", "username", "action", "status", "created_at"}
        missing_columns = required_columns - columns
        if missing_columns:
            raise RuntimeError(
                f"Outdated schema detected: activity_logs missing columns {sorted(missing_columns)}. "
                "Run: python -m alembic upgrade head"
            )


def migrate_teacher_status_onleave() -> None:
    """Rename the legacy 'Onleave' enum value to 'On Leave' in the teachers table.

    The Python enum was corrected from TeacherStatus.ONLEAVE = "Onleave" to
    "On Leave" (two words, proper English). This one-time migration updates
    both the column definition (MySQL) and any existing row data to match.
    Safe to run on every startup — exits immediately if no old rows remain.
    """
    import logging
    _logger = logging.getLogger(__name__)

    with engine.begin() as conn:
        # Fast path: if no rows use the old value, nothing to do.
        result = conn.execute(text("SELECT COUNT(*) FROM teachers WHERE status = 'Onleave'"))
        if result.scalar() == 0:
            return

        _logger.info("Migrating teacher status 'Onleave' → 'On Leave'…")
        dialect = engine.dialect.name

        if dialect in {"mysql", "mariadb"}:
            # Step 1: widen the enum to include both old and new value so that
            #         existing rows remain valid during the data update.
            conn.execute(text(
                "ALTER TABLE teachers MODIFY COLUMN status "
                "ENUM('Active', 'Onleave', 'On Leave', 'Inactive') NOT NULL DEFAULT 'Active'"
            ))
            # Step 2: migrate data
            conn.execute(text("UPDATE teachers SET status = 'On Leave' WHERE status = 'Onleave'"))
            # Step 3: remove the old value from the enum definition
            conn.execute(text(
                "ALTER TABLE teachers MODIFY COLUMN status "
                "ENUM('Active', 'On Leave', 'Inactive') NOT NULL DEFAULT 'Active'"
            ))
        else:
            # SQLite/PostgreSQL store enums as text; a plain UPDATE is enough.
            conn.execute(text("UPDATE teachers SET status = 'On Leave' WHERE status = 'Onleave'"))

        _logger.info("Teacher status migration complete.")


def migrate_teacher_contact_fields() -> None:
    """Add phone, email, hire_date columns to the teachers table if they don't exist.

    These columns were introduced after the initial table creation. The migration
    is idempotent — if the columns already exist it returns immediately.
    """
    import logging
    _logger = logging.getLogger(__name__)

    with engine.connect() as conn:
        from sqlalchemy import inspect as sa_inspect
        inspector = sa_inspect(conn)
        existing_cols = {c["name"] for c in inspector.get_columns("teachers")}

    needed = {"phone", "email", "hire_date"} - existing_cols
    if not needed:
        return

    _logger.info("Adding teacher contact columns: %s", sorted(needed))
    with engine.begin() as conn:
        if "phone" in needed:
            conn.execute(text("ALTER TABLE teachers ADD COLUMN phone VARCHAR(30) NULL"))
        if "email" in needed:
            conn.execute(text("ALTER TABLE teachers ADD COLUMN email VARCHAR(180) NULL"))
        if "hire_date" in needed:
            conn.execute(text("ALTER TABLE teachers ADD COLUMN hire_date DATE NULL"))
    _logger.info("Teacher contact columns added successfully.")


