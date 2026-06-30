from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.logging_config import configure_logging
from app.core.startup_checks import (
    assert_db_reachable,
    assert_activity_logs_schema_is_ready,
    assert_department_schema_is_ready,
    assert_required_model_files_exist,
    assert_secret_key_is_strong,
    assert_weekday_storage_schema_is_ready,
)
from app.db.models import Role, SystemSetting
from app.db.session import SessionLocal
from app.routers import (
    academic_structure,
    activity,
    attendance,
    auth,
    classes,
    courses,
    departments,
    faculties,
    reports,
    schedules,
    student_portal,
    sessions,
    students,
    teachers,
    # training router included separately below
)
from app.services.face_service import face_service
from app.services.schedule_service import schedule_service
from app.routers import training


configure_logging()
logger = logging.getLogger(__name__)


_DEFAULT_SETTINGS: dict[str, str] = {
    # General
    "general.system_name": "Heegan",
    "general.org_name": "Heegan Educational Institution",
    "general.timezone": "Africa/Mogadishu",
    "general.language": "en",
    # Notifications
    "notifications.email": "true",
    "notifications.sms": "false",
    "notifications.system": "true",
    "notifications.attendance": "true",
    "notifications.frequency": "realtime",
    # Security
    "security.session_timeout": "30m",
    # Preferences
    "preferences.default_view": "admin",
    "preferences.date_format": "ddmmyyyy",
    "preferences.time_format": "12h",
    # HR
    "hr.require_approval": "true",
    "hr.notify_on_leave": "true",
}


def seed_settings() -> None:
    db = SessionLocal()
    try:
        existing_keys = {key for (key,) in db.query(SystemSetting.key).all()}
        for key, value in _DEFAULT_SETTINGS.items():
            if key not in existing_keys:
                db.add(SystemSetting(key=key, value=value))
        db.commit()
    finally:
        db.close()


def migrate_timezone_setting() -> None:
    """Convert legacy gmt* placeholder values to proper IANA timezone strings."""
    _legacy = {
        "gmt0": "UTC",
        "gmt3": "Africa/Mogadishu",
        "gmt4": "Asia/Dubai",
    }
    db = SessionLocal()
    try:
        row = db.query(SystemSetting).filter(SystemSetting.key == "general.timezone").first()
        if row and row.value in _legacy:
            row.value = _legacy[row.value]
            db.commit()
            logger.info("Migrated general.timezone from legacy value to %s", row.value)
    finally:
        db.close()


def load_timezone_from_db() -> None:
    """Read the stored timezone from the DB and apply it as the runtime override."""
    from app.utils.datetime_utils import set_runtime_timezone
    db = SessionLocal()
    try:
        row = db.query(SystemSetting).filter(SystemSetting.key == "general.timezone").first()
        if row and row.value:
            set_runtime_timezone(row.value)
            logger.info("Loaded timezone from DB: %s", row.value)
    finally:
        db.close()


def seed_roles() -> None:
    db = SessionLocal()
    try:
        role_names = [
            "SUPER_ADMIN",
            "ACADEMIA",
            "FACULTY",
            "HR",
            "ADMISSIONS",
            "TEACHER",
            "STUDENT",  # students can log in and view their own attendance and schedule (read-only)
        ]
        existing = {name for (name,) in db.query(Role.name).all()}
        for name in role_names:
            if name not in existing:
                db.add(Role(name=name))
        db.commit()
    finally:
        db.close()


@asynccontextmanager
async def lifespan(_: FastAPI):
    assert_secret_key_is_strong()
    assert_db_reachable()
    assert_weekday_storage_schema_is_ready()
    assert_department_schema_is_ready()
    assert_activity_logs_schema_is_ready()
    assert_required_model_files_exist()

    seed_roles()
    seed_settings()
    migrate_timezone_setting()
    load_timezone_from_db()

    # Load AI models exactly once during startup.
    # load_models() may fail gracefully when no embeddings exist yet (first-time
    # setup). Either way, pre-warm the detect-only recognizer so SCRFD + FaceNet
    # stay resident in memory. Without this, the first training run has to
    # re-initialise both models from scratch, which hangs on Windows while ONNX
    # Runtime serialises concurrent model loads.
    face_service.load_models()
    face_service._ensure_detect_recognizer()  # keeps SCRFD + FaceNet in memory
    logger.info("Face models loaded")

    await schedule_service.start()
    logger.info("Startup checks passed and scheduler started")
    try:
        yield
    finally:
        await schedule_service.stop()
        logger.info("Application shutdown complete")


app = FastAPI(title="Course Attendance System API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def handle_cancelled_requests(request: Request, call_next):
    """
    Catch asyncio.CancelledError (raised when a client disconnects mid-request).
    Without this, CancelledError escapes FastAPI's Exception handler (which only
    catches Exception, not BaseException), bypasses CORSMiddleware, and the browser
    sees ERR_FAILED with no CORS headers — reported as a spurious CORS policy error.
    """
    try:
        return await call_next(request)
    except asyncio.CancelledError:
        logger.debug("Request cancelled by client: %s %s", request.method, request.url.path)
        return Response(status_code=499, content=b"")


app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_allow_origins),
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=list(settings.cors_allow_methods),
    allow_headers=list(settings.cors_allow_headers),
)

app.include_router(auth.router)
app.include_router(faculties.router)
app.include_router(departments.router)
app.include_router(classes.router)
app.include_router(students.router)
app.include_router(teachers.router)
app.include_router(courses.router)
app.include_router(academic_structure.router)
app.include_router(schedules.router)
app.include_router(sessions.router)
app.include_router(attendance.router)
app.include_router(student_portal.router)
app.include_router(reports.router)
app.include_router(activity.router)
from app.routers import users
from app.routers import app_settings

# Mount static files for uploaded assets
static_dir = Path(settings.static_dir or "static")
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

app.include_router(users.router)
app.include_router(training.router)
app.include_router(app_settings.router)


def _error_payload(code: str, message: str, path: str, details=None) -> dict:
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details,
            "path": path,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    message = exc.detail if isinstance(exc.detail, str) else "HTTP error"
    details = None if isinstance(exc.detail, str) else exc.detail
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_payload(
            code=f"HTTP_{exc.status_code}",
            message=message,
            details=details,
            path=str(request.url.path),
        ),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=_error_payload(
            code="VALIDATION_ERROR",
            message="Request validation failed",
            details=exc.errors(),
            path=str(request.url.path),
        ),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled server error", exc_info=exc)
    return JSONResponse(
        status_code=500,
        content=_error_payload(
            code="INTERNAL_SERVER_ERROR",
            message="Internal server error",
            details=None,
            path=str(request.url.path),
        ),
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/health/live")
def health_live():
    return {"status": "alive"}


@app.get("/health/ready")
def health_ready():
    checks: dict[str, bool] = {
        "db": False,
        "models": False,
        "scheduler": False,
    }
    try:
        assert_db_reachable()
        checks["db"] = True
        assert_required_model_files_exist()
        checks["models"] = True
        scheduler_status = schedule_service.readiness_status()
        checks["scheduler"] = bool(scheduler_status["healthy"])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail={"checks": checks, "reason": str(exc)}) from exc

    if not all(checks.values()):
        detail = {"checks": checks, "reason": "Service not fully ready"}
        if "scheduler_status" in locals():
            detail["scheduler"] = scheduler_status
        raise HTTPException(status_code=503, detail=detail)
    return {"status": "ready", "checks": checks}


@app.get("/health/scheduler")
def health_scheduler():
    return schedule_service.scheduler_report()
