from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logging_config import configure_logging
from app.core.startup_checks import (
    assert_db_reachable,
    assert_department_schema_is_ready,
    assert_required_model_files_exist,
    assert_secret_key_is_strong,
    assert_weekday_storage_schema_is_ready,
)
from app.db.models import Role
from app.db.session import SessionLocal
from app.routers import (
    academic_structure,
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
)
from app.services.face_service import face_service
from app.services.schedule_service import schedule_service


configure_logging()
logger = logging.getLogger(__name__)


def seed_roles() -> None:
    db = SessionLocal()
    try:
        role_names = ["SUPER_ADMIN", "ACADEMIA", "FACULTY", "FACULTY_ADMIN", "HR", "ADMISSIONS", "TEACHER"]
        existing = {r.name for r in db.query(Role).all()}
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
    assert_required_model_files_exist()

    seed_roles()

    # Load AI models exactly once during startup.
    face_service.load_models()
    logger.info("Face models loaded")

    await schedule_service.start()
    logger.info("Startup checks passed and scheduler started")
    try:
        yield
    finally:
        await schedule_service.stop()
        logger.info("Application shutdown complete")


app = FastAPI(title="Course Attendance System API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_allow_origins),
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
