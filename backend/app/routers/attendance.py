from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.rate_limit import rate_limit_dependency
from app.core.security import require_roles
from app.db.faculty_scope import enforce_faculty_scope, get_optional_faculty_scope_context
from app.db.models import AttendanceSession
from app.db.role_scoped import get_role_scoped_db
from app.schemas.attendance import AttendanceFrameRequest
from app.services.attendance_service import attendance_service


router = APIRouter(prefix="/attendance", tags=["attendance"])


def _session_faculty_id(db: Session, session_id: int) -> int:
    session = db.query(AttendanceSession).filter(AttendanceSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    course = session.course
    if course is None:
        raise HTTPException(status_code=404, detail="Course context not found for session")
    return course.faculty_id


@router.post(
    "/frame",
    dependencies=[
        Depends(require_roles("TEACHER", "FACULTY")),
        Depends(rate_limit_dependency(settings.frame_rate_limit_requests, settings.frame_rate_limit_window_seconds)),
    ],
    responses={
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
        429: {"description": "Too many requests"},
        422: {"description": "Invalid frame payload"},
    },
)
def process_attendance_frame(
    payload: AttendanceFrameRequest,
    db: Session = Depends(get_role_scoped_db),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    if faculty_scope is not None:
        enforce_faculty_scope(_session_faculty_id(db, payload.session_id), faculty_scope)
    return attendance_service.process_frame(db=db, session_id=payload.session_id, image_b64=payload.image)
