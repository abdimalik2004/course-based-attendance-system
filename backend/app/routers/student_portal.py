from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.security import get_current_user, require_roles
from app.db.models import User
from app.db.role_scoped import get_role_scoped_db
from app.schemas.student_portal import AttendanceCreate, AttendanceResponse, ScheduleCreate, ScheduleResponse
from app.services.student_portal_service import student_portal_service


class ExcuseRequestCreate(BaseModel):
    request_date: date
    course_id: int | None = None   # None = all courses that day
    reason: str | None = None


router = APIRouter(tags=["student-portal"])

_STUDENT_ALLOWED_ROLES = ("STUDENT", "SUPER_ADMIN", "ACADEMIA", "FACULTY", "ADMISSIONS", "TEACHER")


# ------------------------------------------------------------------
# /me/ endpoints — live data for the currently logged-in student
# ------------------------------------------------------------------

@router.get(
    "/student-portal/me/attendance",
    dependencies=[Depends(require_roles(*_STUDENT_ALLOWED_ROLES))],
)
async def get_my_attendance(
    db: Session = Depends(get_role_scoped_db),
    current_user: User = Depends(get_current_user),
):
    """Return per-course attendance summary for the logged-in student."""
    return student_portal_service.get_my_attendance(db, current_user)


@router.get(
    "/student-portal/me/profile",
    dependencies=[Depends(require_roles(*_STUDENT_ALLOWED_ROLES))],
)
async def get_my_profile(
    db: Session = Depends(get_role_scoped_db),
    current_user: User = Depends(get_current_user),
):
    """Return the profile for the logged-in student."""
    profile = student_portal_service.get_my_profile(db, current_user)
    if profile is None:
        return {}
    return profile


@router.get(
    "/student-portal/me/schedule",
    dependencies=[Depends(require_roles(*_STUDENT_ALLOWED_ROLES))],
)
async def get_my_schedule(
    db: Session = Depends(get_role_scoped_db),
    current_user: User = Depends(get_current_user),
):
    """Return the class schedule for the logged-in student."""
    return student_portal_service.get_my_schedule(db, current_user)


@router.post(
    "/student-portal/me/excuse-requests",
    dependencies=[Depends(require_roles(*_STUDENT_ALLOWED_ROLES))],
    status_code=201,
)
async def create_excuse_request(
    payload: ExcuseRequestCreate,
    db: Session = Depends(get_role_scoped_db),
    current_user: User = Depends(get_current_user),
):
    """Student submits an excuse request."""
    from app.services.notification_service import (
        create_notification,
        notify_faculty_admins,
        NotificationType,
    )
    result = student_portal_service.create_excuse_request(
        db, current_user,
        payload.request_date, payload.reason, payload.course_id,
    )

    course_label = result.get("course_name") or "all courses"
    request_date = result.get("request_date", str(payload.request_date))

    # 1. Notify all faculty users in this faculty so they see it immediately
    if current_user.faculty_id:
        student_name = result.get("student_name") or current_user.username
        notify_faculty_admins(
            db,
            current_user.faculty_id,
            "New Excuse Request",
            f"{student_name} submitted an excuse request for {request_date} ({course_label}).",
            NotificationType.INFO,
            link="/faculty/excuse-requests",
        )

    # 2. Send the student a confirmation notification so it appears in their bell
    create_notification(
        db,
        current_user.id,
        "Excuse Request Submitted",
        f"Your excuse request for {request_date} ({course_label}) has been submitted and is pending faculty review.",
        NotificationType.INFO,
        link="/student/attendance",
    )

    return result


@router.get(
    "/student-portal/me/excuse-requests",
    dependencies=[Depends(require_roles(*_STUDENT_ALLOWED_ROLES))],
)
async def list_my_excuse_requests(
    db: Session = Depends(get_role_scoped_db),
    current_user: User = Depends(get_current_user),
):
    """Return all excuse requests for the logged-in student."""
    return student_portal_service.list_my_excuse_requests(db, current_user)


@router.get(
    "/student-portal/me/attendance/{course_id}/sessions",
    dependencies=[Depends(require_roles(*_STUDENT_ALLOWED_ROLES))],
)
async def get_my_session_history(
    course_id: int,
    db: Session = Depends(get_role_scoped_db),
    current_user: User = Depends(get_current_user),
):
    """Return individual session-level attendance records for a specific course."""
    return student_portal_service.get_my_session_history(db, current_user, course_id)


# ------------------------------------------------------------------
# Legacy ID-scoped helpers (kept for backward compatibility)
# ------------------------------------------------------------------

def _is_staff(user: User) -> bool:
    staff_roles = {"SUPER_ADMIN", "ACADEMIA", "FACULTY", "ADMISSIONS", "TEACHER"}
    return any(role.name in staff_roles for role in user.roles)


def _require_student_access(current_user: User, student_id: int) -> None:
    if current_user.id == student_id or _is_staff(current_user):
        return
    raise HTTPException(status_code=403, detail="You are not allowed to access this student resource")


@router.get("/students/{student_id}/attendance", response_model=list[AttendanceResponse])
async def list_student_attendance(
    student_id: int,
    db: Session = Depends(get_role_scoped_db),
    current_user: User = Depends(get_current_user),
):
    _require_student_access(current_user, student_id)
    return student_portal_service.list_attendance(db, student_id)


@router.post("/attendance", response_model=AttendanceResponse)
async def create_student_attendance(
    payload: AttendanceCreate,
    db: Session = Depends(get_role_scoped_db),
    current_user: User = Depends(get_current_user),
):
    _require_student_access(current_user, payload.student_id)
    return student_portal_service.create_attendance(db, payload)


@router.get("/attendance/{attendance_id}", response_model=AttendanceResponse)
async def get_student_attendance(
    attendance_id: int,
    db: Session = Depends(get_role_scoped_db),
    current_user: User = Depends(get_current_user),
):
    attendance = student_portal_service.get_attendance(db, attendance_id)
    _require_student_access(current_user, attendance.student_id)
    return attendance


@router.get("/students/{student_id}/schedule", response_model=list[ScheduleResponse])
async def list_student_schedule(
    student_id: int,
    db: Session = Depends(get_role_scoped_db),
    current_user: User = Depends(get_current_user),
):
    _require_student_access(current_user, student_id)
    return student_portal_service.list_schedules(db, student_id)


@router.post("/schedule", response_model=ScheduleResponse)
async def create_student_schedule(
    payload: ScheduleCreate,
    db: Session = Depends(get_role_scoped_db),
    current_user: User = Depends(get_current_user),
):
    _require_student_access(current_user, payload.student_id)
    return student_portal_service.create_schedule(db, payload)


@router.get("/schedule/{schedule_id}", response_model=ScheduleResponse)
async def get_student_schedule(
    schedule_id: int,
    db: Session = Depends(get_role_scoped_db),
    current_user: User = Depends(get_current_user),
):
    schedule = student_portal_service.get_schedule(db, schedule_id)
    _require_student_access(current_user, schedule.student_id)
    return schedule
