from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.models import User
from app.db.role_scoped import get_role_scoped_db
from app.schemas.student_portal import AttendanceCreate, AttendanceResponse, ScheduleCreate, ScheduleResponse
from app.services.student_portal_service import student_portal_service


router = APIRouter(tags=["student-portal"])


def _is_staff(user: User) -> bool:
    staff_roles = {"SUPER_ADMIN", "ACADEMIA", "FACULTY", "FACULTY_ADMIN", "ADMISSIONS", "TEACHER"}
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
