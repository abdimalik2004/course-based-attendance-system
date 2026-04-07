from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.security import get_current_user, require_roles
from app.db.faculty_scope import enforce_faculty_scope, get_optional_faculty_scope_context
from app.db.models import AttendanceSession, Course, CourseAssignment, SessionStatus, Teacher, User
from app.db.role_scoped import get_role_scoped_db
from app.schemas.attendance import AttendanceSessionRead


router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.get("", response_model=list[AttendanceSessionRead], dependencies=[Depends(require_roles("ACADEMIA", "FACULTY", "TEACHER"))])
def list_sessions(
    db: Session = Depends(get_role_scoped_db),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    query = db.query(AttendanceSession).join(Course, Course.id == AttendanceSession.course_id)
    if faculty_scope is not None:
        query = query.filter(Course.faculty_id == faculty_scope.faculty_id)
    return query.order_by(AttendanceSession.start_time.desc()).all()


@router.get(
    "/active",
    response_model=list[AttendanceSessionRead],
    dependencies=[Depends(require_roles("ACADEMIA", "FACULTY", "TEACHER"))],
)
def list_active_sessions(
    db: Session = Depends(get_role_scoped_db),
    current_user: User = Depends(get_current_user),
    faculty_scope = Depends(get_optional_faculty_scope_context),
    teacher_id: int | None = Query(default=None, description="Filter by teacher id"),
    course_id: int | None = Query(default=None, description="Filter by course id"),
    faculty_id: int | None = Query(default=None, description="Filter by faculty id"),
):
    now = datetime.now()
    query = (
        db.query(AttendanceSession)
        .join(Course, Course.id == AttendanceSession.course_id)
        .filter(
            AttendanceSession.status == SessionStatus.ACTIVE,
            AttendanceSession.start_time <= now,
            AttendanceSession.end_time >= now,
        )
    )

    if faculty_scope is not None:
        query = query.filter(Course.faculty_id == faculty_scope.faculty_id)

    role_names = {role.name for role in current_user.roles}
    if "TEACHER" in role_names:
        teacher = db.query(Teacher).filter(Teacher.user_id == current_user.id).first()
        if not teacher:
            raise HTTPException(status_code=403, detail="Teacher profile is not linked to current user")
        teacher_id = teacher.id

    if teacher_id is not None:
        query = query.join(CourseAssignment, CourseAssignment.course_id == AttendanceSession.course_id)
        query = query.filter(CourseAssignment.teacher_id == teacher_id)

    if course_id is not None:
        if faculty_scope is not None:
            target_course = db.query(Course).filter(Course.id == course_id).first()
            if not target_course:
                raise HTTPException(status_code=404, detail="Course not found")
            enforce_faculty_scope(target_course.faculty_id, faculty_scope)
        query = query.filter(AttendanceSession.course_id == course_id)

    if faculty_id is not None:
        query = query.filter(Course.faculty_id == faculty_id)

    return query.order_by(AttendanceSession.start_time.desc()).all()
