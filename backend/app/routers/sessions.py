from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.security import get_current_user, require_roles
from app.db.faculty_scope import enforce_faculty_scope, get_optional_faculty_scope_context
from app.db.models import AttendanceSession, Course, CourseAssignment, SessionStatus, Teacher, User
from app.db.role_scoped import get_role_scoped_db
from app.schemas.attendance import AttendanceSessionEndRequest, AttendanceSessionRead, AttendanceSessionStartRequest
from app.services.attendance_service import attendance_service


router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post(
    "/start",
    response_model=AttendanceSessionRead,
    dependencies=[Depends(require_roles("ACADEMIA", "FACULTY", "TEACHER"))],
)
def start_session(
    payload: AttendanceSessionStartRequest,
    db: Session = Depends(get_role_scoped_db),
    current_user: User = Depends(get_current_user),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    course = db.query(Course).filter(Course.id == payload.course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    if faculty_scope is not None:
        enforce_faculty_scope(course.faculty_id, faculty_scope)

    result = attendance_service.start_session(
        db=db,
        course_id=payload.course_id,
        schedule_id=payload.schedule_id,
        instructor_id=current_user.id,
    )
    if not result["ok"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result["session"]


@router.post(
    "/end",
    response_model=AttendanceSessionRead,
    dependencies=[Depends(require_roles("ACADEMIA", "FACULTY", "TEACHER"))],
)
def end_session(
    payload: AttendanceSessionEndRequest,
    db: Session = Depends(get_role_scoped_db),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    session = db.query(AttendanceSession).filter(AttendanceSession.id == payload.session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if faculty_scope is not None:
        enforce_faculty_scope(session.course.faculty_id, faculty_scope)

    result = attendance_service.end_session(db=db, session_id=payload.session_id)
    if not result["ok"]:
        raise HTTPException(status_code=400, detail=result["message"])
    return result["session"]


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
    query = (
        db.query(AttendanceSession)
        .join(Course, Course.id == AttendanceSession.course_id)
        .filter(AttendanceSession.status == SessionStatus.ACTIVE)
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
