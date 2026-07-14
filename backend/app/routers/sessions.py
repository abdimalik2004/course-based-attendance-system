from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import false as sa_false

from app.core.security import get_current_user, require_roles
from app.db.faculty_scope import enforce_faculty_scope, get_optional_faculty_scope_context
from app.db.models import AcademicYear, AcademicYearStatus, AttendanceRecord, AttendanceSession, AttendanceStatus, Course, CourseAssignment, CourseSemesterAssignment, SessionStatus, Student, Teacher, User
from app.db.role_scoped import get_role_scoped_db
from app.schemas.attendance import AttendanceSessionEndRequest, AttendanceSessionRead, AttendanceSessionStartRequest
from app.services.attendance_service import attendance_service
from app.utils.activity_logger import log_activity
from app.services.notification_service import create_notification, NotificationType
from app.routers.academic_structure import _sync_semester_statuses


router = APIRouter(prefix="/sessions", tags=["sessions"])


# Roles that can start/end sessions (must be a teacher or admin actor)
_SESSION_MGMT_ROLES = ("SUPER_ADMIN", "ACADEMIA", "FACULTY", "TEACHER")


def _notify_absent_students(db: Session, session: AttendanceSession) -> None:
    """Send an in-app notification to every student marked ABSENT in this session."""
    absent_records = (
        db.query(AttendanceRecord)
        .filter(
            AttendanceRecord.session_id == session.id,
            AttendanceRecord.status == AttendanceStatus.ABSENT,
        )
        .all()
    )

    course_title = session.course.title if session.course else f"Course #{session.course_id}"

    for record in absent_records:
        student_user = (
            db.query(User)
            .filter(User.student_id == record.student_id)
            .first()
        )
        if not student_user:
            continue
        create_notification(
            db,
            student_user.id,
            f"Absent — {course_title}",
            (
                f"You were marked absent for today's {course_title} session. "
                f"If you were present, you can submit an excuse request from your Attendance page."
            ),
            NotificationType.WARNING,
            "/student/attendance",
        )
# Roles that can READ sessions
_SESSION_ROLES = ("SUPER_ADMIN", "HR", "ACADEMIA", "FACULTY", "TEACHER")


@router.post(
    "/start",
    response_model=AttendanceSessionRead,
    dependencies=[Depends(require_roles(*_SESSION_MGMT_ROLES))],
)
def start_session(
    payload: AttendanceSessionStartRequest,
    db: Session = Depends(get_role_scoped_db),
    current_user: User = Depends(get_current_user),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    # Auto-deactivate any expired semesters before we check
    _sync_semester_statuses(db)

    # Auto-close any ACTIVE sessions from a previous day that were never ended
    # (e.g. browser was closed without clicking End Session).
    from datetime import date as _date
    today = _date.today()
    stale_sessions = (
        db.query(AttendanceSession)
        .filter(
            AttendanceSession.status == SessionStatus.ACTIVE,
            AttendanceSession.session_date < today,
        )
        .all()
    )
    for stale in stale_sessions:
        attendance_service.close_session_and_mark_absent(db, stale)
    if stale_sessions:
        db.commit()

    course = db.query(Course).filter(Course.id == payload.course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    if faculty_scope is not None:
        enforce_faculty_scope(course.faculty_id, faculty_scope)

    # Check that this course belongs to a currently active semester
    today = date.today()
    semester_assignment = (
        db.query(CourseSemesterAssignment)
        .join(AcademicYear, AcademicYear.id == CourseSemesterAssignment.academic_year_id)
        .filter(CourseSemesterAssignment.course_id == payload.course_id)
        .first()
    )
    if semester_assignment is not None:
        ay = semester_assignment.academic_year
        if ay.status != AcademicYearStatus.ACTIVE or ay.end_date < today or ay.start_date > today:
            term_label = f"{ay.academic_year} — {ay.term_name}" if ay else "its semester"
            raise HTTPException(
                status_code=403,
                detail=(
                    f"Attendance cannot be taken for this course. "
                    f"The semester '{term_label}' is not currently active "
                    f"(it runs {ay.start_date} to {ay.end_date})."
                ),
            )

    # Admin actors (SUPER_ADMIN without a TEACHER/FACULTY role) are restricted to Lab sessions only
    actor_role_names = {role.name for role in current_user.roles}
    is_admin_actor = "SUPER_ADMIN" in actor_role_names and not {"TEACHER", "FACULTY"}.intersection(actor_role_names)
    if is_admin_actor and payload.session_type != "Lab":
        raise HTTPException(
            status_code=403,
            detail="Admin role is only permitted to start Lab sessions. Lecture and Tutorial sessions must be started by a teacher.",
        )

    result = attendance_service.start_session(
        db=db,
        course_id=payload.course_id,
        schedule_id=payload.schedule_id,
        session_type=payload.session_type,
        actor_id=current_user.id,
        notes=payload.notes,
    )
    if not result["ok"]:
        _bad_request_codes = {"WRONG_DAY", "TOO_EARLY", "TOO_LATE", "SESSION_ALREADY_COMPLETED"}
        status_code = 400 if result.get("error_code") in _bad_request_codes else 403
        raise HTTPException(status_code=status_code, detail=result["message"])
    log_activity(
        action=f"Attendance Session Started - {course.title} (Course #{payload.course_id})",
        user=current_user,
        db=db,
    )
    return result["session"]


@router.post(
    "/end",
    response_model=AttendanceSessionRead,
    dependencies=[Depends(require_roles(*_SESSION_MGMT_ROLES))],
)
def end_session(
    payload: AttendanceSessionEndRequest,
    db: Session = Depends(get_role_scoped_db),
    current_user: User = Depends(get_current_user),
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
    log_activity(
        action=f"Attendance Session Closed - {session.course.title} (Session #{payload.session_id})",
        user=current_user,
        db=db,
    )
    # Notify every absent student — only fires when the session was actually
    # closed just now (not if it was already closed before this call)
    if result.get("ended"):
        _notify_absent_students(db, session)
    return result["session"]


@router.get("", response_model=list[AttendanceSessionRead], dependencies=[Depends(require_roles(*_SESSION_ROLES))])
def list_sessions(
    db: Session = Depends(get_role_scoped_db),
    current_user: User = Depends(get_current_user),
    faculty_scope = Depends(get_optional_faculty_scope_context),
    limit: int = Query(default=200, ge=1, le=2000),
    skip: int = Query(default=0, ge=0),
):
    query = db.query(AttendanceSession).join(Course, Course.id == AttendanceSession.course_id)
    if faculty_scope is not None:
        query = query.filter(Course.faculty_id == faculty_scope.faculty_id)

    # Scope TEACHER to only their own sessions — always enforce, never fall through
    role_names = {role.name for role in current_user.roles}
    if "TEACHER" in role_names and not {"SUPER_ADMIN", "ACADEMIA", "FACULTY"}.intersection(role_names):
        teacher = db.query(Teacher).filter(Teacher.user_id == current_user.id).first()
        if teacher:
            query = query.filter(AttendanceSession.teacher_id == teacher.id)
        else:
            # No Teacher record linked to this user account — return nothing
            query = query.filter(sa_false())

    return query.order_by(AttendanceSession.start_time.desc()).offset(skip).limit(limit).all()


@router.get(
    "/active",
    response_model=list[AttendanceSessionRead],
    dependencies=[Depends(require_roles(*_SESSION_MGMT_ROLES))],
)
def list_active_sessions(
    db: Session = Depends(get_role_scoped_db),
    current_user: User = Depends(get_current_user),
    faculty_scope = Depends(get_optional_faculty_scope_context),
    teacher_id: int | None = Query(default=None, description="Filter by teacher id"),
    course_id: int | None = Query(default=None, description="Filter by course id"),
    faculty_id: int | None = Query(default=None, description="Filter by faculty id"),
):
    from datetime import date as _date
    today = _date.today()

    # Auto-close any ACTIVE sessions left open from a previous day so they
    # never resurface as "active" in the banner on future days.
    stale = (
        db.query(AttendanceSession)
        .filter(
            AttendanceSession.status == SessionStatus.ACTIVE,
            AttendanceSession.session_date < today,
        )
        .all()
    )
    for s in stale:
        attendance_service.close_session_and_mark_absent(db, s)
    if stale:
        db.commit()

    query = (
        db.query(AttendanceSession)
        .join(Course, Course.id == AttendanceSession.course_id)
        .filter(
            AttendanceSession.status == SessionStatus.ACTIVE,
            AttendanceSession.session_date == today,   # only today's sessions
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


@router.get(
    "/{session_id}/records",
    dependencies=[Depends(require_roles(*_SESSION_ROLES))],
)
def get_session_records(
    session_id: int,
    db: Session = Depends(get_role_scoped_db),
    current_user: User = Depends(get_current_user),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    """Return all attendance records for a session with student details."""
    session = db.query(AttendanceSession).filter(AttendanceSession.id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if faculty_scope is not None:
        enforce_faculty_scope(session.course.faculty_id, faculty_scope)

    rows = (
        db.query(AttendanceRecord, Student)
        .join(Student, Student.id == AttendanceRecord.student_id)
        .filter(AttendanceRecord.session_id == session_id)
        .order_by(AttendanceRecord.recognized_at.asc())
        .all()
    )

    return [
        {
            "id": record.id,
            "student_id": student.id,
            "student_number": student.student_number,
            "student_name": student.full_name,
            "course_id": record.course_id,
            "session_id": record.session_id,
            "status": record.status.value,
            "confidence": round(record.confidence, 3),
            "recognized_at": record.recognized_at.isoformat(),
        }
        for record, student in rows
    ]
