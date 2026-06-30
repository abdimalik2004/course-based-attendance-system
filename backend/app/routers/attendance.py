from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import or_, false as sa_false
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.rate_limit import rate_limit_dependency
from app.core.security import require_roles, get_current_user
from app.db.faculty_scope import enforce_faculty_scope, get_optional_faculty_scope_context
from app.db.models import AttendanceRecord, AttendanceSession, AttendanceStatus, Course, CourseAssignment, Student, Faculty, Department, Teacher, User
from app.db.role_scoped import get_role_scoped_db
from app.schemas.attendance import AttendanceFrameRequest
from app.services.attendance_service import attendance_service


class UpdateAttendanceStatusRequest(BaseModel):
    status: str


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
        Depends(require_roles("SUPER_ADMIN", "ACADEMIA", "FACULTY", "TEACHER")),
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


@router.get("/records", dependencies=[Depends(require_roles("SUPER_ADMIN", "ADMIN", "ACADEMIA", "FACULTY", "TEACHER", "HR", "ADMISSIONS"))])
def list_attendance_records(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=20, ge=1, le=200),
    search: str | None = Query(default=None),
    faculty: str | None = Query(default=None),
    department: str | None = Query(default=None),
    course: str | None = Query(default=None),
    course_id: int | None = Query(default=None),
    status: str | None = Query(default=None),
    db: Session = Depends(get_role_scoped_db),
    current_user: User = Depends(get_current_user),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    query = (
        db.query(AttendanceRecord, Student, Course)
        .join(Student, Student.id == AttendanceRecord.student_id)
        .join(Course, Course.id == AttendanceRecord.course_id)
    )
    if faculty_scope is not None:
        query = query.filter(Course.faculty_id == faculty_scope.faculty_id)

    # Scope TEACHER: only show records from sessions THIS teacher personally managed.
    # Admin or another teacher may have run sessions on the same course — exclude those.
    role_names = {role.name for role in current_user.roles}
    if "TEACHER" in role_names and not {"SUPER_ADMIN", "ACADEMIA", "FACULTY", "HR", "ADMISSIONS", "ADMIN"}.intersection(role_names):
        teacher = db.query(Teacher).filter(Teacher.user_id == current_user.id).first()
        if teacher:
            # Join to the session and filter by the session's teacher_id
            query = (
                query
                .join(AttendanceSession, AttendanceSession.id == AttendanceRecord.session_id)
                .filter(AttendanceSession.teacher_id == teacher.id)
            )
        else:
            # No Teacher record linked to this user — return nothing
            query = query.filter(sa_false())

    if search:
        pattern = f"%{search.strip()}%"
        query = query.filter(
            or_(
                Student.full_name.ilike(pattern),
                Student.student_number.ilike(pattern),
                Course.title.ilike(pattern),
                Course.code.ilike(pattern),
            )
        )

    if faculty and faculty.lower() != "all":
        pattern = f"%{faculty.strip()}%"
        query = query.filter(
            or_(
                Course.faculty.has(Faculty.name.ilike(pattern)),
                Student.faculty.has(Faculty.name.ilike(pattern)),
                Course.title.ilike(pattern),
                Course.code.ilike(pattern),
            )
        )

    if department and department.lower() != "all":
        pattern = f"%{department.strip()}%"
        query = query.filter(
            or_(
                Course.department.has(Department.name.ilike(pattern)),
                Student.department.has(Department.name.ilike(pattern)),
                Course.title.ilike(pattern),
                Course.code.ilike(pattern),
            )
        )

    if course_id is not None:
        query = query.filter(AttendanceRecord.course_id == course_id)
    elif course and course.lower() != "all":
        pattern = f"%{course.strip()}%"
        query = query.filter(
            or_(
                Course.title.ilike(pattern),
                Course.code.ilike(pattern),
            )
        )

    if status and status.lower() != "all":
        normalized_status = status.strip().upper()
        if normalized_status == "PRESENT":
            status_filter = AttendanceStatus.PRESENT
        elif normalized_status == "LATE":
            status_filter = AttendanceStatus.LATE
        elif normalized_status == "ABSENT":
            status_filter = AttendanceStatus.ABSENT
        elif normalized_status == "EXCUSED":
            status_filter = AttendanceStatus.EXCUSED
        else:
            status_filter = None

        if status_filter is not None:
            query = query.filter(AttendanceRecord.status == status_filter)

    total = query.count()
    records = query.order_by(AttendanceRecord.recognized_at.desc()).offset((page - 1) * limit).limit(limit).all()

    data = []
    for record, student, course_obj in records:
        data.append(
            {
                "id": record.id,
                "courseId": course_obj.id,
                "studentName": student.full_name,
                "course": course_obj.title,
                "sessionId": f"SES-{record.session_id}",
                "status": record.status.value,
                "confidence": round(record.confidence, 1) if record.confidence is not None else None,
                "recognizedAt": record.recognized_at.isoformat() if record.recognized_at else None,
                "faculty": course_obj.faculty.name if course_obj.faculty else None,
                "department": course_obj.department.name if course_obj.department else None,
                "attendedSessions": 1 if record.status in (AttendanceStatus.PRESENT, AttendanceStatus.LATE) else 0,
                "totalSessions": 1,
            }
        )

    return {"data": data, "total": total}


@router.put(
    "/records/{record_id}",
    dependencies=[Depends(require_roles("SUPER_ADMIN", "ADMIN", "ACADEMIA", "FACULTY", "TEACHER", "HR", "ADMISSIONS"))],
)
def update_attendance_record_status(
    record_id: int,
    body: UpdateAttendanceStatusRequest,
    db: Session = Depends(get_role_scoped_db),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    record = db.query(AttendanceRecord).filter(AttendanceRecord.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Attendance record not found")

    if faculty_scope is not None:
        course = db.query(Course).filter(Course.id == record.course_id).first()
        if not course:
            raise HTTPException(status_code=404, detail="Course not found")
        enforce_faculty_scope(course.faculty_id, faculty_scope)

    normalized_status = body.status.strip().upper()
    if normalized_status != "EXCUSED":
        raise HTTPException(status_code=400, detail="Faculty users can only change Absent records to Excused")

    if record.status != AttendanceStatus.ABSENT:
        raise HTTPException(status_code=400, detail="Only absent records can be marked as excused")

    record.status = AttendanceStatus.EXCUSED
    db.add(record)
    db.commit()
    db.refresh(record)
    return {"id": record.id, "status": record.status.value}
