from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import or_, false as sa_false
from sqlalchemy.exc import IntegrityError

from app.core.security import get_current_user, require_roles
from app.db.faculty_scope import enforce_faculty_scope, get_optional_faculty_scope_context
from app.db.models import (
    Faculty,
    Course,
    CourseAssignment,
    CourseSchedule,
    Enrollment,
    Student,
    Teacher,
    Department,
    normalize_course_title,
)
from app.db.role_scoped import get_role_scoped_db
from app.services.enrollment_service import auto_enroll_existing_students_for_course
from app.utils.activity_logger import log_activity
from app.schemas.course import (
    CourseAssignmentCreate,
    CourseAssignmentRead,
    CourseAssignmentUpdate,
    CourseCreate,
    CourseRead,
    CourseUpdate,
    PaginatedCourseAssignmentRead,
    PaginatedCourseRead,
)
from app.schemas.student import StudentRead
from app.utils.weekday_utils import weekdays_intersect


router = APIRouter(prefix="/courses", tags=["courses"])


def _course_code_prefix(*, faculty_code: str) -> str:
    prefix = faculty_code.strip().upper()
    if not prefix:
        raise HTTPException(status_code=400, detail="Faculty code is required for course code generation")
    return prefix


def _generate_course_code(db: Session, *, faculty_code: str) -> str:
    prefix = _course_code_prefix(faculty_code=faculty_code)

    existing = (
        db.query(Course.code)
        .filter(Course.code.ilike(f"{prefix}%"))
        .all()
    )

    max_seq = 0
    for (value,) in existing:
        if not value:
            continue
        normalized = value.strip().upper()
        if not normalized.startswith(prefix):
            continue
        suffix = normalized[len(prefix):]
        if suffix.isdigit():
            max_seq = max(max_seq, int(suffix))

    return f"{prefix}{max_seq + 1:03d}"


def _time_windows_overlap(start_a, end_a, start_b, end_b) -> bool:
    return start_a < end_b and end_a > start_b


def _ensure_unique_course_title_in_faculty(
    db: Session,
    *,
    faculty_id: int,
    title: str,
    exclude_course_id: int | None = None,
) -> None:
    normalized_title = normalize_course_title(title)
    query = db.query(Course).filter(
        Course.faculty_id == faculty_id,
        Course.normalized_title == normalized_title,
    )
    if exclude_course_id is not None:
        query = query.filter(Course.id != exclude_course_id)
    exists = query.first()
    if exists is not None:
        raise HTTPException(status_code=409, detail="Course title already exists in this faculty")


def _ensure_student_schedule_has_no_conflict(db: Session, *, student_id: int, course_id: int) -> None:
    target_schedules = db.query(CourseSchedule).filter(CourseSchedule.course_id == course_id).all()
    if not target_schedules:
        return

    existing_schedules = (
        db.query(CourseSchedule, Course)
        .join(Course, Course.id == CourseSchedule.course_id)
        .join(Enrollment, Enrollment.course_id == Course.id)
        .filter(Enrollment.student_id == student_id)
        .all()
    )

    for target in target_schedules:
        for existing, existing_course in existing_schedules:
            if not weekdays_intersect(target.weekday, existing.weekday):
                continue
            if not _time_windows_overlap(target.start_time, target.end_time, existing.start_time, existing.end_time):
                continue
            raise HTTPException(
                status_code=400,
                detail=(
                    "Student already has another enrolled course session at the same time: "
                    f"{existing_course.code}"
                ),
            )


@router.post("", response_model=CourseRead, dependencies=[Depends(require_roles("ACADEMIA"))])
def create_course(
    payload: CourseCreate,
    db: Session = Depends(get_role_scoped_db),
    faculty_scope = Depends(get_optional_faculty_scope_context),
    current_user = Depends(get_current_user),
):
    faculty = db.query(Faculty).filter(Faculty.id == payload.faculty_id).first()
    if not faculty:
        raise HTTPException(status_code=404, detail="Faculty not found")

    if faculty_scope is not None:
        enforce_faculty_scope(faculty.id, faculty_scope)

    # Verify that the department belongs to the faculty
    department = db.query(Department).filter(
        Department.id == payload.department_id,
        Department.faculty_id == payload.faculty_id
    ).first()
    if not department:
        raise HTTPException(status_code=404, detail="Department not found in the specified faculty")

    _ensure_unique_course_title_in_faculty(
        db,
        faculty_id=faculty.id,
        title=payload.title,
    )

    obj = Course(
        faculty_id=faculty.id,
        department_id=payload.department_id,
        code=_generate_course_code(db, faculty_code=faculty.code),
        title=payload.title,
        normalized_title=normalize_course_title(payload.title),
    )
    db.add(obj)
    try:
        db.flush()
        auto_enroll_existing_students_for_course(db, obj)
        log_activity(
            action=f"Course Created - {payload.title}",
            user=current_user,
            db=db,
        )
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Course code already exists in this faculty") from exc
    db.refresh(obj)
    return obj


@router.get(
    "",
    response_model=PaginatedCourseRead,
    dependencies=[Depends(require_roles("SUPER_ADMIN", "ADMIN", "ACADEMIA", "FACULTY", "TEACHER", "HR", "ADMISSION", "ADMISSIONS"))],
)
def list_courses(
    db: Session = Depends(get_role_scoped_db),
    skip: int = Query(default=0, ge=0, description="Number of rows to skip", examples=[0]),
    limit: int = Query(default=50, ge=1, le=200, description="Page size", examples=[20]),
    faculty_id: int | None = Query(default=None, description="Filter by faculty id", examples=[1]),
    search: str | None = Query(default=None, description="Search by course code or title", examples=["CSC"]),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    query = db.query(Course)
    if faculty_scope is not None:
        if faculty_id is not None:
            enforce_faculty_scope(faculty_id, faculty_scope)
        else:
            faculty_id = faculty_scope.faculty_id
    if faculty_id is not None:
        query = query.filter(Course.faculty_id == faculty_id)
    if search:
        pattern = f"%{search.strip()}%"
        query = query.filter(or_(Course.code.ilike(pattern), Course.title.ilike(pattern)))
    total = query.count()
    items = query.order_by(Course.code).offset(skip).limit(limit).all()
    return {"items": items, "total": total, "skip": skip, "limit": limit}


@router.put("/{course_id}", response_model=CourseRead, dependencies=[Depends(require_roles("ACADEMIA"))])
def update_course(
    course_id: int,
    payload: CourseUpdate,
    db: Session = Depends(get_role_scoped_db),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    obj = db.query(Course).filter(Course.id == course_id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Course not found")

    target_faculty_id = payload.faculty_id if payload.faculty_id is not None else obj.faculty_id
    target_faculty = db.query(Faculty).filter(Faculty.id == target_faculty_id).first()
    if not target_faculty:
        raise HTTPException(status_code=404, detail="Faculty not found")
    if faculty_scope is not None:
        enforce_faculty_scope(target_faculty.id, faculty_scope)

    # Verify that the department belongs to the faculty if department_id is being updated
    if payload.department_id is not None:
        department = db.query(Department).filter(
            Department.id == payload.department_id,
            Department.faculty_id == target_faculty_id
        ).first()
        if not department:
            raise HTTPException(status_code=404, detail="Department not found in the specified faculty")

    next_title = payload.title if payload.title is not None else obj.title
    _ensure_unique_course_title_in_faculty(
        db,
        faculty_id=target_faculty.id,
        title=next_title,
        exclude_course_id=obj.id,
    )

    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(obj, field, value)

    obj.faculty_id = target_faculty.id
    obj.normalized_title = normalize_course_title(next_title)

    db.add(obj)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Update conflicts with existing course data") from exc
    db.refresh(obj)
    return obj


@router.delete("/{course_id}", dependencies=[Depends(require_roles("ACADEMIA"))])
def delete_course(course_id: int, db: Session = Depends(get_role_scoped_db)):
    obj = db.query(Course).filter(Course.id == course_id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Course not found")
    db.delete(obj)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Cannot delete course due to related records") from exc
    return {"deleted": True, "course_id": course_id}


@router.post("/assign-teacher", dependencies=[Depends(require_roles("FACULTY"))])
def assign_teacher(
    payload: CourseAssignmentCreate,
    db: Session = Depends(get_role_scoped_db),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    course = db.query(Course).filter(Course.id == payload.course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    teacher = db.query(Teacher).filter(Teacher.id == payload.teacher_id).first()
    if not teacher:
        raise HTTPException(status_code=404, detail="Teacher not found")

    if faculty_scope is not None:
        enforce_faculty_scope(course.faculty_id, faculty_scope)

    if teacher.faculty_id != course.faculty_id:
        raise HTTPException(
            status_code=400,
            detail="Teacher faculty does not match course faculty",
        )

    assignment = CourseAssignment(**payload.model_dump())
    db.add(assignment)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Teacher already assigned to this course") from exc
    db.refresh(assignment)
    return {
        "id": assignment.id,
        "course_id": assignment.course_id,
        "teacher_id": assignment.teacher_id,
        "is_primary": assignment.is_primary,
    }


@router.get(
    "/assignments",
    response_model=PaginatedCourseAssignmentRead,
    dependencies=[Depends(require_roles("FACULTY", "TEACHER", "ACADEMIA", "HR"))],
)
def list_course_assignments(
    db: Session = Depends(get_role_scoped_db),
    skip: int = Query(default=0, ge=0, description="Number of rows to skip", examples=[0]),
    limit: int = Query(default=50, ge=1, le=200, description="Page size", examples=[20]),
    faculty_id: int | None = Query(default=None, description="Filter by faculty id", examples=[1]),
    course_id: int | None = Query(default=None, description="Filter by course id", examples=[1]),
    teacher_id: int | None = Query(default=None, description="Filter by teacher id", examples=[1]),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    query = db.query(CourseAssignment).join(Course, Course.id == CourseAssignment.course_id)
    if faculty_scope is not None:
        if faculty_id is not None:
            enforce_faculty_scope(faculty_id, faculty_scope)
        else:
            faculty_id = faculty_scope.faculty_id
    if faculty_id is not None:
        query = query.filter(Course.faculty_id == faculty_id)
    if course_id is not None:
        query = query.filter(CourseAssignment.course_id == course_id)
    if teacher_id is not None:
        query = query.filter(CourseAssignment.teacher_id == teacher_id)
    total = query.count()
    items = (
        query
        .options(joinedload(CourseAssignment.course))
        .order_by(Course.code, CourseAssignment.id.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    items_data = [
        {
            "id": a.id,
            "course_id": a.course_id,
            "teacher_id": a.teacher_id,
            "is_primary": a.is_primary,
            "course_title": a.course.title if a.course is not None else None,
            "course_code": a.course.code if a.course is not None else None,
        }
        for a in items
    ]
    return {"items": items_data, "total": total, "skip": skip, "limit": limit}


@router.put("/assignments/{assignment_id}", response_model=CourseAssignmentRead, dependencies=[Depends(require_roles("FACULTY"))])
def update_course_assignment(
    assignment_id: int,
    payload: CourseAssignmentUpdate,
    db: Session = Depends(get_role_scoped_db),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    assignment = db.query(CourseAssignment).filter(CourseAssignment.id == assignment_id).first()
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found")

    course = db.query(Course).filter(Course.id == assignment.course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    if faculty_scope is not None:
        enforce_faculty_scope(course.faculty_id, faculty_scope)

    next_teacher_id = payload.teacher_id if payload.teacher_id is not None else assignment.teacher_id
    teacher = db.query(Teacher).filter(Teacher.id == next_teacher_id).first()
    if not teacher:
        raise HTTPException(status_code=404, detail="Teacher not found")
    if teacher.faculty_id != course.faculty_id:
        raise HTTPException(status_code=400, detail="Teacher faculty does not match course faculty")

    if payload.teacher_id is not None:
        assignment.teacher_id = payload.teacher_id
    if payload.is_primary is not None:
        assignment.is_primary = payload.is_primary

    db.add(assignment)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Update conflicts with existing assignment data") from exc
    db.refresh(assignment)
    return assignment


@router.delete("/assignments/{assignment_id}", dependencies=[Depends(require_roles("FACULTY"))])
def delete_course_assignment(
    assignment_id: int,
    db: Session = Depends(get_role_scoped_db),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    assignment = db.query(CourseAssignment).filter(CourseAssignment.id == assignment_id).first()
    if not assignment:
        raise HTTPException(status_code=404, detail="Assignment not found")

    course = db.query(Course).filter(Course.id == assignment.course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    if faculty_scope is not None:
        enforce_faculty_scope(course.faculty_id, faculty_scope)

    db.delete(assignment)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Cannot delete assignment due to related records") from exc
    return {"deleted": True, "assignment_id": assignment_id}


@router.post("/{course_id}/enroll/{student_id}", dependencies=[Depends(require_roles("FACULTY"))])
def enroll_student(
    course_id: int,
    student_id: int,
    db: Session = Depends(get_role_scoped_db),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    if faculty_scope is not None:
        enforce_faculty_scope(course.faculty_id, faculty_scope)

    student = db.query(Student).filter(Student.id == student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    if student.faculty_id != course.faculty_id:
        raise HTTPException(status_code=400, detail="Student faculty does not match course faculty")

    _ensure_student_schedule_has_no_conflict(db, student_id=student_id, course_id=course_id)

    obj = Enrollment(student_id=student_id, course_id=course_id)
    db.add(obj)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Student already enrolled in this course") from exc
    return {"enrolled": True}


@router.get(
    "/{course_id}/students",
    response_model=list[StudentRead],
    dependencies=[Depends(require_roles("FACULTY", "TEACHER", "ACADEMIA"))],
)
def list_enrolled_students(course_id: int, db: Session = Depends(get_role_scoped_db)):
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    students = (
        db.query(Student)
        .join(Enrollment, Enrollment.student_id == Student.id)
        .filter(Enrollment.course_id == course_id)
        .order_by(Student.student_number)
        .all()
    )
    return students
