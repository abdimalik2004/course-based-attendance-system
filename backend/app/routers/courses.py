from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_
from sqlalchemy.exc import IntegrityError

from app.core.security import require_roles
from app.db.faculty_scope import enforce_faculty_scope, get_optional_faculty_scope_context
from app.db.models import (
    ClassBatch,
    Course,
    CourseAssignment,
    CourseSchedule,
    Enrollment,
    Student,
    Teacher,
    normalize_course_title,
)
from app.db.role_scoped import get_role_scoped_db
from app.schemas.course import (
    CourseAssignmentCreate,
    CourseCreate,
    CourseRead,
    CourseUpdate,
    PaginatedCourseRead,
)
from app.schemas.student import StudentRead
from app.utils.weekday_utils import weekdays_intersect


router = APIRouter(prefix="/courses", tags=["courses"])

_LEADING_ALPHA_RE = re.compile(r"^(?P<alpha>[A-Z]+)")


def _course_code_prefix(*, faculty_code: str, class_batch_name: str | None) -> str:
    if class_batch_name:
        match = _LEADING_ALPHA_RE.match(class_batch_name.strip().upper())
        if match:
            return match.group("alpha")

    prefix = faculty_code.strip().upper()
    if not prefix:
        raise HTTPException(status_code=400, detail="Faculty code is required for course code generation")
    return prefix


def _generate_course_code(db: Session, *, faculty_code: str, class_batch_name: str | None) -> str:
    prefix = _course_code_prefix(faculty_code=faculty_code, class_batch_name=class_batch_name)

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


@router.post("", response_model=CourseRead, dependencies=[Depends(require_roles("FACULTY_ADMIN"))])
def create_course(
    payload: CourseCreate,
    db: Session = Depends(get_role_scoped_db),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    class_batch = db.query(ClassBatch).filter(ClassBatch.id == payload.class_batch_id).first()
    if not class_batch:
        raise HTTPException(status_code=404, detail="Class batch not found")

    if faculty_scope is not None:
        enforce_faculty_scope(class_batch.faculty_id, faculty_scope)
        faculty_code = faculty_scope.faculty_code
    else:
        faculty_code = class_batch.faculty.code if class_batch.faculty else None

    if not faculty_code:
        raise HTTPException(status_code=400, detail="Faculty code is required for course code generation")

    _ensure_unique_course_title_in_faculty(
        db,
        faculty_id=class_batch.faculty_id,
        title=payload.title,
    )

    obj = Course(
        class_batch_id=payload.class_batch_id,
        faculty_id=class_batch.faculty_id,
        code=_generate_course_code(
            db,
            faculty_code=faculty_code,
            class_batch_name=class_batch.name,
        ),
        title=payload.title,
        normalized_title=normalize_course_title(payload.title),
    )
    db.add(obj)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Course code already exists in this class") from exc
    db.refresh(obj)
    return obj


@router.get("", response_model=PaginatedCourseRead, dependencies=[Depends(require_roles("FACULTY_ADMIN", "TEACHER", "ACADEMIA"))])
def list_courses(
    db: Session = Depends(get_role_scoped_db),
    skip: int = Query(default=0, ge=0, description="Number of rows to skip", examples=[0]),
    limit: int = Query(default=50, ge=1, le=200, description="Page size", examples=[20]),
    faculty_id: int | None = Query(default=None, description="Filter by faculty id", examples=[1]),
    department_id: int | None = Query(default=None, description="Filter by department id", examples=[1]),
    class_batch_id: int | None = Query(default=None, description="Filter by class batch id", examples=[1]),
    search: str | None = Query(default=None, description="Search by course code or title", examples=["CSC"]),
):
    query = db.query(Course)
    if faculty_id is not None or department_id is not None:
        query = query.join(ClassBatch, ClassBatch.id == Course.class_batch_id)
    if faculty_id is not None:
        query = query.filter(ClassBatch.faculty_id == faculty_id)
    if department_id is not None:
        query = query.filter(ClassBatch.department_id == department_id)
    if class_batch_id is not None:
        query = query.filter(Course.class_batch_id == class_batch_id)
    if search:
        pattern = f"%{search.strip()}%"
        query = query.filter(or_(Course.code.ilike(pattern), Course.title.ilike(pattern)))
    total = query.count()
    items = query.order_by(Course.code).offset(skip).limit(limit).all()
    return {"items": items, "total": total, "skip": skip, "limit": limit}


@router.put("/{course_id}", response_model=CourseRead, dependencies=[Depends(require_roles("FACULTY_ADMIN"))])
def update_course(
    course_id: int,
    payload: CourseUpdate,
    db: Session = Depends(get_role_scoped_db),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    obj = db.query(Course).filter(Course.id == course_id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Course not found")

    target_batch_id = payload.class_batch_id if payload.class_batch_id is not None else obj.class_batch_id
    target_batch = db.query(ClassBatch).filter(ClassBatch.id == target_batch_id).first()
    if not target_batch:
        raise HTTPException(status_code=404, detail="Class batch not found")
    if faculty_scope is not None:
        enforce_faculty_scope(target_batch.faculty_id, faculty_scope)

    next_title = payload.title if payload.title is not None else obj.title
    _ensure_unique_course_title_in_faculty(
        db,
        faculty_id=target_batch.faculty_id,
        title=next_title,
        exclude_course_id=obj.id,
    )

    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(obj, field, value)

    obj.faculty_id = target_batch.faculty_id
    obj.normalized_title = normalize_course_title(next_title)

    db.add(obj)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Update conflicts with existing course data") from exc
    db.refresh(obj)
    return obj


@router.delete("/{course_id}", dependencies=[Depends(require_roles("FACULTY_ADMIN"))])
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


@router.post("/assign-teacher", dependencies=[Depends(require_roles("FACULTY_ADMIN"))])
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

    class_batch = db.query(ClassBatch).filter(ClassBatch.id == course.class_batch_id).first()
    if not class_batch:
        raise HTTPException(status_code=404, detail="Class batch not found for course")
    if faculty_scope is not None:
        enforce_faculty_scope(class_batch.faculty_id, faculty_scope)

    if teacher.faculty_id != class_batch.faculty_id:
        raise HTTPException(
            status_code=400,
            detail="Teacher faculty does not match course faculty",
        )
    if teacher.department_id != class_batch.department_id:
        raise HTTPException(
            status_code=400,
            detail="Teacher department does not match course department",
        )

    assignment = CourseAssignment(**payload.model_dump())
    db.add(assignment)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Teacher already assigned to this course") from exc
    db.refresh(assignment)
    return {"id": assignment.id}


@router.post("/{course_id}/enroll/{student_id}", dependencies=[Depends(require_roles("FACULTY_ADMIN"))])
def enroll_student(
    course_id: int,
    student_id: int,
    db: Session = Depends(get_role_scoped_db),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    class_batch = db.query(ClassBatch).filter(ClassBatch.id == course.class_batch_id).first()
    if not class_batch:
        raise HTTPException(status_code=404, detail="Class batch not found for course")
    if faculty_scope is not None:
        enforce_faculty_scope(class_batch.faculty_id, faculty_scope)

    student = db.query(Student).filter(Student.id == student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    if student.class_batch_id != course.class_batch_id:
        raise HTTPException(status_code=400, detail="Student class does not match course class")

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
    dependencies=[Depends(require_roles("FACULTY_ADMIN", "TEACHER", "ACADEMIA"))],
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
