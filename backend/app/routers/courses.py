from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_
from sqlalchemy.exc import IntegrityError

from app.core.security import require_roles
from app.db.models import ClassBatch, Course, CourseAssignment, Enrollment, Student, Teacher
from app.db.session import get_db
from app.schemas.course import (
    CourseAssignmentCreate,
    CourseCreate,
    CourseRead,
    CourseUpdate,
    PaginatedCourseRead,
)


router = APIRouter(prefix="/courses", tags=["courses"])


@router.post("", response_model=CourseRead, dependencies=[Depends(require_roles("FACULTY_ADMIN"))])
def create_course(payload: CourseCreate, db: Session = Depends(get_db)):
    obj = Course(**payload.model_dump())
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
    db: Session = Depends(get_db),
    skip: int = Query(default=0, ge=0, description="Number of rows to skip", examples=[0]),
    limit: int = Query(default=50, ge=1, le=200, description="Page size", examples=[20]),
    class_batch_id: int | None = Query(default=None, description="Filter by class batch id", examples=[1]),
    search: str | None = Query(default=None, description="Search by course code or title", examples=["CSC"]),
):
    query = db.query(Course)
    if class_batch_id is not None:
        query = query.filter(Course.class_batch_id == class_batch_id)
    if search:
        pattern = f"%{search.strip()}%"
        query = query.filter(or_(Course.code.ilike(pattern), Course.title.ilike(pattern)))
    total = query.count()
    items = query.order_by(Course.code).offset(skip).limit(limit).all()
    return {"items": items, "total": total, "skip": skip, "limit": limit}


@router.put("/{course_id}", response_model=CourseRead, dependencies=[Depends(require_roles("FACULTY_ADMIN"))])
def update_course(course_id: int, payload: CourseUpdate, db: Session = Depends(get_db)):
    obj = db.query(Course).filter(Course.id == course_id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Course not found")

    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(obj, field, value)

    db.add(obj)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Update conflicts with existing course data") from exc
    db.refresh(obj)
    return obj


@router.delete("/{course_id}", dependencies=[Depends(require_roles("FACULTY_ADMIN"))])
def delete_course(course_id: int, db: Session = Depends(get_db)):
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
def assign_teacher(payload: CourseAssignmentCreate, db: Session = Depends(get_db)):
    course = db.query(Course).filter(Course.id == payload.course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    teacher = db.query(Teacher).filter(Teacher.id == payload.teacher_id).first()
    if not teacher:
        raise HTTPException(status_code=404, detail="Teacher not found")

    class_batch = db.query(ClassBatch).filter(ClassBatch.id == course.class_batch_id).first()
    if not class_batch:
        raise HTTPException(status_code=404, detail="Class batch not found for course")

    if teacher.faculty_id != class_batch.faculty_id:
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
    return {"id": assignment.id}


@router.post("/{course_id}/enroll/{student_id}", dependencies=[Depends(require_roles("FACULTY_ADMIN"))])
def enroll_student(course_id: int, student_id: int, db: Session = Depends(get_db)):
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")

    student = db.query(Student).filter(Student.id == student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    if student.class_batch_id != course.class_batch_id:
        raise HTTPException(status_code=400, detail="Student class does not match course class")

    obj = Enrollment(student_id=student_id, course_id=course_id)
    db.add(obj)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Student already enrolled in this course") from exc
    return {"enrolled": True}
