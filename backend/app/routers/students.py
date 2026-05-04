from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_
from sqlalchemy.exc import IntegrityError

from app.core.security import require_roles
from app.db.faculty_scope import enforce_faculty_scope, get_optional_faculty_scope_context
from app.db.models import Student
from app.db.role_scoped import get_role_scoped_db
from app.services.enrollment_service import auto_enroll_student_in_matching_courses
from app.schemas.student import StudentCreate, StudentRead, StudentUpdate, PaginatedStudentRead
from app.utils.organization import (
    ensure_class_batch_matches_faculty_and_department,
    ensure_department_belongs_to_faculty,
    get_class_batch_or_404,
    get_latest_class_batch_for_faculty_and_department_or_404,
    get_department_or_404,
    get_faculty_or_404,
)
from app.utils.student_numbering import next_available_student_number


router = APIRouter(prefix="/students", tags=["students"])


@router.post("", response_model=StudentRead, dependencies=[Depends(require_roles("ADMISSIONS"))])
def create_student(
    payload: StudentCreate,
    db: Session = Depends(get_role_scoped_db),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    if faculty_scope is None:
        faculty = get_faculty_or_404(db, payload.faculty_id)
        faculty_code = faculty.code
    else:
        enforce_faculty_scope(payload.faculty_id, faculty_scope)
        faculty_code = faculty_scope.faculty_code

    department = get_department_or_404(db, payload.department_id)
    ensure_department_belongs_to_faculty(department, payload.faculty_id)
    class_batch = get_latest_class_batch_for_faculty_and_department_or_404(
        db,
        faculty_id=payload.faculty_id,
        department_id=department.id,
    )
    ensure_class_batch_matches_faculty_and_department(
        class_batch,
        faculty_id=payload.faculty_id,
        department_id=department.id,
    )

    student_number = next_available_student_number(db, faculty_code, class_batch.year, class_batch.name)
    embedding_ref = student_number

    obj = Student(
        student_number=student_number,
        full_name=payload.full_name,
        faculty_id=payload.faculty_id,
        department_id=payload.department_id,
        class_batch_id=class_batch.id,
        embedding_ref=embedding_ref,
    )
    db.add(obj)
    try:
        db.flush()
        auto_enroll_student_in_matching_courses(db, obj)
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Student number already exists") from exc
    db.refresh(obj)
    return obj


@router.get("", response_model=PaginatedStudentRead, dependencies=[Depends(require_roles("ADMISSIONS", "FACULTY", "TEACHER", "ACADEMIA"))])
def list_students(
    db: Session = Depends(get_role_scoped_db),
    skip: int = Query(default=0, ge=0, description="Number of rows to skip", examples=[0]),
    limit: int = Query(default=50, ge=1, le=200, description="Page size", examples=[20]),
    faculty_id: int | None = Query(default=None, description="Filter by faculty id", examples=[1]),
    department_id: int | None = Query(default=None, description="Filter by department id", examples=[1]),
    class_batch_id: int | None = Query(default=None, description="Filter by class batch id", examples=[1]),
    search: str | None = Query(default=None, description="Search by student number or full name", examples=["2201"]),
):
    query = db.query(Student)
    if faculty_id is not None:
        query = query.filter(Student.faculty_id == faculty_id)
    if department_id is not None:
        query = query.filter(Student.department_id == department_id)
    if class_batch_id is not None:
        query = query.filter(Student.class_batch_id == class_batch_id)
    if search:
        pattern = f"%{search.strip()}%"
        query = query.filter(or_(Student.full_name.ilike(pattern), Student.student_number.ilike(pattern)))
    total = query.count()
    items = query.order_by(Student.full_name).offset(skip).limit(limit).all()
    return {"items": items, "total": total, "skip": skip, "limit": limit}


@router.put("/{student_id}", response_model=StudentRead, dependencies=[Depends(require_roles("ADMISSIONS"))])
def update_student(
    student_id: int,
    payload: StudentUpdate,
    db: Session = Depends(get_role_scoped_db),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    obj = db.query(Student).filter(Student.id == student_id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Student not found")

    target_class_batch = get_class_batch_or_404(db, payload.class_batch_id if payload.class_batch_id is not None else obj.class_batch_id)
    target_faculty_id = payload.faculty_id if payload.faculty_id is not None else target_class_batch.faculty_id
    target_department_id = payload.department_id if payload.department_id is not None else target_class_batch.department_id

    if faculty_scope is None:
        get_faculty_or_404(db, target_faculty_id)
    else:
        enforce_faculty_scope(target_faculty_id, faculty_scope)

    department = get_department_or_404(db, target_department_id)
    ensure_department_belongs_to_faculty(department, target_faculty_id)
    ensure_class_batch_matches_faculty_and_department(
        target_class_batch,
        faculty_id=target_faculty_id,
        department_id=target_department_id,
    )

    payload_data = payload.model_dump(exclude_unset=True)
    for field in ("faculty_id", "department_id", "class_batch_id"):
        payload_data.pop(field, None)

    for field, value in payload_data.items():
        setattr(obj, field, value)

    obj.faculty_id = target_class_batch.faculty_id
    obj.department_id = target_class_batch.department_id
    obj.class_batch_id = target_class_batch.id

    db.add(obj)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Update conflicts with existing student data") from exc
    db.refresh(obj)
    return obj


@router.delete("/{student_id}", dependencies=[Depends(require_roles("ADMISSIONS"))])
def delete_student(student_id: int, db: Session = Depends(get_role_scoped_db)):
    obj = db.query(Student).filter(Student.id == student_id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Student not found")
    db.delete(obj)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Cannot delete student due to related records") from exc
    return {"deleted": True, "student_id": student_id}
