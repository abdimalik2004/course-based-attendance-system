from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.faculty_scope import FacultyScopeContext
from app.db.models import ClassBatch, Department, Faculty
from app.db.session import SessionLocal


def get_faculty_or_404(db: Session, faculty_id: int) -> Faculty:
    faculty = db.query(Faculty).filter(Faculty.id == faculty_id).first()
    if not faculty:
        raise HTTPException(status_code=404, detail="Faculty not found")
    return faculty


def get_department_or_404(db: Session, department_id: int) -> Department:
    department = db.query(Department).filter(Department.id == department_id).first()
    if not department:
        raise HTTPException(status_code=404, detail="Department not found")
    return department


def get_class_batch_or_404(db: Session, class_batch_id: int) -> ClassBatch:
    class_batch = db.query(ClassBatch).filter(ClassBatch.id == class_batch_id).first()
    if not class_batch:
        raise HTTPException(status_code=404, detail="Class batch not found")
    return class_batch


def ensure_department_belongs_to_faculty(department: Department, faculty_id: int) -> None:
    if department.faculty_id != faculty_id:
        raise HTTPException(status_code=400, detail="Department does not belong to faculty")


def ensure_class_batch_matches_faculty_and_department(
    class_batch: ClassBatch,
    *,
    faculty_id: int,
    department_id: int,
) -> None:
    if class_batch.faculty_id != faculty_id:
        raise HTTPException(status_code=400, detail="Class batch does not belong to faculty")
    if class_batch.department_id != department_id:
        raise HTTPException(status_code=400, detail="Class batch does not belong to department")


def ensure_faculty_row_available(
    db: Session,
    *,
    faculty_id: int,
    faculty_scope: FacultyScopeContext | None,
) -> Faculty:
    faculty = db.query(Faculty).filter(Faculty.id == faculty_id).first()
    if faculty is not None:
        return faculty

    if faculty_scope is None:
        raise HTTPException(status_code=404, detail="Faculty not found")

    central_db = SessionLocal()
    try:
        central_faculty = central_db.query(Faculty).filter(Faculty.id == faculty_id).first()
    finally:
        central_db.close()

    if central_faculty is None:
        raise HTTPException(status_code=404, detail="Faculty not found")

    materialized = Faculty(
        id=central_faculty.id,
        name=central_faculty.name,
        code=central_faculty.code,
    )
    db.add(materialized)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        existing = db.query(Faculty).filter(Faculty.id == faculty_id).first()
        if existing is not None:
            return existing
        raise HTTPException(status_code=400, detail="Faculty metadata row is missing in database")
    return materialized