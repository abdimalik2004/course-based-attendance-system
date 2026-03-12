from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_
from sqlalchemy.exc import IntegrityError

from app.core.security import require_roles
from app.db.models import Student
from app.db.session import get_db
from app.schemas.student import StudentCreate, StudentRead, StudentUpdate, PaginatedStudentRead


router = APIRouter(prefix="/students", tags=["students"])


@router.post("", response_model=StudentRead, dependencies=[Depends(require_roles("FACULTY_ADMIN"))])
def create_student(payload: StudentCreate, db: Session = Depends(get_db)):
    obj = Student(**payload.model_dump())
    db.add(obj)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Student number already exists") from exc
    db.refresh(obj)
    return obj


@router.get("", response_model=PaginatedStudentRead, dependencies=[Depends(require_roles("FACULTY_ADMIN", "TEACHER"))])
def list_students(
    db: Session = Depends(get_db),
    skip: int = Query(default=0, ge=0, description="Number of rows to skip", examples=[0]),
    limit: int = Query(default=50, ge=1, le=200, description="Page size", examples=[20]),
    faculty_id: int | None = Query(default=None, description="Filter by faculty id", examples=[1]),
    class_batch_id: int | None = Query(default=None, description="Filter by class batch id", examples=[1]),
    search: str | None = Query(default=None, description="Search by student number or full name", examples=["2201"]),
):
    query = db.query(Student)
    if faculty_id is not None:
        query = query.filter(Student.faculty_id == faculty_id)
    if class_batch_id is not None:
        query = query.filter(Student.class_batch_id == class_batch_id)
    if search:
        pattern = f"%{search.strip()}%"
        query = query.filter(or_(Student.full_name.ilike(pattern), Student.student_number.ilike(pattern)))
    total = query.count()
    items = query.order_by(Student.full_name).offset(skip).limit(limit).all()
    return {"items": items, "total": total, "skip": skip, "limit": limit}


@router.put("/{student_id}", response_model=StudentRead, dependencies=[Depends(require_roles("FACULTY_ADMIN"))])
def update_student(student_id: int, payload: StudentUpdate, db: Session = Depends(get_db)):
    obj = db.query(Student).filter(Student.id == student_id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Student not found")

    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(obj, field, value)

    db.add(obj)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Update conflicts with existing student data") from exc
    db.refresh(obj)
    return obj


@router.delete("/{student_id}", dependencies=[Depends(require_roles("FACULTY_ADMIN"))])
def delete_student(student_id: int, db: Session = Depends(get_db)):
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
