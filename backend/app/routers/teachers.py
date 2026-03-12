from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_
from sqlalchemy.exc import IntegrityError

from app.core.security import require_roles
from app.db.models import Teacher
from app.db.session import get_db
from app.schemas.teacher import TeacherCreate, TeacherRead, TeacherUpdate, PaginatedTeacherRead


router = APIRouter(prefix="/teachers", tags=["teachers"])


@router.post("", response_model=TeacherRead, dependencies=[Depends(require_roles("FACULTY_ADMIN"))])
def create_teacher(payload: TeacherCreate, db: Session = Depends(get_db)):
    obj = Teacher(**payload.model_dump())
    db.add(obj)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Teacher number already exists") from exc
    db.refresh(obj)
    return obj


@router.get("", response_model=PaginatedTeacherRead, dependencies=[Depends(require_roles("FACULTY_ADMIN", "ACADEMIA"))])
def list_teachers(
    db: Session = Depends(get_db),
    skip: int = Query(default=0, ge=0, description="Number of rows to skip", examples=[0]),
    limit: int = Query(default=50, ge=1, le=200, description="Page size", examples=[20]),
    faculty_id: int | None = Query(default=None, description="Filter by faculty id", examples=[1]),
    search: str | None = Query(default=None, description="Search by teacher number or full name", examples=["T-1001"]),
):
    query = db.query(Teacher)
    if faculty_id is not None:
        query = query.filter(Teacher.faculty_id == faculty_id)
    if search:
        pattern = f"%{search.strip()}%"
        query = query.filter(or_(Teacher.full_name.ilike(pattern), Teacher.teacher_number.ilike(pattern)))
    total = query.count()
    items = query.order_by(Teacher.full_name).offset(skip).limit(limit).all()
    return {"items": items, "total": total, "skip": skip, "limit": limit}


@router.put("/{teacher_id}", response_model=TeacherRead, dependencies=[Depends(require_roles("FACULTY_ADMIN"))])
def update_teacher(teacher_id: int, payload: TeacherUpdate, db: Session = Depends(get_db)):
    obj = db.query(Teacher).filter(Teacher.id == teacher_id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Teacher not found")

    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(obj, field, value)

    db.add(obj)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Update conflicts with existing teacher data") from exc
    db.refresh(obj)
    return obj


@router.delete("/{teacher_id}", dependencies=[Depends(require_roles("FACULTY_ADMIN"))])
def delete_teacher(teacher_id: int, db: Session = Depends(get_db)):
    obj = db.query(Teacher).filter(Teacher.id == teacher_id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Teacher not found")
    db.delete(obj)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Cannot delete teacher due to related records") from exc
    return {"deleted": True, "teacher_id": teacher_id}
