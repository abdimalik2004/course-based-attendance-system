from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_
from sqlalchemy.exc import IntegrityError

from app.core.security import require_roles
from app.db.models import ClassBatch
from app.db.session import get_db
from app.schemas.classbatch import (
    ClassBatchCreate,
    ClassBatchRead,
    ClassBatchUpdate,
    PaginatedClassBatchRead,
)


router = APIRouter(prefix="/classes", tags=["classes"])


@router.post("", response_model=ClassBatchRead, dependencies=[Depends(require_roles("FACULTY_ADMIN"))])
def create_class_batch(payload: ClassBatchCreate, db: Session = Depends(get_db)):
    obj = ClassBatch(**payload.model_dump())
    db.add(obj)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Class batch already exists for this faculty") from exc
    db.refresh(obj)
    return obj


@router.get("", response_model=PaginatedClassBatchRead, dependencies=[Depends(require_roles("FACULTY_ADMIN", "TEACHER", "ACADEMIA"))])
def list_class_batches(
    db: Session = Depends(get_db),
    skip: int = Query(default=0, ge=0, description="Number of rows to skip", examples=[0]),
    limit: int = Query(default=50, ge=1, le=200, description="Page size", examples=[20]),
    faculty_id: int | None = Query(default=None, description="Filter by faculty id", examples=[1]),
    search: str | None = Query(default=None, description="Search class name", examples=["CIS"]),
):
    query = db.query(ClassBatch)
    if faculty_id is not None:
        query = query.filter(ClassBatch.faculty_id == faculty_id)
    if search:
        pattern = f"%{search.strip()}%"
        query = query.filter(or_(ClassBatch.name.ilike(pattern)))
    total = query.count()
    items = query.order_by(ClassBatch.name).offset(skip).limit(limit).all()
    return {"items": items, "total": total, "skip": skip, "limit": limit}


@router.put("/{class_id}", response_model=ClassBatchRead, dependencies=[Depends(require_roles("FACULTY_ADMIN"))])
def update_class_batch(class_id: int, payload: ClassBatchUpdate, db: Session = Depends(get_db)):
    obj = db.query(ClassBatch).filter(ClassBatch.id == class_id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Class batch not found")

    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(obj, field, value)

    db.add(obj)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Update conflicts with existing class batch data") from exc
    db.refresh(obj)
    return obj


@router.delete("/{class_id}", dependencies=[Depends(require_roles("FACULTY_ADMIN"))])
def delete_class_batch(class_id: int, db: Session = Depends(get_db)):
    obj = db.query(ClassBatch).filter(ClassBatch.id == class_id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Class batch not found")
    db.delete(obj)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Cannot delete class batch due to related records") from exc
    return {"deleted": True, "class_id": class_id}
