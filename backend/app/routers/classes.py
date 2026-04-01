from __future__ import annotations

import logging
import re

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.security import require_roles
from app.db.faculty_scope import FacultyScopeContext, enforce_faculty_scope, get_optional_faculty_scope_context
from app.db.models import ClassBatch
from app.db.role_scoped import get_role_scoped_db
from app.schemas.classbatch import (
    ClassBatchCreate,
    ClassBatchRead,
    ClassBatchUpdate,
    PaginatedClassBatchRead,
)
from app.utils.db_conflicts import classify_integrity_error, integrity_error_mentions
from app.utils.organization import (
    ensure_faculty_row_available,
    ensure_department_belongs_to_faculty,
    get_department_or_404,
    get_faculty_or_404,
)


router = APIRouter(prefix="/classes", tags=["classes"])
logger = logging.getLogger(__name__)
_CLASS_NAME_RE = re.compile(r"^(?P<prefix>[A-Z]+)(?P<number>\d+)$")


def _resolve_create_faculty_id(
    *,
    payload_faculty_id: int | None,
    faculty_scope: FacultyScopeContext | None,
) -> int:
    if faculty_scope is not None:
        if payload_faculty_id is not None:
            enforce_faculty_scope(payload_faculty_id, faculty_scope)
            return payload_faculty_id
        return faculty_scope.faculty_id

    if payload_faculty_id is None:
        raise HTTPException(status_code=400, detail="faculty_id is required")
    return payload_faculty_id


def _class_batch_duplicate_exists(
    db: Session,
    *,
    department_id: int,
    name: str,
    exclude_id: int | None = None,
) -> bool:
    normalized_name = name.strip().lower()
    query = db.query(ClassBatch.id).filter(
        ClassBatch.department_id == department_id,
        func.lower(func.trim(ClassBatch.name)) == normalized_name,
    )
    if exclude_id is not None:
        query = query.filter(ClassBatch.id != exclude_id)
    return db.query(query.exists()).scalar()


def _class_batch_integrity_detail(exc: IntegrityError) -> str:
    error_kind = classify_integrity_error(exc)
    if error_kind == "foreign_key":
        return "Class batch references missing faculty/department metadata in tenant DB"
    if error_kind == "duplicate" and integrity_error_mentions(
        exc,
        "uq_class_batch_department_name",
        "uq_class_batch_faculty_name",
        "class_batches.department_id, class_batches.name",
        "class_batches.faculty_id, class_batches.name",
    ):
        return "Class batch already exists for this department"
    return "Class batch already exists for this department"


def _generate_class_batch_name(
    db: Session,
    *,
    faculty_code: str,
    department_id: int,
) -> str:
    existing_names = (
        db.query(ClassBatch.name)
        .filter(ClassBatch.department_id == department_id)
        .all()
    )

    best_prefix = faculty_code.strip().upper()
    max_seq = 0
    width = 3

    for (raw_name,) in existing_names:
        if not raw_name:
            continue
        normalized = raw_name.strip().upper()
        match = _CLASS_NAME_RE.match(normalized)
        if not match:
            continue
        prefix = match.group("prefix")
        number_text = match.group("number")
        number = int(number_text)
        if number > max_seq:
            max_seq = number
            best_prefix = prefix
            width = max(len(number_text), 3)

    return f"{best_prefix}{max_seq + 1:0{width}d}"


@router.post("", response_model=ClassBatchRead, dependencies=[Depends(require_roles("FACULTY_ADMIN"))])
def create_class_batch(
    payload: ClassBatchCreate,
    db: Session = Depends(get_role_scoped_db),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    faculty_id = _resolve_create_faculty_id(
        payload_faculty_id=payload.faculty_id,
        faculty_scope=faculty_scope,
    )
    if faculty_scope is None:
        get_faculty_or_404(db, faculty_id)
    else:
        ensure_faculty_row_available(
            db,
            faculty_id=faculty_id,
            faculty_scope=faculty_scope,
        )

    department = get_department_or_404(db, payload.department_id)
    ensure_department_belongs_to_faculty(department, faculty_id)

    faculty_code_for_generation = (
        faculty_scope.faculty_code
        if faculty_scope is not None
        else (department.faculty.code if department.faculty else None)
    )
    if not faculty_code_for_generation:
        raise HTTPException(status_code=400, detail="Faculty code is required for class name generation")

    class_name = payload.name or _generate_class_batch_name(
        db,
        faculty_code=faculty_code_for_generation,
        department_id=payload.department_id,
    )

    if _class_batch_duplicate_exists(
        db,
        department_id=payload.department_id,
        name=class_name,
    ):
        raise HTTPException(status_code=409, detail="Class batch already exists for this department")

    obj = ClassBatch(
        faculty_id=faculty_id,
        department_id=payload.department_id,
        name=class_name,
        year=payload.year,
    )
    db.add(obj)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        detail = _class_batch_integrity_detail(exc)
        if classify_integrity_error(exc) == "duplicate" and _class_batch_duplicate_exists(
            db,
            department_id=payload.department_id,
            name=class_name,
        ):
            logger.warning(
                "Class batch duplicate blocked faculty_id=%s department_id=%s name=%s",
                faculty_id,
                payload.department_id,
                class_name,
            )
            raise HTTPException(status_code=409, detail=detail) from exc
        if classify_integrity_error(exc) == "foreign_key":
            raise HTTPException(status_code=400, detail=detail) from exc
        logger.exception("Unexpected integrity error while creating class batch")
        raise HTTPException(status_code=400, detail="Class batch could not be created due to invalid data") from exc
    db.refresh(obj)
    return obj


@router.get("", response_model=PaginatedClassBatchRead, dependencies=[Depends(require_roles("FACULTY_ADMIN", "TEACHER", "ACADEMIA"))])
def list_class_batches(
    db: Session = Depends(get_role_scoped_db),
    skip: int = Query(default=0, ge=0, description="Number of rows to skip", examples=[0]),
    limit: int = Query(default=50, ge=1, le=200, description="Page size", examples=[20]),
    faculty_id: int | None = Query(default=None, description="Filter by faculty id", examples=[1]),
    department_id: int | None = Query(default=None, description="Filter by department id", examples=[1]),
    search: str | None = Query(default=None, description="Search class name", examples=["CIS"]),
):
    query = db.query(ClassBatch)
    if faculty_id is not None:
        query = query.filter(ClassBatch.faculty_id == faculty_id)
    if department_id is not None:
        query = query.filter(ClassBatch.department_id == department_id)
    if search:
        pattern = f"%{search.strip()}%"
        query = query.filter(or_(ClassBatch.name.ilike(pattern)))
    total = query.count()
    items = query.order_by(ClassBatch.name).offset(skip).limit(limit).all()
    return {"items": items, "total": total, "skip": skip, "limit": limit}


@router.put("/{class_id}", response_model=ClassBatchRead, dependencies=[Depends(require_roles("FACULTY_ADMIN"))])
def update_class_batch(
    class_id: int,
    payload: ClassBatchUpdate,
    db: Session = Depends(get_role_scoped_db),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    obj = db.query(ClassBatch).filter(ClassBatch.id == class_id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Class batch not found")

    target_faculty_id = payload.faculty_id if payload.faculty_id is not None else obj.faculty_id
    target_department_id = payload.department_id if payload.department_id is not None else obj.department_id
    if faculty_scope is None:
        get_faculty_or_404(db, target_faculty_id)
    else:
        enforce_faculty_scope(target_faculty_id, faculty_scope)
        ensure_faculty_row_available(
            db,
            faculty_id=target_faculty_id,
            faculty_scope=faculty_scope,
        )

    department = get_department_or_404(db, target_department_id)
    ensure_department_belongs_to_faculty(department, target_faculty_id)

    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(obj, field, value)

    if _class_batch_duplicate_exists(
        db,
        department_id=obj.department_id,
        name=obj.name,
        exclude_id=obj.id,
    ):
        raise HTTPException(status_code=409, detail="Class batch already exists for this department")

    db.add(obj)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        error_kind = classify_integrity_error(exc)
        if error_kind == "duplicate":
            raise HTTPException(status_code=409, detail="Update conflicts with existing class batch data") from exc
        if error_kind == "foreign_key":
            raise HTTPException(status_code=400, detail="Class batch update references invalid faculty/department") from exc
        logger.exception("Unexpected integrity error while updating class batch id=%s", class_id)
        raise HTTPException(status_code=400, detail="Class batch update failed due to invalid data") from exc
    db.refresh(obj)
    return obj


@router.delete("/{class_id}", dependencies=[Depends(require_roles("FACULTY_ADMIN"))])
def delete_class_batch(class_id: int, db: Session = Depends(get_role_scoped_db)):
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
