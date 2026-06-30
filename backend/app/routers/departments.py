from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, or_
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.security import get_current_user, require_roles
from app.db.faculty_scope import FacultyScopeContext, enforce_faculty_scope, get_optional_faculty_scope_context
from app.db.models import Department, User
from app.db.role_scoped import get_role_scoped_db
from app.schemas.department import DepartmentCreate, DepartmentRead, DepartmentUpdate, PaginatedDepartmentRead
from app.utils.db_conflicts import classify_integrity_error, integrity_error_mentions
from app.utils.organization import ensure_faculty_row_available, get_faculty_or_404
from app.utils.activity_logger import log_activity
from app.services.notification_service import notify_faculty_admins, NotificationType


router = APIRouter(prefix="/departments", tags=["departments"])
logger = logging.getLogger(__name__)


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


def _department_duplicate_exists(
    db: Session,
    *,
    faculty_id: int,
    name: str,
    code: str,
    exclude_id: int | None = None,
) -> bool:
    normalized_name = name.strip().lower()
    normalized_code = code.strip().lower()
    query = db.query(Department.id).filter(
        Department.faculty_id == faculty_id,
        or_(
            func.lower(func.trim(Department.name)) == normalized_name,
            func.lower(func.trim(Department.code)) == normalized_code,
        ),
    )
    if exclude_id is not None:
        query = query.filter(Department.id != exclude_id)
    return db.query(query.exists()).scalar()


@router.post("", response_model=DepartmentRead, dependencies=[Depends(require_roles("ACADEMIA"))])
def create_department(
    payload: DepartmentCreate,
    db: Session = Depends(get_role_scoped_db),
    faculty_scope = Depends(get_optional_faculty_scope_context),
    current_user: "User" = Depends(get_current_user),
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

    if _department_duplicate_exists(
        db,
        faculty_id=faculty_id,
        name=payload.name,
        code=payload.code,
    ):
        raise HTTPException(status_code=409, detail="Department with same name/code already exists in this faculty")

    obj = Department(
        faculty_id=faculty_id,
        name=payload.name,
        code=payload.code,
    )
    db.add(obj)
    try:
        db.commit()
        log_activity(
            action=f"Department Created - {payload.name}",
            user=current_user,
            db=db,
        )
    except IntegrityError as exc:
        db.rollback()
        error_kind = classify_integrity_error(exc)
        if error_kind == "duplicate" and (
            integrity_error_mentions(
                exc,
                "uq_department_faculty_name",
                "uq_department_faculty_code",
                "departments.faculty_id, departments.name",
                "departments.faculty_id, departments.code",
            )
            or _department_duplicate_exists(
                db,
                faculty_id=faculty_id,
                name=payload.name,
                code=payload.code,
            )
        ):
            logger.warning(
                "Department duplicate blocked faculty_id=%s name=%s code=%s",
                faculty_id,
                payload.name,
                payload.code,
            )
            raise HTTPException(
                status_code=409,
                detail="Department with same name/code already exists in this faculty",
            ) from exc
        if error_kind == "foreign_key":
            raise HTTPException(status_code=400, detail="Department references an invalid faculty") from exc
        logger.exception("Unexpected integrity error while creating department")
        raise HTTPException(status_code=400, detail="Department could not be created due to invalid data") from exc
    db.refresh(obj)
    notify_faculty_admins(
        db, obj.faculty_id,
        title="New Department Created",
        message=f"Department '{obj.name}' has been created in your faculty.",
        notif_type=NotificationType.SUCCESS,
        link="/faculty/departments",
    )
    return obj


@router.get(
    "",
    response_model=PaginatedDepartmentRead,
    dependencies=[Depends(require_roles("SUPER_ADMIN", "ADMIN", "ACADEMIA", "FACULTY", "TEACHER", "HR", "ADMISSION", "ADMISSIONS"))],
)
def list_departments(
    db: Session = Depends(get_role_scoped_db),
    skip: int = Query(default=0, ge=0, description="Number of rows to skip", examples=[0]),
    limit: int = Query(default=50, ge=1, le=200, description="Page size", examples=[20]),
    faculty_id: int | None = Query(default=None, description="Filter by faculty id", examples=[1]),
    search: str | None = Query(default=None, description="Search by department name or code", examples=["IT"]),
):
    query = db.query(Department)
    if faculty_id is not None:
        query = query.filter(Department.faculty_id == faculty_id)
    if search:
        pattern = f"%{search.strip()}%"
        query = query.filter(or_(Department.name.ilike(pattern), Department.code.ilike(pattern)))
    total = query.count()
    items = query.order_by(Department.name).offset(skip).limit(limit).all()
    return {"items": items, "total": total, "skip": skip, "limit": limit}


@router.put("/{department_id}", response_model=DepartmentRead, dependencies=[Depends(require_roles("ACADEMIA"))])
def update_department(
    department_id: int,
    payload: DepartmentUpdate,
    db: Session = Depends(get_role_scoped_db),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    obj = db.query(Department).filter(Department.id == department_id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Department not found")

    target_faculty_id = payload.faculty_id if payload.faculty_id is not None else obj.faculty_id
    if faculty_scope is None:
        get_faculty_or_404(db, target_faculty_id)
    else:
        enforce_faculty_scope(target_faculty_id, faculty_scope)
        ensure_faculty_row_available(
            db,
            faculty_id=target_faculty_id,
            faculty_scope=faculty_scope,
        )

    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(obj, field, value)

    if _department_duplicate_exists(
        db,
        faculty_id=obj.faculty_id,
        name=obj.name,
        code=obj.code,
        exclude_id=obj.id,
    ):
        raise HTTPException(status_code=409, detail="Department with same name/code already exists in this faculty")

    db.add(obj)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        error_kind = classify_integrity_error(exc)
        if error_kind == "duplicate":
            raise HTTPException(status_code=409, detail="Update conflicts with existing department data") from exc
        if error_kind == "foreign_key":
            raise HTTPException(status_code=400, detail="Department update references invalid faculty") from exc
        logger.exception("Unexpected integrity error while updating department id=%s", department_id)
        raise HTTPException(status_code=400, detail="Department update failed due to invalid data") from exc
    db.refresh(obj)
    return obj


@router.delete("/{department_id}", dependencies=[Depends(require_roles("ACADEMIA"))])
def delete_department(department_id: int, db: Session = Depends(get_role_scoped_db)):
    obj = db.query(Department).filter(Department.id == department_id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Department not found")
    db.delete(obj)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Cannot delete department due to related records") from exc
    return {"deleted": True, "department_id": department_id}