from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_
from sqlalchemy.exc import IntegrityError

from app.core.security import get_current_user, require_roles
from app.db.faculty_scope import (
    enforce_faculty_scope,
    get_optional_faculty_scope_context,
)
from app.db.models import Teacher, User
from app.db.role_scoped import get_role_scoped_db
from app.schemas.teacher import TeacherCreate, TeacherRead, TeacherUpdate, PaginatedTeacherRead
from app.utils.activity_logger import log_activity
from app.utils.organization import ensure_department_belongs_to_faculty, get_department_or_404, get_faculty_or_404


router = APIRouter(prefix="/teachers", tags=["teachers"])


def _generate_teacher_number(db: Session, faculty_code: str) -> str:
    prefix = f"{faculty_code.strip().upper()}T"
    if not prefix or prefix == "T":
        raise HTTPException(status_code=400, detail="Faculty code is required for teacher number generation")

    existing = (
        db.query(Teacher.teacher_number)
        .filter(Teacher.teacher_number.ilike(f"{prefix}%"))
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


@router.post("", response_model=TeacherRead, dependencies=[Depends(require_roles("HR"))])
def create_teacher(
    payload: TeacherCreate,
    db: Session = Depends(get_role_scoped_db),
    current_user: User = Depends(get_current_user),
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

    teacher_number = _generate_teacher_number(db, faculty_code)

    obj = Teacher(
        teacher_number=teacher_number,
        full_name=payload.full_name,
        role=payload.role,
        status=payload.status,
        faculty_id=payload.faculty_id,
        department_id=payload.department_id,
    )
    db.add(obj)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Teacher number already exists") from exc
    db.refresh(obj)
    log_activity(
        action=f"Teacher Registered - {obj.full_name} ({obj.teacher_number})",
        user=current_user,
        db=db,
    )
    return obj


@router.get("", response_model=PaginatedTeacherRead, dependencies=[Depends(require_roles("HR", "FACULTY", "ACADEMIA"))])
def list_teachers(
    db: Session = Depends(get_role_scoped_db),
    skip: int = Query(default=0, ge=0, description="Number of rows to skip", examples=[0]),
    limit: int = Query(default=50, ge=1, le=200, description="Page size", examples=[20]),
    faculty_id: int | None = Query(default=None, description="Filter by faculty id", examples=[1]),
    department_id: int | None = Query(default=None, description="Filter by department id", examples=[1]),
    search: str | None = Query(default=None, description="Search by teacher number or full name", examples=["T-1001"]),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    query = db.query(Teacher)
    if faculty_scope is not None:
        if faculty_id is not None:
            enforce_faculty_scope(faculty_id, faculty_scope)
        else:
            faculty_id = faculty_scope.faculty_id
    if faculty_id is not None:
        query = query.filter(Teacher.faculty_id == faculty_id)
    if department_id is not None:
        query = query.filter(Teacher.department_id == department_id)
    if search:
        pattern = f"%{search.strip()}%"
        query = query.filter(or_(Teacher.full_name.ilike(pattern), Teacher.teacher_number.ilike(pattern)))
    total = query.count()
    items = query.order_by(Teacher.full_name).offset(skip).limit(limit).all()
    return {"items": items, "total": total, "skip": skip, "limit": limit}


@router.put("/{teacher_id}", response_model=TeacherRead, dependencies=[Depends(require_roles("HR"))])
def update_teacher(
    teacher_id: int,
    payload: TeacherUpdate,
    db: Session = Depends(get_role_scoped_db),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    obj = db.query(Teacher).filter(Teacher.id == teacher_id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Teacher not found")

    target_faculty_id = payload.faculty_id if payload.faculty_id is not None else obj.faculty_id
    target_department_id = payload.department_id if payload.department_id is not None else obj.department_id
    if faculty_scope is None:
        get_faculty_or_404(db, target_faculty_id)
    else:
        enforce_faculty_scope(target_faculty_id, faculty_scope)

    department = get_department_or_404(db, target_department_id)
    ensure_department_belongs_to_faculty(department, target_faculty_id)

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


@router.delete("/{teacher_id}", dependencies=[Depends(require_roles("HR"))])
def delete_teacher(teacher_id: int, db: Session = Depends(get_role_scoped_db)):
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
