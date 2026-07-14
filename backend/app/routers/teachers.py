from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import or_
from sqlalchemy.exc import IntegrityError

from app.core.security import get_current_user, require_roles
from app.db.faculty_scope import (
    enforce_faculty_scope,
    get_optional_faculty_scope_context,
)
from app.db.models import Department, Faculty, Role, Teacher, User
from app.db.role_scoped import get_role_scoped_db
from app.db.session import get_db
from app.schemas.teacher import LinkUserPayload, TeacherCreate, TeacherRead, TeacherUpdate, PaginatedTeacherRead
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
        phone=payload.phone,
        email=payload.email,
        hire_date=payload.hire_date,
    )
    db.add(obj)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Teacher number already exists") from exc
    db.refresh(obj)
    # Re-fetch with user relationship so linked_username is populated in response
    obj = db.query(Teacher).options(joinedload(Teacher.user)).filter(Teacher.id == obj.id).one()
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
    items = (
        query.options(joinedload(Teacher.user))
        .order_by(Teacher.full_name)
        .offset(skip)
        .limit(limit)
        .all()
    )
    return {"items": items, "total": total, "skip": skip, "limit": limit}


@router.get("/me", response_model=TeacherRead, dependencies=[Depends(require_roles("TEACHER"))])
def get_my_teacher_profile(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return the Teacher record linked to the authenticated teacher's login account.

    Intended for the teacher role to read their own profile (name, department,
    faculty, status, contact details) without needing an admin-level token.
    Raises 404 if the user account has not yet been linked to a Teacher record.
    """
    teacher = (
        db.query(Teacher)
        .options(joinedload(Teacher.user))
        .filter(Teacher.user_id == current_user.id)
        .first()
    )
    if not teacher:
        raise HTTPException(
            status_code=404,
            detail=(
                "No teacher profile is linked to your account. "
                "Contact HR to link your login before using this endpoint."
            ),
        )
    faculty = db.query(Faculty).filter(Faculty.id == teacher.faculty_id).first() if teacher.faculty_id else None
    department = db.query(Department).filter(Department.id == teacher.department_id).first() if teacher.department_id else None
    return {
        "id": teacher.id,
        "teacher_number": teacher.teacher_number,
        "full_name": teacher.full_name,
        "role": teacher.role,
        "status": teacher.status,
        "faculty_id": teacher.faculty_id,
        "department_id": teacher.department_id,
        "user_id": teacher.user_id,
        "linked_username": teacher.user.username if teacher.user else None,
        "phone": teacher.phone,
        "email": teacher.email,
        "hire_date": teacher.hire_date,
        "faculty_name": faculty.name if faculty else None,
        "department_name": department.name if department else None,
    }


@router.put("/{teacher_id}", response_model=TeacherRead, dependencies=[Depends(require_roles("HR"))])
def update_teacher(
    teacher_id: int,
    payload: TeacherUpdate,
    db: Session = Depends(get_role_scoped_db),
    current_user: User = Depends(get_current_user),
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

    old_status = obj.status
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(obj, field, value)

    db.add(obj)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Update conflicts with existing teacher data") from exc
    db.refresh(obj)
    # Re-fetch with user relationship so linked_username is populated in response
    obj = db.query(Teacher).options(joinedload(Teacher.user)).filter(Teacher.id == obj.id).one()

    # Log status changes explicitly so HR activity panel shows meaningful events
    if payload.status is not None and payload.status != old_status:
        log_activity(
            action=f"Teacher Status Changed - {obj.full_name} ({obj.teacher_number}): {old_status.value} → {obj.status.value}",
            user=current_user,
            db=db,
        )
    else:
        log_activity(
            action=f"Teacher Updated - {obj.full_name} ({obj.teacher_number})",
            user=current_user,
            db=db,
        )
    return obj


@router.delete("/{teacher_id}", dependencies=[Depends(require_roles("HR"))])
def delete_teacher(
    teacher_id: int,
    db: Session = Depends(get_role_scoped_db),
    current_user: User = Depends(get_current_user),
):
    obj = db.query(Teacher).filter(Teacher.id == teacher_id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Teacher not found")
    teacher_label = f"{obj.full_name} ({obj.teacher_number})"
    db.delete(obj)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Cannot delete teacher due to related records") from exc
    log_activity(
        action=f"Teacher Deleted - {teacher_label}",
        user=current_user,
        db=db,
    )
    return {"deleted": True, "teacher_id": teacher_id}


@router.get(
    "/available-users",
    dependencies=[Depends(require_roles("HR", "SUPER_ADMIN"))],
)
def list_available_users(db: Session = Depends(get_db)):
    """Return users with the TEACHER role that are not yet linked to any teacher.

    Used by the "Link Login Account" modal so HR can pick an account to link.
    If a `teacher_id` is also currently linked to the user being edited, that
    user is excluded from the results (they're already taken).
    """
    teacher_role = db.query(Role).filter(Role.name == "TEACHER").first()
    if not teacher_role:
        return []

    # IDs already claimed by a teacher
    linked_ids: set[int] = {
        row[0]
        for row in db.query(Teacher.user_id).filter(Teacher.user_id.isnot(None)).all()
    }

    users = (
        db.query(User)
        .join(User.roles)
        .filter(Role.name == "TEACHER", User.is_active.is_(True))
        .all()
    )

    return [
        {"id": u.id, "username": u.username, "email": u.email}
        for u in users
        if u.id not in linked_ids
    ]


@router.patch("/{teacher_id}/link-user", response_model=TeacherRead, dependencies=[Depends(require_roles("HR", "SUPER_ADMIN"))])
def link_user_to_teacher(
    teacher_id: int,
    payload: LinkUserPayload,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Link or unlink a user account from a teacher.

    - Pass `user_id` to link a specific user account.
    - Pass `user_id: null` to unlink the current account.
    """
    obj = db.query(Teacher).options(joinedload(Teacher.user)).filter(Teacher.id == teacher_id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Teacher not found")

    if payload.user_id is not None:
        # Validate target user exists
        target_user = db.query(User).filter(User.id == payload.user_id).first()
        if not target_user:
            raise HTTPException(status_code=404, detail="User not found")
        # Ensure user isn't already linked to a *different* teacher
        existing = (
            db.query(Teacher)
            .filter(Teacher.user_id == payload.user_id, Teacher.id != teacher_id)
            .first()
        )
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"This user account is already linked to teacher '{existing.full_name}'",
            )

    old_username = obj.linked_username
    obj.user_id = payload.user_id
    db.add(obj)
    db.commit()
    # Re-fetch with user relationship for correct linked_username in both response and log
    obj = db.query(Teacher).options(joinedload(Teacher.user)).filter(Teacher.id == teacher_id).one()

    if payload.user_id is not None:
        log_activity(
            action=f"Teacher Account Linked - {obj.full_name} ({obj.teacher_number}) → @{obj.linked_username}",
            user=current_user,
            db=db,
        )
    else:
        log_activity(
            action=f"Teacher Account Unlinked - {obj.full_name} ({obj.teacher_number}) (was @{old_username})",
            user=current_user,
            db=db,
        )

    return obj
