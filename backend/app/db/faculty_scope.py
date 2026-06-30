from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, HTTPException

from app.core.security import get_current_user
from app.db.models import Faculty, User
from app.db.session import SessionLocal


@dataclass(frozen=True)
class FacultyScopeContext:
    faculty_id: int
    faculty_code: str


def get_optional_faculty_scope_context(
    current_user: User = Depends(get_current_user),
) -> FacultyScopeContext | None:
    role_names = {role.name for role in current_user.roles}
    if "SUPER_ADMIN" in role_names:
        return None

    if "ACADEMIA" in role_names or "HR" in role_names or "ADMISSIONS" in role_names:
        return None

    if "FACULTY" not in role_names and "FACULTY" not in role_names:
        return None

    if current_user.faculty_id is None:
        raise HTTPException(status_code=403, detail="Faculty-scoped user is missing faculty association")

    central_db = SessionLocal()
    try:
        faculty = central_db.query(Faculty).filter(Faculty.id == current_user.faculty_id).first()
    finally:
        central_db.close()

    if not faculty:
        raise HTTPException(status_code=404, detail="Faculty not found for current user")

    return FacultyScopeContext(
        faculty_id=faculty.id,
        faculty_code=faculty.code,
    )


def enforce_faculty_scope(target_faculty_id: int, faculty_scope: FacultyScopeContext | None) -> None:
    if faculty_scope is None:
        return

    if target_faculty_id != faculty_scope.faculty_id:
        raise HTTPException(status_code=403, detail="Faculty-scoped user cannot operate on another faculty")


def get_central_user_for_faculty(
    user_id: int,
    faculty_scope: FacultyScopeContext | None,
) -> User:
    central_db = SessionLocal()
    try:
        user = central_db.query(User).filter(User.id == user_id).first()
    finally:
        central_db.close()

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if faculty_scope is not None and user.faculty_id != faculty_scope.faculty_id:
        raise HTTPException(status_code=400, detail="User does not belong to the current faculty")

    return user