from __future__ import annotations

from collections.abc import Iterable

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.core.security import get_password_hash
from app.db.models import Role, User


def _normalize_role_names(role_names: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for role_name in role_names:
        candidate = role_name.strip().upper()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        normalized.append(candidate)
    if not normalized:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one valid role is required")
    return normalized


def _load_roles(db: Session, role_names: list[str]) -> list[Role]:
    roles = db.query(Role).filter(Role.name.in_(role_names)).all()
    if len(roles) != len(role_names):
        found = {role.name for role in roles}
        missing = [role_name for role_name in role_names if role_name not in found]
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail={"message": "Invalid role names", "missing": missing})
    return roles


def create_user(
    db: Session,
    *,
    email: str,
    username: str,
    password: str,
    role_names: Iterable[str],
) -> User:
    normalized_roles = _normalize_role_names(role_names)

    existing_username = db.query(User).filter(User.username == username).first()
    if existing_username is not None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists")

    existing_email = db.query(User).filter(User.email == email).first()
    if existing_email is not None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already exists")

    user = User(
        email=email,
        username=username,
        hashed_password=get_password_hash(password),
        is_active=True,
    )
    user.roles = _load_roles(db, normalized_roles)
    db.add(user)
    db.flush()
    return user
