from __future__ import annotations

from collections.abc import Generator

from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.models import User
from app.db.session import SessionLocal


def get_role_scoped_db(current_user: User = Depends(get_current_user)) -> Generator[Session, None, None]:
    role_names = {role.name for role in current_user.roles}
    if "SUPER_ADMIN" in role_names:
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()
        return

    if ("FACULTY" in role_names or "FACULTY_ADMIN" in role_names) and current_user.faculty_id is None:
        raise HTTPException(status_code=403, detail="Faculty-scoped user is missing faculty association")

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
