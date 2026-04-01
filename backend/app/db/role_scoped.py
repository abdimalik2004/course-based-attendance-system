from __future__ import annotations

from collections.abc import Generator

from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import get_current_user
from app.db.models import Faculty, User
from app.db.session import SessionLocal, get_tenant_sessionmaker


def get_role_scoped_db(current_user: User = Depends(get_current_user)) -> Generator[Session, None, None]:
    # Central DB remains the source of truth for auth/platform data. Faculty-scoped
    # operational routes use tenant DBs once runtime routing is enabled.
    if not settings.tenant_db_runtime_routing_enabled:
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()
        return

    role_names = {role.name for role in current_user.roles}
    if "ACADEMIA" in role_names:
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()
        return

    if current_user.faculty_id is None:
        raise HTTPException(status_code=403, detail="Faculty-scoped user is missing faculty association")

    central_db = SessionLocal()
    try:
        faculty = central_db.query(Faculty).filter(Faculty.id == current_user.faculty_id).first()
    finally:
        central_db.close()

    if not faculty:
        raise HTTPException(status_code=404, detail="Faculty not found for current user")
    if not faculty.tenant_db_name:
        raise HTTPException(status_code=503, detail="Faculty tenant database is not configured")
    if faculty.tenant_db_provisioned_at is None:
        raise HTTPException(status_code=503, detail="Faculty tenant database is not provisioned")

    tenant_db = get_tenant_sessionmaker(faculty.tenant_db_name)()
    try:
        yield tenant_db
    finally:
        tenant_db.close()
