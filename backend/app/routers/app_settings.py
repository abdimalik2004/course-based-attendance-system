"""System Settings router — GET /settings  &  PUT /settings (bulk upsert)."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.models import SystemSetting, User
from app.db.session import get_db
from app.utils.datetime_utils import set_runtime_timezone

router = APIRouter(prefix="/settings", tags=["settings"])


def _require_admin(current_user: User = Depends(get_current_user)) -> User:
    role_names = {r.name for r in current_user.roles}
    if "SUPER_ADMIN" not in role_names and "HR" not in role_names:
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="Insufficient permissions.")
    return current_user


@router.get("", response_model=dict[str, str])
def get_settings(
    db: Session = Depends(get_db),
    _: User = Depends(_require_admin),
) -> dict[str, str]:
    """Return all system settings as a flat key→value dict."""
    rows = db.query(SystemSetting).all()
    return {row.key: row.value for row in rows}


@router.put("", response_model=dict[str, str])
def update_settings(
    payload: dict[str, Any],
    db: Session = Depends(get_db),
    _: User = Depends(_require_admin),
) -> dict[str, str]:
    """Bulk-upsert settings.  Send only the keys you want to change."""
    for key, value in payload.items():
        row = db.query(SystemSetting).filter(SystemSetting.key == key).first()
        if row:
            row.value = str(value)
        else:
            db.add(SystemSetting(key=key, value=str(value)))

    db.commit()

    # Apply timezone change immediately so the running server uses the new
    # value without needing a restart.
    if "general.timezone" in payload and payload["general.timezone"]:
        set_runtime_timezone(str(payload["general.timezone"]))

    rows = db.query(SystemSetting).all()
    return {row.key: row.value for row in rows}
