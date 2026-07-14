"""Faculty-side excuse request endpoints.

GET  /excuse-requests         — list all requests for faculty (scoped by faculty_id)
PATCH /excuse-requests/{id}  — approve or deny a pending request
"""
from __future__ import annotations

from fastapi import APIRouter, Body, Depends
from sqlalchemy.orm import Session

from app.core.security import get_current_user, require_roles
from app.db.models import User
from app.db.role_scoped import get_role_scoped_db
from app.services.excuse_request_service import excuse_request_service

router = APIRouter(tags=["excuse-requests"])

_FACULTY_ROLES = ("FACULTY", "SUPER_ADMIN", "ACADEMIA")


@router.get(
    "/excuse-requests",
    dependencies=[Depends(require_roles(*_FACULTY_ROLES))],
)
async def list_excuse_requests(
    db: Session = Depends(get_role_scoped_db),
    current_user: User = Depends(get_current_user),
):
    """Return all excuse requests for the current user's faculty."""
    faculty_id = current_user.faculty_id
    if faculty_id is None:
        return []
    return excuse_request_service.list_for_faculty(db, faculty_id)


@router.patch(
    "/excuse-requests/{request_id}",
    dependencies=[Depends(require_roles(*_FACULTY_ROLES))],
)
async def review_excuse_request(
    request_id: int,
    action: str = Body(..., embed=True),   # "approve" | "deny"
    db: Session = Depends(get_role_scoped_db),
    current_user: User = Depends(get_current_user),
):
    """Approve or deny a pending excuse request."""
    faculty_id = current_user.faculty_id
    if faculty_id is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="No faculty assigned to your account")
    return excuse_request_service.review(db, request_id, faculty_id, current_user, action)
