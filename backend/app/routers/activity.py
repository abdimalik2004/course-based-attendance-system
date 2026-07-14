"""Activity log endpoints — used by the admin dashboard Recent Activity table."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from app.core.security import require_roles
from app.db.models import ActivityLog, ActivityLogStatus
from app.db.session import get_db
from app.services.activity_websocket_manager import activity_ws_manager
from app.utils.datetime_utils import current_local_datetime

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/activity", tags=["activity"])


def _safe_status(status_val) -> str:
    """Return the string value of an ActivityLogStatus safely.

    SQLAlchemy may return either the enum member or the raw stored string
    depending on version and driver.  Both cases are handled here.
    """
    if isinstance(status_val, ActivityLogStatus):
        return status_val.value
    if isinstance(status_val, str):
        return status_val
    return str(status_val)


@router.get(
    "/recent",
    dependencies=[Depends(require_roles("SUPER_ADMIN"))],
)
def recent_activity(
    limit: int = Query(default=30, ge=1, le=100),
    hours: int = Query(default=2, ge=1, le=24),
    db: Session = Depends(get_db),
):
    """
    Return the most recent activity log entries from the last N hours, newest first.

    This endpoint is optimized for the admin dashboard Recent Activity widget.
    It automatically filters to show only activities from the last 2 hours by default,
    but this can be customized via the `hours` query parameter.

    Query Parameters:
        limit: Maximum number of records to return (default: 30, max: 100)
        hours: Number of hours to look back (default: 2, max: 24)

    Returns:
        List of activity log records with id, username, action, status, and created_at (ISO format).

    Example:
        GET /activity/recent?limit=20&hours=2
        Returns max 20 activities from the last 2 hours, newest first.
    """
    try:
        cutoff_time = current_local_datetime() - timedelta(hours=hours)

        rows = (
            db.query(ActivityLog)
            .filter(ActivityLog.created_at >= cutoff_time)
            .order_by(ActivityLog.created_at.desc())
            .limit(limit)
            .all()
        )

        return [
            {
                "id": r.id,
                "username": r.username,
                "action": r.action,
                "status": _safe_status(r.status),
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]
    except Exception:
        logger.exception("Error fetching recent activity logs")
        return []


@router.get(
    "/stats",
    dependencies=[Depends(require_roles("SUPER_ADMIN"))],
)
def activity_stats(
    hours: int = Query(default=24, ge=1, le=168),
    db: Session = Depends(get_db),
):
    """
    Return activity statistics for the last N hours.

    This endpoint provides summary statistics useful for dashboards and reports.

    Query Parameters:
        hours: Number of hours to look back (default: 24, max: 168/1 week)

    Returns:
        Dictionary containing:
        - total_activities: Total number of activities in the time period
        - success_count: Number of successful activities
        - failed_count: Number of failed activities
        - pending_count: Number of pending activities
        - unique_users: Number of unique users who performed activities
    """
    cutoff_time = current_local_datetime() - timedelta(hours=hours)

    total = db.query(ActivityLog).filter(ActivityLog.created_at >= cutoff_time).count()

    from app.db.models import ActivityLogStatus

    success = (
        db.query(ActivityLog)
        .filter(
            and_(
                ActivityLog.created_at >= cutoff_time,
                ActivityLog.status == ActivityLogStatus.SUCCESS,
            )
        )
        .count()
    )

    failed = (
        db.query(ActivityLog)
        .filter(
            and_(
                ActivityLog.created_at >= cutoff_time,
                ActivityLog.status == ActivityLogStatus.FAILED,
            )
        )
        .count()
    )

    pending = (
        db.query(ActivityLog)
        .filter(
            and_(
                ActivityLog.created_at >= cutoff_time,
                ActivityLog.status == ActivityLogStatus.PENDING,
            )
        )
        .count()
    )

    unique_users = (
        db.query(ActivityLog.username)
        .filter(ActivityLog.created_at >= cutoff_time)
        .distinct()
        .count()
    )

    return {
        "total_activities": total,
        "success_count": success,
        "failed_count": failed,
        "pending_count": pending,
        "unique_users": unique_users,
    }


@router.get(
    "/logs",
    dependencies=[Depends(require_roles("SUPER_ADMIN"))],
)
def list_activity_logs(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    username: str | None = Query(default=None, description="Filter by exact or partial username"),
    status: str | None = Query(default=None, description="Filter by status: Success, Failed, Pending"),
    action: str | None = Query(default=None, description="Keyword search in action text"),
    date_from: str | None = Query(default=None, description="ISO date string, e.g. 2024-01-01"),
    date_to: str | None = Query(default=None, description="ISO date string, e.g. 2024-12-31"),
    db: Session = Depends(get_db),
):
    """
    Return paginated activity logs with optional filters.

    Supports filtering by username (partial match), status, action keyword, and date range.
    Returns total count so the frontend can render pagination controls.
    """
    query = db.query(ActivityLog)

    if username:
        query = query.filter(ActivityLog.username.ilike(f"%{username.strip()}%"))

    if status:
        status_upper = status.strip().capitalize()
        try:
            status_enum = ActivityLogStatus(status_upper)
            query = query.filter(ActivityLog.status == status_enum)
        except ValueError:
            pass  # unknown status value — ignore filter

    if action:
        query = query.filter(ActivityLog.action.ilike(f"%{action.strip()}%"))

    if date_from:
        try:
            dt_from = datetime.fromisoformat(date_from).replace(tzinfo=timezone.utc)
            query = query.filter(ActivityLog.created_at >= dt_from)
        except ValueError:
            pass

    if date_to:
        try:
            dt_to = datetime.fromisoformat(date_to).replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
            query = query.filter(ActivityLog.created_at <= dt_to)
        except ValueError:
            pass

    total = query.count()
    rows = query.order_by(ActivityLog.created_at.desc()).offset(skip).limit(limit).all()

    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "items": [
            {
                "id": r.id,
                "username": r.username,
                "action": r.action,
                "status": _safe_status(r.status),
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ],
    }


@router.get(
    "/hr-recent",
    dependencies=[Depends(require_roles("HR", "SUPER_ADMIN"))],
)
def hr_recent_activity(
    limit: int = Query(default=8, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """Return recent teacher-related activity logs for the HR dashboard.

    Scoped to log entries whose action text contains 'Teacher', so HR users
    only see their own domain — not attendance sessions, login events, etc.
    """
    rows = (
        db.query(ActivityLog)
        .filter(ActivityLog.action.ilike("%teacher%"))
        .order_by(ActivityLog.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": r.id,
            "username": r.username,
            "action": r.action,
            "status": _safe_status(r.status),
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


@router.websocket("/ws/recent")
async def websocket_recent_activity(websocket: WebSocket, token: str | None = None):
    """
    WebSocket endpoint for real-time activity updates.

    Requires a valid SUPER_ADMIN access token passed as a query parameter:
        ws://host/activity/ws/recent?token=<access_token>

    Connection flow:
    1. Client connects with ?token= query param
    2. Server validates token and role — closes with 4001 if unauthorized
    3. Manager accepts and registers the authenticated connection
    4. Client receives activity updates as they occur
    """
    from app.core.security import decode_token, TokenPayloadError, SUPER_ADMIN_ROLE
    from app.db.session import SessionLocal
    from app.db.models import User

    # Validate bearer token passed via query param (WebSocket can't use Authorization header)
    if not token:
        await websocket.close(code=4001, reason="Missing token")
        return

    db = SessionLocal()
    try:
        try:
            username = decode_token(token)
        except TokenPayloadError:
            await websocket.close(code=4001, reason="Invalid token")
            return

        user = db.query(User).filter(User.username == username, User.is_active == True).first()  # noqa: E712
        if not user:
            await websocket.close(code=4001, reason="Unauthorized")
            return

        assigned_roles = {role.name for role in user.roles}
        if SUPER_ADMIN_ROLE not in assigned_roles:
            await websocket.close(code=4003, reason="Forbidden: SUPER_ADMIN role required")
            return
    finally:
        db.close()

    await activity_ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Ignore incoming messages — this is a server-push-only channel
    except WebSocketDisconnect:
        await activity_ws_manager.disconnect(websocket)
    except Exception as exc:
        logger.error("WebSocket error: %s", exc)
        await activity_ws_manager.disconnect(websocket)

