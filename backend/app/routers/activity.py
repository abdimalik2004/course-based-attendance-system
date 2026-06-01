"""Activity log endpoints — used by the admin dashboard Recent Activity table."""
from __future__ import annotations

import logging
from datetime import timedelta

from fastapi import APIRouter, Depends, Query, WebSocket, WebSocketDisconnect
from sqlalchemy import and_
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


@router.websocket("/ws/recent")
async def websocket_recent_activity(websocket: WebSocket):
    """
    WebSocket endpoint for real-time activity updates.

    Clients connecting to this endpoint will receive real-time notifications
    whenever a new activity is logged in the system.

    Connection flow:
    1. Client connects via WebSocket
    2. Manager accepts and registers connection
    3. Client receives activity updates as they occur
    4. Client/server disconnect cleans up resources

    Expected client code:
        const ws = new WebSocket('ws://localhost:8000/activity/ws/recent');
        ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            if (message.type === 'activity') {
                // Handle new activity
                console.log(message.data);
            }
        };

    Error handling:
        - Connection failures are logged and ignored
        - Disconnects are handled gracefully
    """
    await activity_ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and listen for messages
            # In a real scenario, clients might send control messages
            data = await websocket.receive_text()
            # For now, we just ignore incoming messages
            # Could implement ping/pong or control commands here
    except WebSocketDisconnect:
        await activity_ws_manager.disconnect(websocket)
    except Exception as e:
        # Log error but don't crash
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"WebSocket error: {e}")
        await activity_ws_manager.disconnect(websocket)

