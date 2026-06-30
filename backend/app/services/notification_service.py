"""Notification service — create, persist and broadcast per-user notifications.

Supports both async callers (await create_notification_async(...)) and sync
callers (create_notification(...)) — the latter uses asyncio.create_task() for
the WebSocket push, matching the same pattern as activity_logger.py.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from fastapi import WebSocket
from sqlalchemy.orm import Session

from app.db.models import Notification, NotificationType, User, Role

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WebSocket manager (per-user)
# ---------------------------------------------------------------------------

class NotificationWebSocketManager:
    """Manages per-user WebSocket connections for real-time notification push."""

    def __init__(self):
        self._connections: dict[int, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, user_id: int):
        await websocket.accept()
        self._connections.setdefault(user_id, []).append(websocket)

    async def disconnect(self, websocket: WebSocket, user_id: int):
        conns = self._connections.get(user_id, [])
        try:
            conns.remove(websocket)
        except ValueError:
            pass
        if not conns:
            self._connections.pop(user_id, None)

    async def push(self, user_id: int, payload: dict):
        dead: list[WebSocket] = []
        for ws in list(self._connections.get(user_id, [])):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect(ws, user_id)


notification_ws_manager = NotificationWebSocketManager()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _notif_to_dict(n: Notification) -> dict:
    return {
        "id": n.id,
        "title": n.title,
        "message": n.message,
        "type": n.type.value if hasattr(n.type, "value") else str(n.type),
        "is_read": n.is_read,
        "link": n.link,
        "created_at": n.created_at.isoformat(),
    }


async def _push_ws(user_id: int, payload: dict):
    try:
        await notification_ws_manager.push(user_id, payload)
    except Exception as exc:
        logger.debug("WS push failed for user %s: %s", user_id, exc)


def _schedule_ws_push(user_id: int, payload: dict):
    """Schedule a WS push from synchronous code (fire-and-forget)."""
    try:
        loop = asyncio.get_running_loop()
        asyncio.create_task(_push_ws(user_id, payload))
    except RuntimeError:
        logger.debug("No running event loop; skipping WS push for user %s", user_id)


# ---------------------------------------------------------------------------
# Public API — synchronous (for use in sync routers)
# ---------------------------------------------------------------------------

def create_notification(
    db: Session,
    user_id: int,
    title: str,
    message: str,
    notif_type: NotificationType = NotificationType.INFO,
    link: str | None = None,
) -> Notification:
    """Persist a notification and schedule a real-time WS push (non-blocking)."""
    notif = Notification(
        user_id=user_id,
        title=title,
        message=message,
        type=notif_type,
        link=link,
    )
    db.add(notif)
    db.commit()
    db.refresh(notif)
    _schedule_ws_push(user_id, _notif_to_dict(notif))
    return notif


def notify_faculty_admins(
    db: Session,
    faculty_id: int,
    title: str,
    message: str,
    notif_type: NotificationType = NotificationType.INFO,
    link: str | None = None,
):
    """Notify all users with FACULTY role that belong to the given faculty."""
    faculty_role = db.query(Role).filter(Role.name == "FACULTY").first()
    if not faculty_role:
        return
    admins = (
        db.query(User)
        .filter(
            User.faculty_id == faculty_id,
            User.roles.any(Role.id == faculty_role.id),
        )
        .all()
    )
    for user in admins:
        create_notification(db, user.id, title, message, notif_type, link)


def notify_admins(
    db: Session,
    title: str,
    message: str,
    notif_type: NotificationType = NotificationType.INFO,
    link: str | None = None,
):
    """Notify all SUPER_ADMIN users."""
    admin_role = db.query(Role).filter(Role.name == "SUPER_ADMIN").first()
    if not admin_role:
        return
    for u in db.query(User).filter(User.roles.any(Role.id == admin_role.id)).all():
        create_notification(db, u.id, title, message, notif_type, link)
