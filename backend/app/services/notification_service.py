"""Notification service — create, persist and broadcast per-user notifications.

Supports both async callers (await create_notification_async(...)) and sync
callers (create_notification(...)) — the latter uses run_coroutine_threadsafe
to schedule the WebSocket push onto the app's event loop from threadpool threads.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from fastapi import WebSocket
from sqlalchemy.orm import Session

from app.db.models import Notification, NotificationType, User, Role

logger = logging.getLogger(__name__)

# Event loop captured at startup (set by main.py lifespan) so that sync route
# handlers running in uvicorn's threadpool can still schedule coroutines.
_app_event_loop: asyncio.AbstractEventLoop | None = None


def set_app_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Call once from the lifespan startup to register the running event loop."""
    global _app_event_loop
    _app_event_loop = loop


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
    """Return the notification wrapped in the WS envelope the frontend expects."""
    payload = {
        "id": n.id,
        "title": n.title,
        "message": n.message,
        "type": n.type.value if hasattr(n.type, "value") else str(n.type),
        "is_read": n.is_read,
        "link": n.link,
        "created_at": n.created_at.isoformat(),
    }
    # Frontend useNotificationsStore expects: {type: "notification", payload: {...}}
    return {"type": "notification", "payload": payload}


async def _push_ws(user_id: int, envelope: dict):
    try:
        await notification_ws_manager.push(user_id, envelope)
    except Exception as exc:
        logger.debug("WS push failed for user %s: %s", user_id, exc)


def _schedule_ws_push(user_id: int, envelope: dict):
    """Schedule a WS push from synchronous code (fire-and-forget).

    Sync FastAPI route handlers run in a threadpool — there is no running event
    loop in that thread, so asyncio.create_task() raises RuntimeError.  Instead
    we use run_coroutine_threadsafe with the loop captured at startup.
    """
    if _app_event_loop is not None and _app_event_loop.is_running():
        asyncio.run_coroutine_threadsafe(_push_ws(user_id, envelope), _app_event_loop)
    else:
        # Fallback for async callers (e.g. async route handlers)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_push_ws(user_id, envelope))
        except RuntimeError:
            logger.debug("No event loop available; skipping WS push for user %s", user_id)


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
