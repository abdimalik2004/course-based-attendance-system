"""Notifications router — REST CRUD + per-user WebSocket."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.core.security import decode_token, get_current_user
from app.db.models import Notification, User
from app.db.session import get_db, SessionLocal
from app.services.notification_service import notification_ws_manager

router = APIRouter(prefix="/notifications", tags=["notifications"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class NotificationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    message: str
    type: str
    is_read: bool
    link: str | None
    created_at: str

    @classmethod
    def from_orm_custom(cls, n: Notification) -> "NotificationRead":
        return cls(
            id=n.id,
            title=n.title,
            message=n.message,
            type=n.type.value if hasattr(n.type, "value") else str(n.type),
            is_read=n.is_read,
            link=n.link,
            created_at=n.created_at.isoformat(),
        )


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@router.get("", response_model=list[NotificationRead])
def list_notifications(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Return the authenticated user's notifications, newest first."""
    rows = (
        db.query(Notification)
        .filter(Notification.user_id == current_user.id)
        .order_by(Notification.created_at.desc())
        .limit(50)
        .all()
    )
    return [NotificationRead.from_orm_custom(n) for n in rows]


@router.get("/unread-count")
def unread_count(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    count = (
        db.query(Notification)
        .filter(Notification.user_id == current_user.id, Notification.is_read == False)
        .count()
    )
    return {"count": count}


@router.put("/{notif_id}/read", response_model=NotificationRead)
def mark_read(
    notif_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    n = db.query(Notification).filter(
        Notification.id == notif_id,
        Notification.user_id == current_user.id,
    ).first()
    if not n:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Notification not found")
    n.is_read = True
    db.commit()
    db.refresh(n)
    return NotificationRead.from_orm_custom(n)


@router.put("/read-all")
def mark_all_read(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    db.query(Notification).filter(
        Notification.user_id == current_user.id,
        Notification.is_read == False,
    ).update({"is_read": True})
    db.commit()
    return {"ok": True}


@router.delete("/{notif_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_notification(
    notif_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    n = db.query(Notification).filter(
        Notification.id == notif_id,
        Notification.user_id == current_user.id,
    ).first()
    if not n:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Notification not found")
    db.delete(n)
    db.commit()


# ---------------------------------------------------------------------------
# WebSocket — real-time push
# ---------------------------------------------------------------------------

@router.websocket("/ws/{user_id}")
async def notifications_ws(
    websocket: WebSocket,
    user_id: int,
    token: str = Query(...),
):
    """
    Per-user WebSocket.  The frontend connects here after login with ?token=<jwt>.
    The server pushes new notifications as JSON objects as they arrive.
    """
    # Validate the JWT token from the query string
    db = SessionLocal()
    try:
        try:
            username = decode_token(token)
        except Exception:
            await websocket.close(code=4001)
            return

        user = db.query(User).filter(User.username == username, User.is_active == True).first()
        if not user or user.id != user_id:
            await websocket.close(code=4001)
            return
    finally:
        db.close()

    await notification_ws_manager.connect(websocket, user_id)
    try:
        while True:
            # Keep alive — ignore any client messages
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        await notification_ws_manager.disconnect(websocket, user_id)
