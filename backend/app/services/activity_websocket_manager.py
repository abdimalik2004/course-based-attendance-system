"""WebSocket manager for real-time activity log broadcasting.

This module provides a centralized manager for WebSocket connections that
want to receive real-time activity log updates.

Usage:
    # In activity_logger.py after logging an activity
    await activity_ws_manager.broadcast_activity(activity)

    # In a WebSocket endpoint
    await activity_ws_manager.connect(websocket, user_id)
    await activity_ws_manager.disconnect(websocket, user_id)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING

from fastapi import WebSocket

if TYPE_CHECKING:
    from app.db.models import ActivityLog

logger = logging.getLogger(__name__)


class ActivityWebSocketManager:
    """Manager for WebSocket connections receiving activity updates."""

    def __init__(self):
        """Initialize the WebSocket manager."""
        # Map of user_id to list of WebSocket connections
        self.active_connections: dict[int | str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, user_id: int | str = "admin"):
        """
        Register a new WebSocket connection for a user.

        Args:
            websocket: The WebSocket connection
            user_id: User ID (default "admin" for admin dashboard)

        Raises:
            RuntimeError: If accept fails
        """
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        self.active_connections[user_id].append(websocket)
        logger.debug(f"WebSocket connected for user {user_id}")

    async def disconnect(self, websocket: WebSocket, user_id: int | str = "admin"):
        """
        Unregister a WebSocket connection.

        Args:
            websocket: The WebSocket connection
            user_id: User ID
        """
        if user_id in self.active_connections:
            try:
                self.active_connections[user_id].remove(websocket)
                if not self.active_connections[user_id]:
                    del self.active_connections[user_id]
                logger.debug(f"WebSocket disconnected for user {user_id}")
            except ValueError:
                pass

    async def broadcast_activity(self, activity: ActivityLog):
        """
        Broadcast an activity log to all connected admin clients.

        This is called whenever a new activity is logged.

        Args:
            activity: The ActivityLog object to broadcast
        """
        message = {
            "type": "activity",
            "data": {
                "id": activity.id,
                "username": activity.username,
                "action": activity.action,
                "status": activity.status.value,
                "created_at": activity.created_at.isoformat(),
            },
        }

        # Broadcast to admin dashboard connections
        await self._broadcast_to_user("admin", message)

    async def _broadcast_to_user(self, user_id: int | str, message: dict):
        """
        Broadcast a message to all connections for a specific user.

        Args:
            user_id: User ID to broadcast to
            message: Message dictionary to send
        """
        if user_id not in self.active_connections:
            return

        disconnected = []
        for websocket in self.active_connections[user_id]:
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.debug(f"Error sending WebSocket message: {e}")
                disconnected.append(websocket)

        # Clean up disconnected clients
        for ws in disconnected:
            await self.disconnect(ws, user_id)

    async def broadcast_to_all(self, message: dict):
        """
        Broadcast a message to all connected clients.

        Args:
            message: Message dictionary to send
        """
        for user_id in list(self.active_connections.keys()):
            await self._broadcast_to_user(user_id, message)

    def get_connection_count(self, user_id: int | str = "admin") -> int:
        """
        Get the number of active connections for a user.

        Args:
            user_id: User ID to check

        Returns:
            Number of active connections
        """
        return len(self.active_connections.get(user_id, []))

    def get_total_connections(self) -> int:
        """
        Get the total number of active connections across all users.

        Returns:
            Total number of active connections
        """
        return sum(len(conns) for conns in self.active_connections.values())


# Global instance
activity_ws_manager = ActivityWebSocketManager()
