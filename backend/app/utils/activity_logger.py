"""Utility for centralized activity logging across the entire application.

This module provides flexible activity logging functionality that supports both:
1. Transaction-managed logging (within existing DB sessions)
2. Standalone logging (creates its own session)

Usage within a router with existing session:
    from app.utils.activity_logger import log_activity
    log_activity(
        db=db,
        action="Teacher Registered",
        user=current_user,
        status="Success"
    )

Standalone usage (no session required):
    log_activity(
        action="Login Failed",
        username="invalid_user",
        status="Failed"
    )
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.models import ActivityLog, ActivityLogStatus, User
from app.db.session import SessionLocal

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def log_activity(
    action: str,
    user: User | None = None,
    username: str | None = None,
    status: str | ActivityLogStatus = "Success",
    db: Session | None = None,
    own_transaction: bool = True,
) -> ActivityLog | None:
    """
    Persist an activity log entry with flexible session management.

    This function supports two usage patterns:
    1. Within existing transaction: Pass db session, set own_transaction=False
    2. Standalone: Don't pass db, function creates its own session (own_transaction=True)

    Args:
        action: Description of the activity/action being logged
        user: User object performing the action (will extract username and user_id)
        username: Alternative username if User object not available
        status: Status of the action - "Success", "Failed", "Pending" (case-insensitive)
        db: Existing database session (optional)
        own_transaction: If True and no db provided, creates new session and commits.
                        If False, assumes caller manages transaction (flushes only).

    Returns:
        The created ActivityLog object if successful, None if an error occurs.

    Examples:
        # With User object within existing transaction
        log_activity(
            action="User Registered",
            user=current_user,
            db=db,
            own_transaction=False
        )

        # Standalone with just username
        log_activity(
            action="Failed Login Attempt",
            username="invalid_user",
            status="Failed"
        )

        # With custom session management
        db_session = SessionLocal()
        try:
            log_activity(
                action="Course Created",
                user=admin_user,
                db=db_session,
                own_transaction=True
            )
        except Exception as e:
            db_session.rollback()
            raise
    """
    try:
        # Resolve username from User object or explicit username param
        resolved_username = (
            username
            or (user.username if user else None)
            or "System"
        ).strip()

        # Normalize status string to enum
        if isinstance(status, str):
            _status_map = {s.value.lower(): s for s in ActivityLogStatus}
            status_enum = _status_map.get(status.lower(), ActivityLogStatus.SUCCESS)
        else:
            status_enum = status

        # Ensure action is not empty
        if not action or not action.strip():
            logger.warning("Activity log action is empty, skipping log")
            return None

        # Create log entry
        entry = ActivityLog(
            user_id=user.id if user else None,
            username=resolved_username,
            action=action.strip(),
            status=status_enum,
        )

        # Determine session management
        own_session = False
        if db is None:
            db = SessionLocal()
            own_session = True

        try:
            db.add(entry)

            if own_transaction:
                # Commit the log entry immediately (works for both owned and
                # borrowed sessions — the main resource was already committed
                # by the caller before log_activity was invoked).
                db.commit()
                db.refresh(entry)
                logger.debug(f"Logged activity: {action} by {resolved_username}")

                # Broadcast to WebSocket subscribers (non-blocking)
                _schedule_websocket_broadcast(entry)
            else:
                # Caller manages the transaction: stage only, do not commit.
                db.flush()
                logger.debug(f"Flushed activity log: {action} by {resolved_username}")

            return entry

        except SQLAlchemyError as exc:
            db.rollback()
            logger.warning(
                f"Activity log insert failed for action '{action}': {exc}",
                exc_info=exc,
            )
            return None
        finally:
            if own_session:
                db.close()

    except Exception as e:
        logger.error(f"Unexpected error in log_activity: {e}", exc_info=e)
        return None


def _schedule_websocket_broadcast(activity: ActivityLog) -> None:
    """
    Schedule a WebSocket broadcast of the activity.

    This is non-blocking and runs in the background using asyncio.
    If asyncio is not available or the event loop is not running,
    it fails silently to avoid blocking the main request.

    Args:
        activity: The ActivityLog object to broadcast
    """
    try:
        # Try to get the running event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, skip broadcast
            logger.debug("No async event loop, skipping WebSocket broadcast")
            return

        # Create a task to broadcast without blocking
        from app.services.activity_websocket_manager import activity_ws_manager

        asyncio.create_task(_async_broadcast_activity(activity))
    except Exception as e:
        # Silently fail if WebSocket broadcast fails
        logger.debug(f"Could not broadcast activity to WebSocket: {e}")


async def _async_broadcast_activity(activity: ActivityLog) -> None:
    """
    Asynchronously broadcast an activity to WebSocket subscribers.

    Args:
        activity: The ActivityLog object to broadcast
    """
    try:
        from app.services.activity_websocket_manager import activity_ws_manager

        await activity_ws_manager.broadcast_activity(activity)
    except Exception as e:
        logger.debug(f"WebSocket broadcast failed: {e}")


def log_authentication(
    action: str,
    username: str = "System",
    status: str | ActivityLogStatus = "Success",
    user: User | None = None,
    db: Session | None = None,
) -> ActivityLog | None:
    """
    Log authentication-related activities (login, logout, failed auth).

    Args:
        action: Authentication action ("User Login", "User Logout", "Failed Login Attempt")
        username: Username of the user
        status: Status of the action
        user: User object (optional)
        db: Database session (optional)

    Returns:
        The created ActivityLog object if successful, None if an error occurs.
    """
    return log_activity(
        action=action,
        user=user,
        username=username,
        status=status,
        db=db,
    )


def log_user_management(
    action: str,
    user: User | None = None,
    username: str | None = None,
    status: str | ActivityLogStatus = "Success",
    db: Session | None = None,
) -> ActivityLog | None:
    """
    Log user management activities (create, update, delete, register).

    Args:
        action: User management action
        user: User performing the action
        username: Alternative username if User not available
        status: Status of the action
        db: Database session (optional)

    Returns:
        The created ActivityLog object if successful, None if an error occurs.
    """
    return log_activity(
        action=action,
        user=user,
        username=username,
        status=status,
        db=db,
    )


def log_academic_action(
    action: str,
    user: User | None = None,
    username: str | None = None,
    status: str | ActivityLogStatus = "Success",
    db: Session | None = None,
) -> ActivityLog | None:
    """
    Log academic structure activities (faculty, department, course, schedule).

    Args:
        action: Academic action
        user: User performing the action
        username: Alternative username if User not available
        status: Status of the action
        db: Database session (optional)

    Returns:
        The created ActivityLog object if successful, None if an error occurs.
    """
    return log_activity(
        action=action,
        user=user,
        username=username,
        status=status,
        db=db,
    )


def log_attendance_action(
    action: str,
    user: User | None = None,
    username: str | None = None,
    status: str | ActivityLogStatus = "Success",
    db: Session | None = None,
) -> ActivityLog | None:
    """
    Log attendance-related activities (marked, session started, finalized).

    Args:
        action: Attendance action
        user: User performing the action
        username: Alternative username if User not available
        status: Status of the action
        db: Database session (optional)

    Returns:
        The created ActivityLog object if successful, None if an error occurs.
    """
    return log_activity(
        action=action,
        user=user,
        username=username,
        status=status,
        db=db,
    )


def log_file_operation(
    action: str,
    user: User | None = None,
    username: str | None = None,
    status: str | ActivityLogStatus = "Success",
    db: Session | None = None,
) -> ActivityLog | None:
    """
    Log file operation activities (upload, download, export).

    Args:
        action: File operation action
        user: User performing the action
        username: Alternative username if User not available
        status: Status of the action
        db: Database session (optional)

    Returns:
        The created ActivityLog object if successful, None if an error occurs.
    """
    return log_activity(
        action=action,
        user=user,
        username=username,
        status=status,
        db=db,
    )


def log_system_action(
    action: str,
    status: str | ActivityLogStatus = "Success",
    db: Session | None = None,
) -> ActivityLog | None:
    """
    Log system-level activities (backup, settings update, maintenance).

    Args:
        action: System action
        status: Status of the action
        db: Database session (optional)

    Returns:
        The created ActivityLog object if successful, None if an error occurs.
    """
    return log_activity(
        action=action,
        username="System",
        status=status,
        db=db,
    )
