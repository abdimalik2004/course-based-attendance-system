from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
import logging

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import CourseSchedule
from app.db.session import SessionLocal
from app.utils.datetime_utils import current_local_datetime, schedule_weekday_from_datetime
from app.utils.weekday_utils import storage_contains_weekday


class ScheduleService:
    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._stop_event: asyncio.Event | None = None
        self._logger = logging.getLogger(__name__)
        self._last_loop_started_at: str | None = None
        self._last_loop_completed_at: str | None = None
        self._last_loop_error: str | None = None

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def _parse_iso8601(value: str | None) -> datetime | None:
        if not value:
            return None
        return datetime.fromisoformat(value)

    def readiness_status(self) -> dict[str, object]:
        report = self.scheduler_report()
        healthy = report["scheduler_running"] and report["last_loop_error"] is None
        reason = None

        if not report["scheduler_running"]:
            reason = "Scheduler service is not running"
        elif report["last_loop_error"] is not None:
            reason = f"Scheduler loop error: {report['last_loop_error']}"

        return {
            "healthy": healthy,
            "reason": reason,
            "report": report,
        }

    async def start(self) -> None:
        if self._task and not self._task.done():
            self._logger.info("Schedule service already running")
            return
        self._stop_event = asyncio.Event()
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_loop())
        self._logger.info("Schedule service started")

    async def stop(self) -> None:
        if self._task is None:
            return
        if self._stop_event is not None:
            self._stop_event.set()
        await self._task
        self._logger.info("Schedule service stopped")

    async def _run_loop(self) -> None:
        while self._stop_event is not None and not self._stop_event.is_set():
            try:
                self._run_once()
            except Exception as exc:  # noqa: BLE001
                self._last_loop_error = str(exc)
                self._logger.exception("Scheduler loop iteration failed", exc_info=exc)
            await asyncio.sleep(settings.scheduler_poll_seconds)

    def _run_once(self) -> None:
        now_iso = self._utc_now().isoformat()
        self._last_loop_started_at = now_iso
        self._last_loop_error = None

        db = SessionLocal()
        try:
            self._tick(db)
            self._last_loop_completed_at = self._utc_now().isoformat()
        except Exception as exc:  # noqa: BLE001
            self._last_loop_error = str(exc)
            self._logger.exception("Central scheduler tick failed", exc_info=exc)
        finally:
            db.close()

    def _tick(self, db: Session) -> None:
        now = current_local_datetime()

        weekday = schedule_weekday_from_datetime(now)
        schedules = [row for row in db.query(CourseSchedule).all() if storage_contains_weekday(row.weekday, weekday)]
        self._logger.debug(
            "Scheduler tick at %s with %d schedule(s); automated attendance session management is disabled",
            now.isoformat(),
            len(schedules),
        )

    def _close_overdue_active_sessions(self, db: Session, now: datetime) -> tuple[int, int]:
        return 0, 0

    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    def scheduler_report(self) -> dict:
        return {
            "scheduler_running": self.is_running(),
            "mode": "central",
            "last_loop_started_at": self._last_loop_started_at,
            "last_loop_completed_at": self._last_loop_completed_at,
            "last_loop_error": self._last_loop_error,
        }


schedule_service = ScheduleService()
