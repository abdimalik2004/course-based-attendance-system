from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
import logging

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import AttendanceSession, CourseSchedule, SessionStatus
from app.db.session import SessionLocal
from app.services.attendance_service import attendance_service
from app.utils.datetime_utils import combine_today, current_local_datetime, schedule_weekday_from_datetime
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

        closed_count, absent_count = self._close_overdue_active_sessions(db, now)
        if closed_count:
            self._logger.info(
                "Catch-up closed %d overdue session(s) and marked %d absence record(s)",
                closed_count,
                absent_count,
            )

        weekday = schedule_weekday_from_datetime(now)

        schedules = [
            row
            for row in db.query(CourseSchedule).all()
            if storage_contains_weekday(row.weekday, weekday)
        ]
        self._logger.debug("Scheduler tick at %s with %d schedules", now.isoformat(), len(schedules))
        for schedule in schedules:
            start_dt = combine_today(schedule.start_time)
            end_dt = combine_today(schedule.end_time)

            session = (
                db.query(AttendanceSession)
                .filter(
                    AttendanceSession.schedule_id == schedule.id,
                    AttendanceSession.session_date == date.today(),
                )
                .first()
            )
            same_course_day_session = (
                db.query(AttendanceSession)
                .filter(
                    AttendanceSession.course_id == schedule.course_id,
                    AttendanceSession.session_date == date.today(),
                )
                .first()
            )

            if start_dt <= now <= end_dt and session is None and same_course_day_session is None:
                db.add(
                    AttendanceSession(
                        course_id=schedule.course_id,
                        schedule_id=schedule.id,
                        session_date=date.today(),
                        start_time=start_dt,
                        end_time=end_dt,
                        status=SessionStatus.ACTIVE,
                    )
                )
                db.commit()
                self._logger.info(
                    "Created attendance session: schedule_id=%s course_id=%s date=%s",
                    schedule.id,
                    schedule.course_id,
                    date.today().isoformat(),
                )
                continue

            # If backend was down for the whole window, backfill a missed session
            # and close it immediately so absences are still generated.
            if now > end_dt and session is None and same_course_day_session is None:
                missed_session = AttendanceSession(
                    course_id=schedule.course_id,
                    schedule_id=schedule.id,
                    session_date=date.today(),
                    start_time=start_dt,
                    end_time=end_dt,
                    status=SessionStatus.ACTIVE,
                )
                db.add(missed_session)
                db.commit()
                db.refresh(missed_session)

                absent_count = attendance_service.close_session_and_mark_absent(db, missed_session)
                self._logger.info(
                    "Backfilled and closed missed session_id=%s for course_id=%s and marked %d absences",
                    missed_session.id,
                    missed_session.course_id,
                    absent_count,
                )
                continue

            if start_dt <= now <= end_dt and session is not None:
                self._logger.debug(
                    "Skipped duplicate session creation for schedule_id=%s date=%s session_id=%s",
                    schedule.id,
                    date.today().isoformat(),
                    session.id,
                )

            if start_dt <= now <= end_dt and session is None and same_course_day_session is not None:
                self._logger.warning(
                    "Skipped schedule_id=%s because course_id=%s already has session_id=%s for date=%s",
                    schedule.id,
                    schedule.course_id,
                    same_course_day_session.id,
                    date.today().isoformat(),
                )

            if session and session.status == SessionStatus.ACTIVE and now >= end_dt:
                absent_count = attendance_service.close_session_and_mark_absent(db, session)
                self._logger.info(
                    "Closed session_id=%s for course_id=%s and marked %d absences",
                    session.id,
                    session.course_id,
                    absent_count,
                )

    def _close_overdue_active_sessions(self, db: Session, now: datetime) -> tuple[int, int]:
        overdue_sessions = (
            db.query(AttendanceSession)
            .filter(
                AttendanceSession.status == SessionStatus.ACTIVE,
                AttendanceSession.end_time <= now,
            )
            .all()
        )

        closed = 0
        absences = 0
        for session in overdue_sessions:
            absences += attendance_service.close_session_and_mark_absent(db, session)
            closed += 1
        return closed, absences

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
