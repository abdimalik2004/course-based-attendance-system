from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timezone
import logging
from threading import Lock

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import AttendanceSession, CourseSchedule, Faculty, SessionStatus
from app.db.session import SessionLocal, get_tenant_sessionmaker
from app.services.attendance_service import attendance_service
from app.utils.datetime_utils import combine_today, current_local_datetime, schedule_weekday_from_datetime
from app.utils.weekday_utils import storage_contains_weekday


@dataclass
class TenantTickStatus:
    faculty_id: int
    faculty_code: str
    tenant_db_name: str
    last_tick_started_at: str | None = None
    last_tick_completed_at: str | None = None
    last_success_at: str | None = None
    last_error: str | None = None
    total_success: int = 0
    total_failures: int = 0
    consecutive_failures: int = 0


class ScheduleService:
    def __init__(self) -> None:
        self._task: asyncio.Task | None = None
        self._stop_event: asyncio.Event | None = None
        self._logger = logging.getLogger(__name__)
        self._tenant_tick_status: dict[str, TenantTickStatus] = {}
        self._status_lock = Lock()
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

    def _tenant_alert_reasons(self, status: TenantTickStatus, now: datetime | None = None) -> list[str]:
        alert_reasons: list[str] = []
        threshold = max(1, settings.scheduler_tenant_failure_threshold)
        stale_seconds = max(1, settings.scheduler_tenant_stale_seconds)
        reference_now = now or self._utc_now()

        if status.consecutive_failures >= threshold:
            alert_reasons.append(f"consecutive_failures>={threshold}")

        last_completed = self._parse_iso8601(status.last_tick_completed_at)
        if self.is_running() and last_completed is not None:
            age_seconds = (reference_now - last_completed).total_seconds()
            if age_seconds > stale_seconds:
                alert_reasons.append(f"last_tick_stale>{stale_seconds}s")

        return alert_reasons

    def readiness_status(self) -> dict[str, object]:
        report = self.tenant_tick_report()
        healthy = report["scheduler_running"] and report["last_loop_error"] is None
        reason = None

        if not report["scheduler_running"]:
            reason = "Scheduler service is not running"
        elif report["last_loop_error"] is not None:
            reason = f"Scheduler loop error: {report['last_loop_error']}"
        elif report["tenant_mode_enabled"] and report["tenant_count"] == 0:
            healthy = False
            reason = "Tenant scheduler is enabled but no eligible tenants were discovered"
        elif report["unhealthy_tenant_count"] > 0:
            healthy = False
            tenant_codes = ", ".join(item["faculty_code"] for item in report["unhealthy_tenants"])
            reason = f"Tenant scheduler unhealthy for: {tenant_codes}"

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

        if settings.tenant_db_runtime_routing_enabled and settings.tenant_db_scheduler_enabled:
            self._tick_all_tenants()
            self._last_loop_completed_at = self._utc_now().isoformat()
            return

        db = SessionLocal()
        try:
            self._tick(db)
            self._last_loop_completed_at = self._utc_now().isoformat()
        except Exception as exc:  # noqa: BLE001
            self._last_loop_error = str(exc)
            self._logger.exception("Central scheduler tick failed", exc_info=exc)
        finally:
            db.close()

    def _tick_all_tenants(self) -> None:
        central_db = SessionLocal()
        try:
            faculties = (
                central_db.query(Faculty)
                .filter(
                    Faculty.tenant_db_name.is_not(None),
                    Faculty.tenant_db_provisioned_at.is_not(None),
                )
                .order_by(Faculty.id)
                .all()
            )
        finally:
            central_db.close()

        for faculty in faculties:
            if not faculty.tenant_db_name:
                continue

            started_at = self._utc_now().isoformat()
            with self._status_lock:
                status = self._tenant_tick_status.get(faculty.tenant_db_name)
                if status is None:
                    status = TenantTickStatus(
                        faculty_id=faculty.id,
                        faculty_code=faculty.code,
                        tenant_db_name=faculty.tenant_db_name,
                    )
                    self._tenant_tick_status[faculty.tenant_db_name] = status
                else:
                    status.faculty_id = faculty.id
                    status.faculty_code = faculty.code
                status.last_tick_started_at = started_at

            tenant_db = get_tenant_sessionmaker(faculty.tenant_db_name)()
            try:
                self._tick(tenant_db)
                finished_at = self._utc_now().isoformat()
                with self._status_lock:
                    status = self._tenant_tick_status[faculty.tenant_db_name]
                    status.last_tick_completed_at = finished_at
                    status.last_success_at = finished_at
                    status.last_error = None
                    status.total_success += 1
                    status.consecutive_failures = 0
            except Exception as exc:  # noqa: BLE001
                tenant_db.rollback()
                finished_at = self._utc_now().isoformat()
                with self._status_lock:
                    status = self._tenant_tick_status[faculty.tenant_db_name]
                    status.last_tick_completed_at = finished_at
                    status.last_error = str(exc)
                    status.total_failures += 1
                    status.consecutive_failures += 1
                self._logger.exception(
                    "Scheduler tick failed for faculty_id=%s tenant_db=%s",
                    faculty.id,
                    faculty.tenant_db_name,
                    exc_info=exc,
                )
            finally:
                tenant_db.close()

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

    def tenant_tick_report(self) -> dict:
        generated_at = self._utc_now()
        with self._status_lock:
            tenant_items = [
                {
                    "faculty_id": status.faculty_id,
                    "faculty_code": status.faculty_code,
                    "tenant_db_name": status.tenant_db_name,
                    "last_tick_started_at": status.last_tick_started_at,
                    "last_tick_completed_at": status.last_tick_completed_at,
                    "last_success_at": status.last_success_at,
                    "last_error": status.last_error,
                    "total_success": status.total_success,
                    "total_failures": status.total_failures,
                    "consecutive_failures": status.consecutive_failures,
                    "alert_reasons": self._tenant_alert_reasons(status, generated_at),
                }
                for status in sorted(self._tenant_tick_status.values(), key=lambda row: row.faculty_id)
            ]

        unhealthy_tenants = [
            {**item, "is_healthy": False}
            for item in tenant_items
            if item["alert_reasons"]
        ]
        tenant_items = [
            {**item, "is_healthy": not item["alert_reasons"]}
            for item in tenant_items
        ]

        mode = "tenant" if settings.tenant_db_runtime_routing_enabled and settings.tenant_db_scheduler_enabled else "central"
        return {
            "scheduler_running": self.is_running(),
            "mode": mode,
            "tenant_mode_enabled": settings.tenant_db_runtime_routing_enabled and settings.tenant_db_scheduler_enabled,
            "last_loop_started_at": self._last_loop_started_at,
            "last_loop_completed_at": self._last_loop_completed_at,
            "last_loop_error": self._last_loop_error,
            "alert_thresholds": {
                "consecutive_failures": max(1, settings.scheduler_tenant_failure_threshold),
                "stale_seconds": max(1, settings.scheduler_tenant_stale_seconds),
            },
            "tenant_count": len(tenant_items),
            "unhealthy_tenant_count": len(unhealthy_tenants),
            "unhealthy_tenants": unhealthy_tenants,
            "tenants": tenant_items,
        }


schedule_service = ScheduleService()
