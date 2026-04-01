from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.core.config import settings


_FIXED_TIMEZONE_FALLBACKS = {
    "Africa/Mogadishu": timezone(timedelta(hours=3)),
    "EAT": timezone(timedelta(hours=3)),
}


def _resolve_timezone():
    try:
        return ZoneInfo(settings.app_timezone)
    except ZoneInfoNotFoundError:
        fallback = _FIXED_TIMEZONE_FALLBACKS.get(settings.app_timezone)
        if fallback is None:
            return timezone.utc
        return fallback


def current_local_datetime() -> datetime:
    # Return naive local wall-clock time in the configured timezone so it matches
    # the naive DateTime values stored in the database.
    return datetime.now(_resolve_timezone()).replace(tzinfo=None)


def combine_today(clock_time: time) -> datetime:
    now = current_local_datetime()
    return now.replace(
        hour=clock_time.hour,
        minute=clock_time.minute,
        second=clock_time.second,
        microsecond=0,
    )


def schedule_weekday_from_datetime(value: datetime) -> int:
    # Desired mapping: Saturday=1, Sunday=2, Monday=3, ..., Friday=7.
    return ((value.weekday() + 2) % 7) + 1
