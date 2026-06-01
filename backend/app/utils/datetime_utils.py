from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.core.config import settings


_FIXED_TIMEZONE_FALLBACKS: dict[str, timezone] = {
    "Africa/Mogadishu": timezone(timedelta(hours=3)),
    "Africa/Nairobi":   timezone(timedelta(hours=3)),
    "Africa/Djibouti":  timezone(timedelta(hours=3)),
    "Asia/Riyadh":      timezone(timedelta(hours=3)),
    "Asia/Kuwait":      timezone(timedelta(hours=3)),
    "Asia/Aden":        timezone(timedelta(hours=3)),
    "Asia/Dubai":       timezone(timedelta(hours=4)),
    "Asia/Muscat":      timezone(timedelta(hours=4)),
    "EAT":              timezone(timedelta(hours=3)),
}

# Runtime override — set at startup (from DB) and on PUT /settings.
# Takes priority over the .env APP_TIMEZONE value.
_runtime_timezone: str | None = None


def set_runtime_timezone(tz_name: str) -> None:
    """Override the active timezone at runtime (no restart needed)."""
    global _runtime_timezone
    _runtime_timezone = tz_name.strip() if tz_name else None


def get_runtime_timezone() -> str:
    """Return the currently active IANA timezone name."""
    return _runtime_timezone or settings.app_timezone


def _resolve_timezone():
    tz_name = get_runtime_timezone()
    try:
        return ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        fallback = _FIXED_TIMEZONE_FALLBACKS.get(tz_name)
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
