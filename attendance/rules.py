from datetime import datetime, time, timedelta


def build_session_key(course_id: str, session_label: str, timestamp: datetime) -> str:
    date_key = timestamp.strftime("%Y%m%d")
    label = session_label if session_label else "default"
    return f"{course_id}:{date_key}:{label}"


def _parse_time(value: str):
    if not value:
        return None
    return datetime.strptime(value, "%H:%M").time()


def _normalize_days(value: str):
    if not value:
        return set()
    parts = [part.strip().lower() for part in value.split(",") if part.strip()]
    return {part[:3] for part in parts}


def select_course_for_time(courses, timestamp: datetime):
    now_time = timestamp.time()
    today = timestamp.strftime("%a").lower()[:3]

    scheduled = []
    for course in courses:
        start_time = _parse_time(course.get("start_time"))
        end_time = _parse_time(course.get("end_time"))
        days = _normalize_days(course.get("days"))
        if start_time is None or end_time is None or not days:
            continue
        if today not in days:
            continue
        scheduled.append((start_time, end_time, course))

    scheduled.sort(key=lambda item: item[0])
    shift_index = 0
    for start_time, end_time, course in scheduled:
        shift_index += 1
        if start_time <= now_time <= end_time:
            return course, shift_index
    return None, None


def build_auto_session_label(shift_index: int, timestamp: datetime) -> str:
    date_key = timestamp.strftime("%Y%m%d")
    return f"{date_key}:shift-{shift_index}"


def assess_course_time(course, timestamp: datetime):
    if not course:
        return "missing"

    start_time = _parse_time(course.get("start_time"))
    end_time = _parse_time(course.get("end_time"))
    days = _normalize_days(course.get("days"))
    today = timestamp.strftime("%a").lower()[:3]
    now_time = timestamp.time()

    if start_time is None or end_time is None or not days:
        return "no_schedule"
    if today not in days:
        return "wrong_day"
    if now_time < start_time:
        return "too_early"
    if now_time > end_time:
        return "too_late"
    return "ok"


def derive_attendance_status(course, timestamp: datetime, grace_minutes: int = 40):
    start_time = _parse_time(course.get("start_time")) if course else None
    if start_time is None:
        return "on_time"

    start_dt = datetime.combine(timestamp.date(), start_time)
    late_cutoff = start_dt + timedelta(minutes=grace_minutes)
    return "late" if timestamp > late_cutoff else "on_time"
