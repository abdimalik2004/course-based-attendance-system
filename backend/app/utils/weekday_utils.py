from __future__ import annotations


_DAY_TO_INDEX = {
    "sat": 1,
    "sun": 2,
    "mon": 3,
    "tue": 4,
    "wed": 5,
    "thu": 6,
    "fri": 7,
}

_INDEX_TO_DAY = {value: key for key, value in _DAY_TO_INDEX.items()}


def parse_weekday(value: int | str) -> int:
    if isinstance(value, int):
        if 1 <= value <= 7:
            return value
        raise ValueError("weekday must be between 1 and 7")

    raw = value.strip().lower()
    if not raw:
        raise ValueError("weekday cannot be empty")

    if raw.isdigit():
        number = int(raw)
        if 1 <= number <= 7:
            return number
        raise ValueError("weekday must be between 1 and 7")

    key = raw[:3]
    if key in _DAY_TO_INDEX:
        return _DAY_TO_INDEX[key]

    raise ValueError("weekday must be one of sat,sun,mon,tue,wed,thu,fri or 1..7")


def parse_weekday_list(value: int | str | list[int | str]) -> list[int]:
    if isinstance(value, list):
        parsed = [parse_weekday(item) for item in value]
    elif isinstance(value, int):
        parsed = [parse_weekday(value)]
    else:
        raw = value.strip()
        if not raw:
            raise ValueError("weekday cannot be empty")
        parts = [part.strip() for part in raw.split(",") if part.strip()]
        parsed = [parse_weekday(part) for part in parts]

    # Keep order but remove duplicates.
    seen: set[int] = set()
    unique: list[int] = []
    for item in parsed:
        if item in seen:
            continue
        seen.add(item)
        unique.append(item)
    return unique


def weekday_code(value: int) -> str:
    return _INDEX_TO_DAY.get(value, "")


def encode_weekday_storage(weekdays: list[int]) -> str:
    # Store as comma-separated day codes, e.g. "sat,sun,mon".
    codes = [weekday_code(day) for day in weekdays]
    codes = [code for code in codes if code]
    if not codes:
        raise ValueError("weekday cannot be empty")
    return ",".join(codes)


def decode_weekday_storage(stored_value: int | str) -> list[int]:
    # Backward compatibility:
    # - legacy single-day numeric values 1..7
    # - previous encoded integer value 1000+bitmask
    # - current csv code storage "sat,sun,mon"
    if isinstance(stored_value, int):
        if 1 <= stored_value <= 7:
            return [stored_value]
        if stored_value >= 1000:
            mask = stored_value - 1000
            return [day for day in range(1, 8) if mask & (1 << (day - 1))]
        return []

    raw = stored_value.strip().lower()
    if not raw:
        return []

    if raw.isdigit():
        return decode_weekday_storage(int(raw))

    parts = [part.strip() for part in raw.split(",") if part.strip()]
    return parse_weekday_list(parts)


def decode_weekday_codes(stored_value: int | str) -> list[str]:
    return [weekday_code(day) for day in decode_weekday_storage(stored_value) if weekday_code(day)]


def storage_contains_weekday(stored_value: int | str, weekday: int) -> bool:
    return weekday in set(decode_weekday_storage(stored_value))


def weekdays_intersect(stored_a: int | str, stored_b: int | str) -> bool:
    return bool(set(decode_weekday_storage(stored_a)) & set(decode_weekday_storage(stored_b)))