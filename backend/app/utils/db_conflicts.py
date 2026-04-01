from __future__ import annotations

from sqlalchemy.exc import IntegrityError


_DUPLICATE_MARKERS = (
    "unique constraint failed",
    "duplicate entry",
    "duplicate key value violates unique constraint",
    "is not unique",
)

_FOREIGN_KEY_MARKERS = (
    "foreign key constraint fails",
    "foreign key constraint failed",
    "violates foreign key constraint",
)


def classify_integrity_error(exc: IntegrityError) -> str:
    message = str(getattr(exc, "orig", exc)).lower()
    if any(marker in message for marker in _FOREIGN_KEY_MARKERS):
        return "foreign_key"
    if any(marker in message for marker in _DUPLICATE_MARKERS):
        return "duplicate"
    return "other"


def integrity_error_mentions(exc: IntegrityError, *markers: str) -> bool:
    message = str(getattr(exc, "orig", exc)).lower()
    return any(marker.lower() in message for marker in markers)