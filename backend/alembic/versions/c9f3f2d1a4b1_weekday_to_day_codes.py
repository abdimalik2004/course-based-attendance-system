"""weekday_to_day_codes

Revision ID: c9f3f2d1a4b1
Revises: aa8dbe49d336
Create Date: 2026-03-13 04:10:00.000000

"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c9f3f2d1a4b1"
down_revision: Union[str, Sequence[str], None] = "aa8dbe49d336"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


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


def _decode_legacy_weekday(value: int | str | None) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        raw = value.strip().lower()
        if not raw:
            return ""
        if "," in raw:
            parts = [part.strip() for part in raw.split(",") if part.strip()]
            return ",".join(parts)
        if raw in _DAY_TO_INDEX:
            return raw
        if raw.isdigit():
            value = int(raw)
        else:
            return raw

    if isinstance(value, int):
        if 1 <= value <= 7:
            return _INDEX_TO_DAY[value]
        if value >= 1000:
            mask = value - 1000
            days = [
                _INDEX_TO_DAY[day]
                for day in range(1, 8)
                if mask & (1 << (day - 1))
            ]
            return ",".join(days)
    return ""


def upgrade() -> None:
    bind = op.get_bind()

    inspector = sa.inspect(bind)
    weekday_col = next(
        (col for col in inspector.get_columns("course_schedules") if col.get("name") == "weekday"),
        None,
    )
    if weekday_col is not None:
        col_type_name = str(weekday_col.get("type", "")).upper()
        if "CHAR" in col_type_name or "TEXT" in col_type_name or "STRING" in col_type_name:
            # Schema already uses day-code text storage; treat this migration as applied.
            return

    rows = list(bind.execute(sa.text("SELECT id, weekday FROM course_schedules")))
    converted: list[tuple[str, int]] = []
    for row in rows:
        schedule_id = int(row[0])
        weekday_raw = row[1]
        weekday_codes = _decode_legacy_weekday(weekday_raw)
        if not weekday_codes:
            continue
        converted.append((weekday_codes, schedule_id))

    with op.batch_alter_table("course_schedules") as batch_op:
        batch_op.drop_constraint("uq_course_weekday_time", type_="unique")
        batch_op.alter_column(
            "weekday",
            existing_type=sa.Integer(),
            type_=sa.String(length=64),
            existing_nullable=False,
        )

    for weekday_codes, schedule_id in converted:
        bind.execute(
            sa.text("UPDATE course_schedules SET weekday = :weekday WHERE id = :id"),
            {"weekday": weekday_codes, "id": schedule_id},
        )

    with op.batch_alter_table("course_schedules") as batch_op:
        batch_op.create_unique_constraint(
            "uq_course_weekday_time",
            ["course_id", "weekday", "start_time"],
        )


def downgrade() -> None:
    bind = op.get_bind()

    rows = list(bind.execute(sa.text("SELECT id, weekday FROM course_schedules")))
    converted: list[tuple[int, int]] = []
    for row in rows:
        schedule_id = int(row[0])
        weekday_raw = (row[1] or "").strip().lower()
        if not weekday_raw:
            continue
        if "," in weekday_raw:
            mask = 0
            for part in [part.strip() for part in weekday_raw.split(",") if part.strip()]:
                day = _DAY_TO_INDEX.get(part)
                if day:
                    mask |= 1 << (day - 1)
            value = 1000 + mask
        else:
            value = _DAY_TO_INDEX.get(weekday_raw, 1)
        converted.append((value, schedule_id))

    with op.batch_alter_table("course_schedules") as batch_op:
        batch_op.drop_constraint("uq_course_weekday_time", type_="unique")
        batch_op.alter_column(
            "weekday",
            existing_type=sa.String(length=64),
            type_=sa.Integer(),
            existing_nullable=False,
        )

    for weekday_value, schedule_id in converted:
        bind.execute(
            sa.text("UPDATE course_schedules SET weekday = :weekday WHERE id = :id"),
            {"weekday": weekday_value, "id": schedule_id},
        )

    with op.batch_alter_table("course_schedules") as batch_op:
        batch_op.create_unique_constraint(
            "uq_course_weekday_time",
            ["course_id", "weekday", "start_time"],
        )
