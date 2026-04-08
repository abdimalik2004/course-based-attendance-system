"""super_admin_and_schedule_weekdays

Revision ID: a1b2c3d4e5f6
Revises: f4e5d6c7b8a9
Create Date: 2026-04-08 00:00:00.000000
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from app.core.security import get_password_hash
from app.utils.weekday_utils import decode_weekday_storage


revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "f4e5d6c7b8a9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _ensure_super_admin_role(bind: sa.engine.Connection) -> int:
    role_row = bind.execute(sa.text("SELECT id FROM roles WHERE name = :name"), {"name": "SUPER_ADMIN"}).first()
    if role_row is not None:
        return int(role_row[0])

    result = bind.execute(sa.text("INSERT INTO roles (name) VALUES (:name)"), {"name": "SUPER_ADMIN"})
    role_id = getattr(result, "lastrowid", None)
    if role_id:
        return int(role_id)

    role_row = bind.execute(sa.text("SELECT id FROM roles WHERE name = :name"), {"name": "SUPER_ADMIN"}).first()
    if role_row is None:
        raise RuntimeError("Failed to create SUPER_ADMIN role")
    return int(role_row[0])


def _ensure_admin_user(bind: sa.engine.Connection, role_id: int) -> None:
    user_row = bind.execute(sa.text("SELECT id FROM users WHERE username = :username"), {"username": "admin"}).first()
    if user_row is None:
        bind.execute(
            sa.text(
                "INSERT INTO users (username, email, hashed_password, is_active) "
                "VALUES (:username, :email, :hashed_password, :is_active)"
            ),
            {
                "username": "admin",
                "email": "admin@university.edu",
                "hashed_password": get_password_hash("admin"),
                "is_active": 1,
            },
        )
        user_row = bind.execute(sa.text("SELECT id FROM users WHERE username = :username"), {"username": "admin"}).first()
        if user_row is None:
            raise RuntimeError("Failed to create admin user")

    user_id = int(user_row[0])
    link_row = bind.execute(
        sa.text("SELECT 1 FROM user_role_links WHERE user_id = :user_id AND role_id = :role_id"),
        {"user_id": user_id, "role_id": role_id},
    ).first()
    if link_row is None:
        bind.execute(
            sa.text("INSERT INTO user_role_links (user_id, role_id) VALUES (:user_id, :role_id)"),
            {"user_id": user_id, "role_id": role_id},
        )


def _backfill_schedule_weekdays(bind: sa.engine.Connection) -> None:
    inspector = sa.inspect(bind)
    if "course_schedule_weekdays" not in inspector.get_table_names():
        op.create_table(
            "course_schedule_weekdays",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("schedule_id", sa.Integer(), nullable=False),
            sa.Column("weekday", sa.Integer(), nullable=False),
            sa.ForeignKeyConstraint(["schedule_id"], ["course_schedules.id"], ondelete="CASCADE"),
            sa.UniqueConstraint("schedule_id", "weekday", name="uq_schedule_weekday"),
        )

    existing_pairs = set(
        tuple(row)
        for row in bind.execute(sa.text("SELECT schedule_id, weekday FROM course_schedule_weekdays"))
    )

    rows = list(bind.execute(sa.text("SELECT id, weekday FROM course_schedules")))
    inserts: list[dict[str, int]] = []
    for row in rows:
        schedule_id = int(row[0])
        weekday_raw = row[1]
        for weekday in decode_weekday_storage(weekday_raw):
            pair = (schedule_id, weekday)
            if pair in existing_pairs:
                continue
            existing_pairs.add(pair)
            inserts.append({"schedule_id": schedule_id, "weekday": weekday})

    if inserts:
        op.bulk_insert(
            sa.table(
                "course_schedule_weekdays",
                sa.column("schedule_id", sa.Integer()),
                sa.column("weekday", sa.Integer()),
            ),
            inserts,
        )


def upgrade() -> None:
    bind = op.get_bind()

    role_id = _ensure_super_admin_role(bind)
    _ensure_admin_user(bind, role_id)
    _backfill_schedule_weekdays(bind)


def downgrade() -> None:
    bind = op.get_bind()

    op.drop_table("course_schedule_weekdays")

    bind.execute(
        sa.text("DELETE FROM user_role_links WHERE role_id IN (SELECT id FROM roles WHERE name = :name)"),
        {"name": "SUPER_ADMIN"},
    )
    bind.execute(sa.text("DELETE FROM users WHERE username = :username"), {"username": "admin"})
    bind.execute(sa.text("DELETE FROM roles WHERE name = :name"), {"name": "SUPER_ADMIN"})
