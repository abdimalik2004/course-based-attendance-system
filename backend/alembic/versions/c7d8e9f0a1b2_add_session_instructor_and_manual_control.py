"""add_session_instructor_and_manual_control

Revision ID: c7d8e9f0a1b2
Revises: f4e5d6c7b8a9
Create Date: 2026-05-02 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c7d8e9f0a1b2"
down_revision: Union[str, Sequence[str], None] = "f4e5d6c7b8a9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("attendance_sessions", sa.Column("instructor_id", sa.Integer(), nullable=True))
    op.alter_column("attendance_sessions", "end_time", existing_type=sa.DateTime(), nullable=True)
    op.create_foreign_key(
        "fk_attendance_sessions_instructor_id_users",
        "attendance_sessions",
        "users",
        ["instructor_id"],
        ["id"],
        ondelete="SET NULL",
    )


def downgrade() -> None:
    op.drop_constraint("fk_attendance_sessions_instructor_id_users", "attendance_sessions", type_="foreignkey")
    op.alter_column("attendance_sessions", "end_time", existing_type=sa.DateTime(), nullable=False)
    op.execute("ALTER TABLE attendance_sessions DROP COLUMN instructor_id")