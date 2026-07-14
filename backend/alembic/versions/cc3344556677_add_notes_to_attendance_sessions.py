"""add notes to attendance_sessions

Revision ID: cc3344556677
Revises: bb2233445566, merge_e5f6a7b8c9d0_f9e0a1b2c3d4
Create Date: 2026-07-11 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "cc3344556677"
down_revision: Union[str, Sequence[str], None] = ("bb2233445566", "merge_e5f6a7b8c9d0_f9e0a1b2c3d4")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add optional notes column to attendance_sessions
    op.add_column(
        "attendance_sessions",
        sa.Column("notes", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("attendance_sessions", "notes")
