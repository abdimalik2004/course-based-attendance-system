"""add student_id fk to users

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f7
Create Date: 2026-05-22

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b2c3d4e5f6a7"
down_revision: Union[str, None] = "a1b2c3d4e5f7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("student_id", sa.Integer(), nullable=True),
    )
    op.create_index("ix_users_student_id", "users", ["student_id"], unique=False)
    op.create_foreign_key(
        "fk_users_student_id",
        "users",
        "students",
        ["student_id"],
        ["id"],
    )


def downgrade() -> None:
    op.drop_constraint("fk_users_student_id", "users", type_="foreignkey")
    op.drop_index("ix_users_student_id", table_name="users")
    op.drop_column("users", "student_id")
