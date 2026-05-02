"""add_student_role

Revision ID: a9b8c7d6e5f4
Revises: e4f5a6b7c8d9
Create Date: 2026-05-02 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a9b8c7d6e5f4"
down_revision: Union[str, Sequence[str], None] = "e4f5a6b7c8d9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        sa.text(
            """
            INSERT INTO roles (name)
            SELECT 'Student'
            WHERE NOT EXISTS (
                SELECT 1
                FROM roles
                WHERE name = 'Student'
            )
            """
        )
    )


def downgrade() -> None:
    op.execute(sa.text("DELETE FROM roles WHERE name = 'Student'"))