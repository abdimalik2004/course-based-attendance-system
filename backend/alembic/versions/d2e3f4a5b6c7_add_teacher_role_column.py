"""add_teacher_role_column

Revision ID: d2e3f4a5b6c7
Revises: b1c2d3e4f5a6
Create Date: 2026-05-02 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d2e3f4a5b6c7"
down_revision: Union[str, Sequence[str], None] = "b1c2d3e4f5a6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE teachers ADD COLUMN role ENUM('Professor', 'Associate Professor', 'Assistant Professor', 'Lecturer') NOT NULL DEFAULT 'Lecturer' AFTER full_name"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE teachers DROP COLUMN role")