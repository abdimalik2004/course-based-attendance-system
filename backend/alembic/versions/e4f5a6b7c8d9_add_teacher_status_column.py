"""add_teacher_status_column

Revision ID: e4f5a6b7c8d9
Revises: d2e3f4a5b6c7
Create Date: 2026-05-02 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e4f5a6b7c8d9"
down_revision: Union[str, Sequence[str], None] = "d2e3f4a5b6c7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE teachers ADD COLUMN status ENUM('Active', 'Onleave', 'Inactive') NOT NULL DEFAULT 'Active' AFTER role")


def downgrade() -> None:
    op.execute("ALTER TABLE teachers DROP COLUMN status")