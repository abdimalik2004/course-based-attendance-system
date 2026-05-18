"""add_student_status_and_created_at

Revision ID: 1111aabbccdd
Revises: ffff11112222
Create Date: 2026-05-18 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "1111aabbccdd"
down_revision: Union[str, Sequence[str], None] = "ffff11112222"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE students ADD COLUMN status ENUM('pending', 'approved', 'rejected') "
        "NOT NULL DEFAULT 'pending' AFTER embedding_ref"
    )
    op.execute(
        "ALTER TABLE students ADD COLUMN created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP AFTER status"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE students DROP COLUMN created_at")
    op.execute("ALTER TABLE students DROP COLUMN status")
