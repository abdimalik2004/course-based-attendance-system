"""make_student_class_batch_nullable

Revision ID: d4e5f6a7b8c9
Revises: b0c1d2e3f4a5
Create Date: 2026-05-04 16:45:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, Sequence[str], None] = "b0c1d2e3f4a5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.alter_column("students", "class_batch_id", existing_type=sa.Integer(), nullable=True)


def downgrade() -> None:
    op.execute(
        "UPDATE students SET class_batch_id = (SELECT id FROM class_batches ORDER BY id LIMIT 1) WHERE class_batch_id IS NULL"
    )
    op.alter_column("students", "class_batch_id", existing_type=sa.Integer(), nullable=False)