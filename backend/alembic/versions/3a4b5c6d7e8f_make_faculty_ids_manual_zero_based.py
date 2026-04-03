"""restore_faculty_auto_increment

Revision ID: 3a4b5c6d7e8f
Revises: f1a2b3c4d5e6
Create Date: 2026-04-02 19:55:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "3a4b5c6d7e8f"
down_revision: Union[str, Sequence[str], None] = "f1a2b3c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("faculties") as batch_op:
        batch_op.alter_column(
            "id",
            existing_type=sa.Integer(),
            existing_nullable=False,
            autoincrement=True,
        )


def downgrade() -> None:
    with op.batch_alter_table("faculties") as batch_op:
        batch_op.alter_column(
            "id",
            existing_type=sa.Integer(),
            existing_nullable=False,
            autoincrement=False,
        )
