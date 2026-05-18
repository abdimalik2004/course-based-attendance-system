"""add_department_id_to_courses

Revision ID: f9e0a1b2c3d4
Revises: f4e5d6c7b8a9
Create Date: 2026-05-06 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision: str = "f9e0a1b2c3d4"
down_revision: Union[str, Sequence[str], None] = "f4e5d6c7b8a9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()
    inspector = inspect(bind)
    existing = {c["name"] for c in inspector.get_columns("courses")}
    if "department_id" not in existing:
        op.add_column(
            "courses",
            sa.Column("department_id", sa.Integer(), nullable=False),
        )
        op.create_foreign_key(
            "fk_courses_department_id",
            "courses",
            "departments",
            ["department_id"],
            ["id"],
            ondelete="CASCADE",
        )


def downgrade() -> None:
    """Downgrade schema."""
    bind = op.get_bind()
    inspector = inspect(bind)
    existing = {c["name"] for c in inspector.get_columns("courses")}
    if "department_id" in existing:
        # drop FK then column
        try:
            op.drop_constraint("fk_courses_department_id", "courses", type_="foreignkey")
        except Exception:
            # ignore if constraint missing
            pass
        op.drop_column("courses", "department_id")
