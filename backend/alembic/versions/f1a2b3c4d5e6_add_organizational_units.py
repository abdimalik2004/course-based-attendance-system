"""add_organizational_units

Revision ID: f1a2b3c4d5e6
Revises: e7a1b2c3d4e5
Create Date: 2026-04-02 22:10:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f1a2b3c4d5e6"
down_revision: Union[str, Sequence[str], None] = "e7a1b2c3d4e5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "organizational_units",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("code", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.UniqueConstraint("code", name="uq_organizational_units_code"),
        sa.UniqueConstraint("name", name="uq_organizational_units_name"),
    )

    org_units = sa.table(
        "organizational_units",
        sa.column("code", sa.String(length=32)),
        sa.column("name", sa.String(length=120)),
    )

    op.bulk_insert(
        org_units,
        [
            {"code": "ACADEMIA", "name": "Academia"},
            {"code": "FACULTIES", "name": "Faculties"},
            {"code": "HR", "name": "HR"},
            {"code": "ADMISSIONS", "name": "Admissions"},
        ],
    )


def downgrade() -> None:
    op.drop_table("organizational_units")
