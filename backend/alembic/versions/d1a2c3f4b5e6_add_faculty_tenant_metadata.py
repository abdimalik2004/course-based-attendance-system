"""add_faculty_tenant_metadata

Revision ID: d1a2c3f4b5e6
Revises: 5b2f9c7a6d11
Create Date: 2026-03-15 19:20:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d1a2c3f4b5e6"
down_revision: Union[str, Sequence[str], None] = "5b2f9c7a6d11"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("faculties") as batch_op:
        batch_op.add_column(sa.Column("tenant_db_name", sa.String(length=120), nullable=True))
        batch_op.add_column(sa.Column("tenant_db_provisioned_at", sa.DateTime(), nullable=True))
        batch_op.create_unique_constraint("uq_faculties_tenant_db_name", ["tenant_db_name"])


def downgrade() -> None:
    with op.batch_alter_table("faculties") as batch_op:
        batch_op.drop_constraint("uq_faculties_tenant_db_name", type_="unique")
        batch_op.drop_column("tenant_db_provisioned_at")
        batch_op.drop_column("tenant_db_name")
