"""drop_faculty_tenant_metadata

Revision ID: e7a1b2c3d4e5
Revises: b6a7c8d9e0f1
Create Date: 2026-04-02 20:10:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e7a1b2c3d4e5"
down_revision: Union[str, Sequence[str], None] = "b6a7c8d9e0f1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _unique_constraint_names(bind, table_name: str) -> set[str]:
    inspector = sa.inspect(bind)
    return {
        constraint.get("name")
        for constraint in inspector.get_unique_constraints(table_name)
        if constraint.get("name")
    }


def _column_names(bind, table_name: str) -> set[str]:
    inspector = sa.inspect(bind)
    return {column.get("name") for column in inspector.get_columns(table_name)}


def upgrade() -> None:
    bind = op.get_bind()
    unique_constraints = _unique_constraint_names(bind, "faculties")
    columns = _column_names(bind, "faculties")

    with op.batch_alter_table("faculties") as batch_op:
        if "uq_faculties_tenant_db_name" in unique_constraints:
            batch_op.drop_constraint("uq_faculties_tenant_db_name", type_="unique")
        if "tenant_db_provisioned_at" in columns:
            batch_op.drop_column("tenant_db_provisioned_at")
        if "tenant_db_name" in columns:
            batch_op.drop_column("tenant_db_name")


def downgrade() -> None:
    bind = op.get_bind()
    columns = _column_names(bind, "faculties")
    unique_constraints = _unique_constraint_names(bind, "faculties")

    with op.batch_alter_table("faculties") as batch_op:
        if "tenant_db_name" not in columns:
            batch_op.add_column(sa.Column("tenant_db_name", sa.String(length=120), nullable=True))
        if "tenant_db_provisioned_at" not in columns:
            batch_op.add_column(sa.Column("tenant_db_provisioned_at", sa.DateTime(), nullable=True))
        if "uq_faculties_tenant_db_name" not in unique_constraints:
            batch_op.create_unique_constraint("uq_faculties_tenant_db_name", ["tenant_db_name"])
