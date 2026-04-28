"""add_faculty_duration_fields

Revision ID: 4c5d6e7f8a90
Revises: 28f032f7a7b3
Create Date: 2026-04-09 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "4c5d6e7f8a90"
down_revision: Union[str, Sequence[str], None] = "28f032f7a7b3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _column_names(bind, table_name: str) -> set[str]:
    inspector = sa.inspect(bind)
    return {column.get("name") for column in inspector.get_columns(table_name)}


def _check_constraint_names(bind, table_name: str) -> set[str]:
    inspector = sa.inspect(bind)
    return {
        constraint.get("name")
        for constraint in inspector.get_check_constraints(table_name)
        if constraint.get("name")
    }


def upgrade() -> None:
    bind = op.get_bind()
    columns = _column_names(bind, "faculties")

    with op.batch_alter_table("faculties") as batch_op:
        if "years" not in columns:
            batch_op.add_column(sa.Column("years", sa.Integer(), nullable=False, server_default=sa.text("4")))
        if "semesters" not in columns:
            batch_op.add_column(sa.Column("semesters", sa.Integer(), nullable=False, server_default=sa.text("8")))

    op.execute(
        sa.text(
            """
            UPDATE faculties
            SET years = COALESCE(years, 4),
                semesters = COALESCE(semesters, COALESCE(years, 4) * 2)
            """
        )
    )

    checks = _check_constraint_names(bind, "faculties")
    with op.batch_alter_table("faculties") as batch_op:
        if "ck_faculties_years_minimum" not in checks:
            batch_op.create_check_constraint("ck_faculties_years_minimum", "years >= 3")
        if "ck_faculties_semesters_match_years" not in checks:
            batch_op.create_check_constraint("ck_faculties_semesters_match_years", "semesters = years * 2")


def downgrade() -> None:
    bind = op.get_bind()
    columns = _column_names(bind, "faculties")
    checks = _check_constraint_names(bind, "faculties")

    with op.batch_alter_table("faculties") as batch_op:
        if "ck_faculties_semesters_match_years" in checks:
            batch_op.drop_constraint("ck_faculties_semesters_match_years", type_="check")
        if "ck_faculties_years_minimum" in checks:
            batch_op.drop_constraint("ck_faculties_years_minimum", type_="check")
        if "semesters" in columns:
            batch_op.drop_column("semesters")
        if "years" in columns:
            batch_op.drop_column("years")