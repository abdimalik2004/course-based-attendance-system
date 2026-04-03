"""fix_classbatch_uniqueness_scope

Revision ID: f4e5d6c7b8a9
Revises: d1a2c3f4b5e6
Create Date: 2026-03-19 17:15:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f4e5d6c7b8a9"
down_revision: Union[str, Sequence[str], None] = "d1a2c3f4b5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _unique_constraint_names(bind, table_name: str) -> set[str]:
    inspector = sa.inspect(bind)
    return {
        constraint.get("name")
        for constraint in inspector.get_unique_constraints(table_name)
        if constraint.get("name")
    }


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    # Legacy databases created from early migrations may still enforce class name
    # uniqueness at (faculty_id, name), which incorrectly blocks same class names
    # across different departments within one faculty.
    unique_constraints = _unique_constraint_names(bind, "class_batches")
    index_names = {index.get("name") for index in inspector.get_indexes("class_batches") if index.get("name")}

    if "ix_class_batches_faculty_id" not in index_names:
        op.create_index("ix_class_batches_faculty_id", "class_batches", ["faculty_id"], unique=False)

    if "uq_class_batch_faculty_name" in unique_constraints:
        with op.batch_alter_table("class_batches") as batch_op:
            batch_op.drop_constraint("uq_class_batch_faculty_name", type_="unique")

    unique_constraints = _unique_constraint_names(bind, "class_batches")
    if "uq_class_batch_department_name" not in unique_constraints:
        with op.batch_alter_table("class_batches") as batch_op:
            batch_op.create_unique_constraint("uq_class_batch_department_name", ["department_id", "name"])

    department_unique_constraints = _unique_constraint_names(bind, "departments")
    with op.batch_alter_table("departments") as batch_op:
        if "uq_department_faculty_name" not in department_unique_constraints:
            batch_op.create_unique_constraint("uq_department_faculty_name", ["faculty_id", "name"])
        if "uq_department_faculty_code" not in department_unique_constraints:
            batch_op.create_unique_constraint("uq_department_faculty_code", ["faculty_id", "code"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    unique_constraints = _unique_constraint_names(bind, "class_batches")
    index_names = {index.get("name") for index in inspector.get_indexes("class_batches") if index.get("name")}

    with op.batch_alter_table("class_batches") as batch_op:
        if "uq_class_batch_department_name" in unique_constraints:
            batch_op.drop_constraint("uq_class_batch_department_name", type_="unique")
        if "uq_class_batch_faculty_name" not in unique_constraints:
            batch_op.create_unique_constraint("uq_class_batch_faculty_name", ["faculty_id", "name"])

    if "ix_class_batches_faculty_id" in index_names:
        op.drop_index("ix_class_batches_faculty_id", table_name="class_batches")
