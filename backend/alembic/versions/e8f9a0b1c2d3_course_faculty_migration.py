"""course_faculty_migration

Revision ID: e8f9a0b1c2d3
Revises: b6a7c8d9e0f1
Create Date: 2026-04-06 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e8f9a0b1c2d3"
down_revision: Union[str, Sequence[str], None] = "b6a7c8d9e0f1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _column_names(bind, table_name: str) -> set[str]:
    inspector = sa.inspect(bind)
    return {column.get("name") for column in inspector.get_columns(table_name)}


def _unique_constraint_names(bind, table_name: str) -> set[str]:
    inspector = sa.inspect(bind)
    return {
        constraint.get("name")
        for constraint in inspector.get_unique_constraints(table_name)
        if constraint.get("name")
    }


def _index_names(bind, table_name: str) -> set[str]:
    inspector = sa.inspect(bind)
    return {
        index.get("name")
        for index in inspector.get_indexes(table_name)
        if index.get("name")
    }


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = _column_names(bind, "courses")
    unique_constraints = _unique_constraint_names(bind, "courses")
    index_names = _index_names(bind, "courses")
    foreign_keys = {
        fk.get("name")
        for fk in inspector.get_foreign_keys("courses")
        if fk.get("name")
    }

    if "class_batch_id" in columns:
        op.execute(
            sa.text(
                """
                UPDATE courses
                SET faculty_id = (
                    SELECT cb.faculty_id
                    FROM class_batches cb
                    WHERE cb.id = courses.class_batch_id
                )
                WHERE class_batch_id IS NOT NULL
                """
            )
        )

    duplicate_rows = bind.execute(
        sa.text(
            """
            SELECT faculty_id, code, COUNT(*) AS duplicate_count
            FROM courses
            GROUP BY faculty_id, code
            HAVING COUNT(*) > 1
            """
        )
    ).fetchall()
    if duplicate_rows:
        details = ", ".join(
            f"(faculty_id={row[0]}, code='{row[1]}', count={row[2]})"
            for row in duplicate_rows
        )
        raise RuntimeError(
            "Cannot apply uq_course_faculty_code because duplicate course codes exist within a faculty: "
            + details
        )

    with op.batch_alter_table("courses") as batch_op:
        for fk_name in foreign_keys:
            fk = next(
                (
                    item
                    for item in inspector.get_foreign_keys("courses")
                    if item.get("name") == fk_name
                ),
                None,
            )
            if not fk:
                continue
            constrained = set(fk.get("constrained_columns") or [])
            if "class_batch_id" in constrained:
                batch_op.drop_constraint(fk_name, type_="foreignkey")
        if "uq_course_batch_code" in unique_constraints:
            batch_op.drop_constraint("uq_course_batch_code", type_="unique")
        if "class_batch_id" in columns:
            batch_op.drop_column("class_batch_id")
        if "ix_courses_faculty_id" not in index_names:
            batch_op.create_index("ix_courses_faculty_id", ["faculty_id"], unique=False)
        if "uq_course_faculty_code" not in unique_constraints:
            batch_op.create_unique_constraint("uq_course_faculty_code", ["faculty_id", "code"])


def downgrade() -> None:
    bind = op.get_bind()
    columns = _column_names(bind, "courses")
    unique_constraints = _unique_constraint_names(bind, "courses")
    index_names = _index_names(bind, "courses")

    faculty_to_batch = {
        faculty_id: class_batch_id
        for faculty_id, class_batch_id in bind.execute(
            sa.text(
                """
                SELECT faculty_id, MIN(id) AS class_batch_id
                FROM class_batches
                GROUP BY faculty_id
                """
            )
        ).fetchall()
    }

    if "class_batch_id" not in columns:
        with op.batch_alter_table("courses") as batch_op:
            batch_op.add_column(sa.Column("class_batch_id", sa.Integer(), nullable=True))

    if faculty_to_batch:
        course_rows = bind.execute(sa.text("SELECT id, faculty_id FROM courses")).fetchall()
        for course_id, faculty_id in course_rows:
            class_batch_id = faculty_to_batch.get(faculty_id)
            if class_batch_id is not None:
                bind.execute(
                    sa.text("UPDATE courses SET class_batch_id = :class_batch_id WHERE id = :course_id"),
                    {"class_batch_id": class_batch_id, "course_id": course_id},
                )

    with op.batch_alter_table("courses") as batch_op:
        if "uq_course_faculty_code" in unique_constraints:
            batch_op.drop_constraint("uq_course_faculty_code", type_="unique")
        if "ix_courses_faculty_id" in index_names:
            batch_op.drop_index("ix_courses_faculty_id")
        if "uq_course_batch_code" not in unique_constraints:
            batch_op.create_unique_constraint("uq_course_batch_code", ["class_batch_id", "code"])
