"""course_title_unique_per_faculty

Revision ID: b6a7c8d9e0f1
Revises: f4e5d6c7b8a9
Create Date: 2026-04-01 13:10:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b6a7c8d9e0f1"
down_revision: Union[str, Sequence[str], None] = "f4e5d6c7b8a9"
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

    with op.batch_alter_table("courses") as batch_op:
        batch_op.add_column(sa.Column("faculty_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("normalized_title", sa.String(length=200), nullable=True))

    op.execute(
        sa.text(
            """
            UPDATE courses
            SET
                faculty_id = (
                    SELECT cb.faculty_id
                    FROM class_batches cb
                    WHERE cb.id = courses.class_batch_id
                ),
                normalized_title = lower(trim(title))
            """
        )
    )

    duplicates = bind.execute(
        sa.text(
            """
            SELECT faculty_id, normalized_title, COUNT(*) AS duplicate_count
            FROM courses
            GROUP BY faculty_id, normalized_title
            HAVING COUNT(*) > 1
            """
        )
    ).fetchall()
    if duplicates:
        details = ", ".join(
            f"(faculty_id={row[0]}, normalized_title='{row[1]}', count={row[2]})"
            for row in duplicates
        )
        raise RuntimeError(
            "Cannot apply uq_course_faculty_normalized_title because duplicate course titles exist: "
            + details
        )

    with op.batch_alter_table("courses") as batch_op:
        batch_op.alter_column("faculty_id", nullable=False)
        batch_op.alter_column("normalized_title", nullable=False)
        batch_op.create_foreign_key(
            "fk_courses_faculty_id_faculties",
            "faculties",
            ["faculty_id"],
            ["id"],
            ondelete="CASCADE",
        )

    unique_constraints = _unique_constraint_names(bind, "courses")
    if "uq_course_faculty_normalized_title" not in unique_constraints:
        with op.batch_alter_table("courses") as batch_op:
            batch_op.create_unique_constraint(
                "uq_course_faculty_normalized_title",
                ["faculty_id", "normalized_title"],
            )


def downgrade() -> None:
    bind = op.get_bind()
    unique_constraints = _unique_constraint_names(bind, "courses")

    with op.batch_alter_table("courses") as batch_op:
        if "uq_course_faculty_normalized_title" in unique_constraints:
            batch_op.drop_constraint("uq_course_faculty_normalized_title", type_="unique")
        batch_op.drop_constraint("fk_courses_faculty_id_faculties", type_="foreignkey")
        batch_op.drop_column("normalized_title")
        batch_op.drop_column("faculty_id")
