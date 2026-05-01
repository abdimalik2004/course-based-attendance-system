"""add_academic_structure_tables

Revision ID: c2d3e4f5a6b7
Revises: 28f032f7a7b3
Create Date: 2026-04-29 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c2d3e4f5a6b7"
down_revision: Union[str, Sequence[str], None] = "4c5d6e7f8a90"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "academic_years",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("academic_year", sa.String(length=32), nullable=False),
        sa.Column("term_name", sa.String(length=64), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.Column("end_date", sa.Date(), nullable=False),
        sa.Column(
            "status",
            sa.Enum("active", "inactive", "draft", name="academic_year_status"),
            nullable=False,
            server_default=sa.text("'draft'"),
        ),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("end_date > start_date", name="ck_academic_years_date_order"),
        sa.UniqueConstraint("academic_year", name="uq_academic_years_academic_year"),
    )
    op.create_index("ix_academic_years_id", "academic_years", ["id"], unique=False)
    op.create_index("ix_academic_years_status", "academic_years", ["status"], unique=False)

    op.create_table(
        "course_semester_assignments",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("course_id", sa.Integer(), nullable=False),
        sa.Column("faculty_id", sa.Integer(), nullable=False),
        sa.Column("department_id", sa.Integer(), nullable=False),
        sa.Column("academic_year_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["course_id"], ["courses.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["faculty_id"], ["faculties.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["department_id"], ["departments.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["academic_year_id"], ["academic_years.id"]),
        sa.UniqueConstraint(
            "course_id",
            "faculty_id",
            "department_id",
            "academic_year_id",
            name="uq_course_semester_assignment",
        ),
    )
    op.create_index("ix_course_semester_assignments_id", "course_semester_assignments", ["id"], unique=False)
    op.create_index("ix_course_semester_assignments_course_id", "course_semester_assignments", ["course_id"], unique=False)
    op.create_index("ix_course_semester_assignments_faculty_id", "course_semester_assignments", ["faculty_id"], unique=False)
    op.create_index("ix_course_semester_assignments_department_id", "course_semester_assignments", ["department_id"], unique=False)
    op.create_index("ix_course_semester_assignments_academic_year_id", "course_semester_assignments", ["academic_year_id"], unique=False)

    op.create_table(
        "class_course_assignments",
        sa.Column("id", sa.Integer(), primary_key=True, nullable=False),
        sa.Column("class_id", sa.Integer(), nullable=False),
        sa.Column("course_id", sa.Integer(), nullable=False),
        sa.Column("faculty_id", sa.Integer(), nullable=False),
        sa.Column("department_id", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["class_id"], ["class_batches.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["course_id"], ["courses.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["faculty_id"], ["faculties.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["department_id"], ["departments.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("class_id", "course_id", "faculty_id", "department_id", name="uq_class_course_assignment"),
    )
    op.create_index("ix_class_course_assignments_id", "class_course_assignments", ["id"], unique=False)
    op.create_index("ix_class_course_assignments_class_id", "class_course_assignments", ["class_id"], unique=False)
    op.create_index("ix_class_course_assignments_course_id", "class_course_assignments", ["course_id"], unique=False)
    op.create_index("ix_class_course_assignments_faculty_id", "class_course_assignments", ["faculty_id"], unique=False)
    op.create_index("ix_class_course_assignments_department_id", "class_course_assignments", ["department_id"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_class_course_assignments_department_id", table_name="class_course_assignments")
    op.drop_index("ix_class_course_assignments_faculty_id", table_name="class_course_assignments")
    op.drop_index("ix_class_course_assignments_course_id", table_name="class_course_assignments")
    op.drop_index("ix_class_course_assignments_class_id", table_name="class_course_assignments")
    op.drop_index("ix_class_course_assignments_id", table_name="class_course_assignments")
    op.drop_table("class_course_assignments")

    op.drop_index("ix_course_semester_assignments_academic_year_id", table_name="course_semester_assignments")
    op.drop_index("ix_course_semester_assignments_department_id", table_name="course_semester_assignments")
    op.drop_index("ix_course_semester_assignments_faculty_id", table_name="course_semester_assignments")
    op.drop_index("ix_course_semester_assignments_course_id", table_name="course_semester_assignments")
    op.drop_index("ix_course_semester_assignments_id", table_name="course_semester_assignments")
    op.drop_table("course_semester_assignments")

    op.drop_index("ix_academic_years_status", table_name="academic_years")
    op.drop_index("ix_academic_years_id", table_name="academic_years")
    op.drop_table("academic_years")
