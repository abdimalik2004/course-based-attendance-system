"""rename student attendance and schedule tables

Revision ID: 8d1c2b3a4f5e
Revises: 5fdacb70a9f9
Create Date: 2026-05-01 21:30:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "8d1c2b3a4f5e"
down_revision: Union[str, Sequence[str], None] = "5fdacb70a9f9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(table_name: str) -> bool:
    inspector = sa.inspect(op.get_bind())
    return table_name in inspector.get_table_names()


def _index_exists(table_name: str, index_name: str) -> bool:
    if not _table_exists(table_name):
        return False
    inspector = sa.inspect(op.get_bind())
    return any(index["name"] == index_name for index in inspector.get_indexes(table_name))


def _check_constraint_exists(table_name: str, constraint_name: str) -> bool:
    if not _table_exists(table_name):
        return False
    inspector = sa.inspect(op.get_bind())
    return any(constraint["name"] == constraint_name for constraint in inspector.get_check_constraints(table_name))


def upgrade() -> None:
    if _table_exists("attendance"):
        if _check_constraint_exists("attendance", "ck_attendance_classes_attended_not_greater_than_total"):
            op.drop_constraint("ck_attendance_classes_attended_not_greater_than_total", "attendance", type_="check")
        if _check_constraint_exists("attendance", "ck_attendance_classes_attended_nonnegative"):
            op.drop_constraint("ck_attendance_classes_attended_nonnegative", "attendance", type_="check")
        if _check_constraint_exists("attendance", "ck_attendance_total_classes_positive"):
            op.drop_constraint("ck_attendance_total_classes_positive", "attendance", type_="check")
        op.rename_table("attendance", "student_attendance")

    if _table_exists("schedule"):
        if _check_constraint_exists("schedule", "ck_schedule_grace_period_nonnegative"):
            op.drop_constraint("ck_schedule_grace_period_nonnegative", "schedule", type_="check")
        op.rename_table("schedule", "student_schedule")

    if _table_exists("student_attendance") and not _index_exists("student_attendance", "ix_student_attendance_course_code"):
        op.create_index(op.f("ix_student_attendance_course_code"), "student_attendance", ["course_code"], unique=False)
    if _table_exists("student_attendance") and not _index_exists("student_attendance", "ix_student_attendance_id"):
        op.create_index(op.f("ix_student_attendance_id"), "student_attendance", ["id"], unique=False)
    if _table_exists("student_attendance") and not _index_exists("student_attendance", "ix_student_attendance_student_id"):
        op.create_index(op.f("ix_student_attendance_student_id"), "student_attendance", ["student_id"], unique=False)
    if _table_exists("student_schedule") and not _index_exists("student_schedule", "ix_student_schedule_course_code"):
        op.create_index(op.f("ix_student_schedule_course_code"), "student_schedule", ["course_code"], unique=False)
    if _table_exists("student_schedule") and not _index_exists("student_schedule", "ix_student_schedule_id"):
        op.create_index(op.f("ix_student_schedule_id"), "student_schedule", ["id"], unique=False)
    if _table_exists("student_schedule") and not _index_exists("student_schedule", "ix_student_schedule_student_id"):
        op.create_index(op.f("ix_student_schedule_student_id"), "student_schedule", ["student_id"], unique=False)

    if _index_exists("student_attendance", "ix_attendance_course_code"):
        op.drop_index("ix_attendance_course_code", table_name="student_attendance")
    if _index_exists("student_attendance", "ix_attendance_id"):
        op.drop_index("ix_attendance_id", table_name="student_attendance")
    if _index_exists("student_attendance", "ix_attendance_student_id"):
        op.drop_index("ix_attendance_student_id", table_name="student_attendance")
    if _index_exists("student_schedule", "ix_schedule_course_code"):
        op.drop_index("ix_schedule_course_code", table_name="student_schedule")
    if _index_exists("student_schedule", "ix_schedule_id"):
        op.drop_index("ix_schedule_id", table_name="student_schedule")
    if _index_exists("student_schedule", "ix_schedule_student_id"):
        op.drop_index("ix_schedule_student_id", table_name="student_schedule")

    if _table_exists("student_attendance") and not _check_constraint_exists("student_attendance", "ck_student_attendance_classes_attended_nonnegative"):
        op.create_check_constraint(
            "ck_student_attendance_classes_attended_nonnegative",
            "student_attendance",
            "classes_attended >= 0",
        )
    if _table_exists("student_attendance") and not _check_constraint_exists("student_attendance", "ck_student_attendance_total_classes_positive"):
        op.create_check_constraint(
            "ck_student_attendance_total_classes_positive",
            "student_attendance",
            "total_classes > 0",
        )
    if _table_exists("student_attendance") and not _check_constraint_exists("student_attendance", "ck_student_attendance_classes_attended_not_greater_than_total"):
        op.create_check_constraint(
            "ck_student_attendance_classes_attended_not_greater_than_total",
            "student_attendance",
            "classes_attended <= total_classes",
        )
    if _table_exists("student_schedule") and not _check_constraint_exists("student_schedule", "ck_student_schedule_grace_period_nonnegative"):
        op.create_check_constraint(
            "ck_student_schedule_grace_period_nonnegative",
            "student_schedule",
            "grace_period_minutes >= 0",
        )


def downgrade() -> None:
    if _table_exists("student_schedule"):
        if _check_constraint_exists("student_schedule", "ck_student_schedule_grace_period_nonnegative"):
            op.drop_constraint("ck_student_schedule_grace_period_nonnegative", "student_schedule", type_="check")
        op.rename_table("student_schedule", "schedule")

    if _table_exists("student_attendance"):
        if _check_constraint_exists("student_attendance", "ck_student_attendance_classes_attended_not_greater_than_total"):
            op.drop_constraint("ck_student_attendance_classes_attended_not_greater_than_total", "student_attendance", type_="check")
        if _check_constraint_exists("student_attendance", "ck_student_attendance_total_classes_positive"):
            op.drop_constraint("ck_student_attendance_total_classes_positive", "student_attendance", type_="check")
        if _check_constraint_exists("student_attendance", "ck_student_attendance_classes_attended_nonnegative"):
            op.drop_constraint("ck_student_attendance_classes_attended_nonnegative", "student_attendance", type_="check")
        op.rename_table("student_attendance", "attendance")

    if _table_exists("attendance") and not _index_exists("attendance", "ix_attendance_course_code"):
        op.create_index(op.f("ix_attendance_course_code"), "attendance", ["course_code"], unique=False)
    if _table_exists("attendance") and not _index_exists("attendance", "ix_attendance_id"):
        op.create_index(op.f("ix_attendance_id"), "attendance", ["id"], unique=False)
    if _table_exists("attendance") and not _index_exists("attendance", "ix_attendance_student_id"):
        op.create_index(op.f("ix_attendance_student_id"), "attendance", ["student_id"], unique=False)
    if _table_exists("schedule") and not _index_exists("schedule", "ix_schedule_course_code"):
        op.create_index(op.f("ix_schedule_course_code"), "schedule", ["course_code"], unique=False)
    if _table_exists("schedule") and not _index_exists("schedule", "ix_schedule_id"):
        op.create_index(op.f("ix_schedule_id"), "schedule", ["id"], unique=False)
    if _table_exists("schedule") and not _index_exists("schedule", "ix_schedule_student_id"):
        op.create_index(op.f("ix_schedule_student_id"), "schedule", ["student_id"], unique=False)

    if _index_exists("attendance", "ix_student_attendance_course_code"):
        op.drop_index("ix_student_attendance_course_code", table_name="attendance")
    if _index_exists("attendance", "ix_student_attendance_id"):
        op.drop_index("ix_student_attendance_id", table_name="attendance")
    if _index_exists("attendance", "ix_student_attendance_student_id"):
        op.drop_index("ix_student_attendance_student_id", table_name="attendance")
    if _index_exists("schedule", "ix_student_schedule_course_code"):
        op.drop_index("ix_student_schedule_course_code", table_name="schedule")
    if _index_exists("schedule", "ix_student_schedule_id"):
        op.drop_index("ix_student_schedule_id", table_name="schedule")
    if _index_exists("schedule", "ix_student_schedule_student_id"):
        op.drop_index("ix_student_schedule_student_id", table_name="schedule")

    if _table_exists("attendance") and not _check_constraint_exists("attendance", "ck_attendance_classes_attended_nonnegative"):
        op.create_check_constraint(
            "ck_attendance_classes_attended_nonnegative",
            "attendance",
            "classes_attended >= 0",
        )
    if _table_exists("attendance") and not _check_constraint_exists("attendance", "ck_attendance_total_classes_positive"):
        op.create_check_constraint(
            "ck_attendance_total_classes_positive",
            "attendance",
            "total_classes > 0",
        )
    if _table_exists("attendance") and not _check_constraint_exists("attendance", "ck_attendance_classes_attended_not_greater_than_total"):
        op.create_check_constraint(
            "ck_attendance_classes_attended_not_greater_than_total",
            "attendance",
            "classes_attended <= total_classes",
        )
    if _table_exists("schedule") and not _check_constraint_exists("schedule", "ck_schedule_grace_period_nonnegative"):
        op.create_check_constraint(
            "ck_schedule_grace_period_nonnegative",
            "schedule",
            "grace_period_minutes >= 0",
        )
