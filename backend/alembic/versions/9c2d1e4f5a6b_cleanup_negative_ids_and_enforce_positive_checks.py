"""cleanup_negative_ids_and_enforce_positive_checks

Revision ID: 9c2d1e4f5a6b
Revises: 3a4b5c6d7e8f
Create Date: 2026-04-03 11:20:00.000000
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "9c2d1e4f5a6b"
down_revision: Union[str, Sequence[str], None] = "3a4b5c6d7e8f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


_ID_TABLES = (
    "roles",
    "organizational_units",
    "faculties",
    "departments",
    "class_batches",
    "users",
    "students",
    "teachers",
    "courses",
    "course_assignments",
    "enrollments",
    "course_schedules",
    "attendance_sessions",
    "attendance_records",
)


def _cleanup_negative_rows() -> None:
    # Child tables first to avoid FK violations
    op.execute(sa.text("""
        DELETE FROM attendance_records
        WHERE id < 0 OR student_id < 0 OR course_id < 0 OR session_id < 0
    """))

    op.execute(sa.text("""
        DELETE FROM attendance_sessions
        WHERE id < 0 OR course_id < 0 OR schedule_id < 0
    """))

    op.execute(sa.text("""
        DELETE FROM course_schedules
        WHERE id < 0 OR course_id < 0
    """))

    op.execute(sa.text("""
        DELETE FROM enrollments
        WHERE id < 0 OR student_id < 0 OR course_id < 0
    """))

    op.execute(sa.text("""
        DELETE FROM course_assignments
        WHERE id < 0 OR course_id < 0 OR teacher_id < 0
    """))

    op.execute(sa.text("""
        DELETE FROM students
        WHERE id < 0 OR faculty_id < 0 OR department_id < 0 OR class_batch_id < 0
    """))

    op.execute(sa.text("""
        DELETE FROM teachers
        WHERE id < 0 OR faculty_id < 0 OR department_id < 0 OR user_id < 0
    """))

    op.execute(sa.text("""
        DELETE FROM courses
        WHERE id < 0 OR class_batch_id < 0 OR faculty_id < 0
    """))

    op.execute(sa.text("""
        DELETE FROM class_batches
        WHERE id < 0 OR faculty_id < 0 OR department_id < 0
    """))

    op.execute(sa.text("""
        DELETE FROM departments
        WHERE id < 0 OR faculty_id < 0
    """))

    op.execute(sa.text("DELETE FROM user_role_links WHERE user_id < 0 OR role_id < 0"))
    op.execute(sa.text("DELETE FROM users WHERE id < 0 OR faculty_id < 0"))
    op.execute(sa.text("DELETE FROM faculties WHERE id < 0"))
    op.execute(sa.text("DELETE FROM organizational_units WHERE id < 0"))
    op.execute(sa.text("DELETE FROM roles WHERE id < 0"))


def _sync_sqlite_sequences() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "sqlite":
        return

    exists = bind.execute(
        sa.text("SELECT name FROM sqlite_master WHERE type='table' AND name='sqlite_sequence'")
    ).first()

    if exists is None:
        return

    for table_name in _ID_TABLES:
        bind.execute(
            sa.text(f"""
                UPDATE sqlite_sequence
                SET seq = (SELECT COALESCE(MAX(id), 0) FROM {table_name})
                WHERE name = :table_name
            """),
            {"table_name": table_name},
        )


def _set_autoincrement() -> None:
    for table_name in _ID_TABLES:
        with op.batch_alter_table(table_name) as batch_op:
            batch_op.alter_column(
                "id",
                existing_type=sa.Integer(),
                existing_nullable=False,
                autoincrement=True,
            )


def _create_positive_id_constraints() -> None:
    dialect = op.get_bind().dialect.name
    if dialect in {"mysql", "mariadb"}:
        return

    with op.batch_alter_table("faculties") as batch_op:
        batch_op.create_check_constraint("ck_faculties_id_positive", "id > 0")

    with op.batch_alter_table("departments") as batch_op:
        batch_op.create_check_constraint("ck_departments_id_positive", "id > 0")
        batch_op.create_check_constraint("ck_departments_faculty_id_positive", "faculty_id > 0")

    with op.batch_alter_table("class_batches") as batch_op:
        batch_op.create_check_constraint("ck_class_batches_id_positive", "id > 0")
        batch_op.create_check_constraint("ck_class_batches_faculty_id_positive", "faculty_id > 0")
        batch_op.create_check_constraint("ck_class_batches_department_id_positive", "department_id > 0")

    with op.batch_alter_table("courses") as batch_op:
        batch_op.create_check_constraint("ck_courses_id_positive", "id > 0")
        batch_op.create_check_constraint("ck_courses_class_batch_id_positive", "class_batch_id > 0")
        batch_op.create_check_constraint("ck_courses_faculty_id_positive", "faculty_id > 0")

    with op.batch_alter_table("students") as batch_op:
        batch_op.create_check_constraint("ck_students_id_positive", "id > 0")
        batch_op.create_check_constraint("ck_students_faculty_id_positive", "faculty_id > 0")
        batch_op.create_check_constraint("ck_students_department_id_positive", "department_id > 0")
        batch_op.create_check_constraint("ck_students_class_batch_id_positive", "class_batch_id > 0")

    with op.batch_alter_table("enrollments") as batch_op:
        batch_op.create_check_constraint("ck_enrollments_id_positive", "id > 0")
        batch_op.create_check_constraint("ck_enrollments_student_id_positive", "student_id > 0")
        batch_op.create_check_constraint("ck_enrollments_course_id_positive", "course_id > 0")

    with op.batch_alter_table("attendance_sessions") as batch_op:
        batch_op.create_check_constraint("ck_attendance_sessions_id_positive", "id > 0")
        batch_op.create_check_constraint("ck_attendance_sessions_course_id_positive", "course_id > 0")
        batch_op.create_check_constraint("ck_attendance_sessions_schedule_id_positive", "schedule_id > 0")

    with op.batch_alter_table("attendance_records") as batch_op:
        batch_op.create_check_constraint("ck_attendance_records_id_positive", "id > 0")
        batch_op.create_check_constraint("ck_attendance_records_student_id_positive", "student_id > 0")
        batch_op.create_check_constraint("ck_attendance_records_course_id_positive", "course_id > 0")
        batch_op.create_check_constraint("ck_attendance_records_session_id_positive", "session_id > 0")


def upgrade() -> None:
    _cleanup_negative_rows()
    _sync_sqlite_sequences()
    _set_autoincrement()
    _create_positive_id_constraints()


def downgrade() -> None:
    dialect = op.get_bind().dialect.name
    if dialect in {"mysql", "mariadb"}:
        return

    with op.batch_alter_table("attendance_records") as batch_op:
        batch_op.drop_constraint("ck_attendance_records_session_id_positive", type_="check")
        batch_op.drop_constraint("ck_attendance_records_course_id_positive", type_="check")
        batch_op.drop_constraint("ck_attendance_records_student_id_positive", type_="check")
        batch_op.drop_constraint("ck_attendance_records_id_positive", type_="check")

    with op.batch_alter_table("attendance_sessions") as batch_op:
        batch_op.drop_constraint("ck_attendance_sessions_schedule_id_positive", type_="check")
        batch_op.drop_constraint("ck_attendance_sessions_course_id_positive", type_="check")
        batch_op.drop_constraint("ck_attendance_sessions_id_positive", type_="check")

    with op.batch_alter_table("enrollments") as batch_op:
        batch_op.drop_constraint("ck_enrollments_course_id_positive", type_="check")
        batch_op.drop_constraint("ck_enrollments_student_id_positive", type_="check")
        batch_op.drop_constraint("ck_enrollments_id_positive", type_="check")

    with op.batch_alter_table("students") as batch_op:
        batch_op.drop_constraint("ck_students_class_batch_id_positive", type_="check")
        batch_op.drop_constraint("ck_students_department_id_positive", type_="check")
        batch_op.drop_constraint("ck_students_faculty_id_positive", type_="check")
        batch_op.drop_constraint("ck_students_id_positive", type_="check")

    with op.batch_alter_table("courses") as batch_op:
        batch_op.drop_constraint("ck_courses_faculty_id_positive", type_="check")
        batch_op.drop_constraint("ck_courses_class_batch_id_positive", type_="check")
        batch_op.drop_constraint("ck_courses_id_positive", type_="check")

    with op.batch_alter_table("class_batches") as batch_op:
        batch_op.drop_constraint("ck_class_batches_department_id_positive", type_="check")
        batch_op.drop_constraint("ck_class_batches_faculty_id_positive", type_="check")
        batch_op.drop_constraint("ck_class_batches_id_positive", type_="check")

    with op.batch_alter_table("departments") as batch_op:
        batch_op.drop_constraint("ck_departments_faculty_id_positive", type_="check")
        batch_op.drop_constraint("ck_departments_id_positive", type_="check")

    with op.batch_alter_table("faculties") as batch_op:
        batch_op.drop_constraint("ck_faculties_id_positive", type_="check")