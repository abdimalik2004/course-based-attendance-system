"""add_departments

Revision ID: 5b2f9c7a6d11
Revises: c9f3f2d1a4b1
Create Date: 2026-03-15 18:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "5b2f9c7a6d11"
down_revision: Union[str, Sequence[str], None] = "c9f3f2d1a4b1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _default_department(name: str, code: str) -> tuple[str, str]:
    normalized_name = (name or "").strip().upper()
    normalized_code = (code or "").strip().upper()

    if normalized_code in {"FCS", "CIS"} or "COMPUTER SCIENCE" in normalized_name:
        return "Department of Information Technology", "IT"
    if normalized_code in {"ENG", "FOE"} or "ENGINEERING" in normalized_name:
        return "Department of Architecture", "ARCH"
    return "General Department", "GEN"


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    table_names = set(inspector.get_table_names())

    if "departments" not in table_names:
        op.create_table(
            "departments",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("faculty_id", sa.Integer(), nullable=False),
            sa.Column("name", sa.String(length=150), nullable=False),
            sa.Column("code", sa.String(length=30), nullable=False),
            sa.Column("created_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
            sa.ForeignKeyConstraint(["faculty_id"], ["faculties.id"], ondelete="CASCADE"),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("faculty_id", "code", name="uq_department_faculty_code"),
            sa.UniqueConstraint("faculty_id", "name", name="uq_department_faculty_name"),
        )
        table_names.add("departments")

    existing_department_indexes = {index["name"] for index in inspector.get_indexes("departments")}
    if op.f("ix_departments_id") not in existing_department_indexes:
        op.create_index(op.f("ix_departments_id"), "departments", ["id"], unique=False)

    existing_class_batch_columns = {column["name"] for column in inspector.get_columns("class_batches")}
    if "department_id" not in existing_class_batch_columns:
        op.add_column("class_batches", sa.Column("department_id", sa.Integer(), nullable=True))

    existing_student_columns = {column["name"] for column in inspector.get_columns("students")}
    if "department_id" not in existing_student_columns:
        op.add_column("students", sa.Column("department_id", sa.Integer(), nullable=True))

    existing_teacher_columns = {column["name"] for column in inspector.get_columns("teachers")}
    if "department_id" not in existing_teacher_columns:
        op.add_column("teachers", sa.Column("department_id", sa.Integer(), nullable=True))

    metadata = sa.MetaData()

    faculties = sa.Table(
        "faculties",
        metadata,
        sa.Column("id", sa.Integer()),
        sa.Column("name", sa.String(length=150)),
        sa.Column("code", sa.String(length=30)),
    )
    departments = sa.Table(
        "departments",
        metadata,
        sa.Column("id", sa.Integer()),
        sa.Column("faculty_id", sa.Integer()),
        sa.Column("name", sa.String(length=150)),
        sa.Column("code", sa.String(length=30)),
    )
    class_batches = sa.Table(
        "class_batches",
        metadata,
        sa.Column("id", sa.Integer()),
        sa.Column("faculty_id", sa.Integer()),
        sa.Column("department_id", sa.Integer()),
    )
    students = sa.Table(
        "students",
        metadata,
        sa.Column("id", sa.Integer()),
        sa.Column("faculty_id", sa.Integer()),
        sa.Column("class_batch_id", sa.Integer()),
        sa.Column("department_id", sa.Integer()),
    )
    teachers = sa.Table(
        "teachers",
        metadata,
        sa.Column("id", sa.Integer()),
        sa.Column("faculty_id", sa.Integer()),
        sa.Column("department_id", sa.Integer()),
    )

    department_ids_by_faculty: dict[int, int] = {}
    faculty_rows = bind.execute(sa.select(faculties.c.id, faculties.c.name, faculties.c.code)).all()
    for faculty_id, faculty_name, faculty_code in faculty_rows:
        department_name, department_code = _default_department(faculty_name, faculty_code)
        existing_department_id = bind.execute(
            sa.select(departments.c.id).where(
                departments.c.faculty_id == faculty_id,
                departments.c.code == department_code,
            )
        ).scalar_one_or_none()
        if existing_department_id is None:
            bind.execute(
                sa.insert(departments).values(
                    faculty_id=faculty_id,
                    name=department_name,
                    code=department_code,
                )
            )
            existing_department_id = bind.execute(
                sa.select(departments.c.id).where(
                    departments.c.faculty_id == faculty_id,
                    departments.c.code == department_code,
                )
            ).scalar_one()
        department_ids_by_faculty[faculty_id] = existing_department_id

    class_batch_rows = bind.execute(
        sa.select(class_batches.c.id, class_batches.c.faculty_id, class_batches.c.department_id)
    ).all()
    class_batch_department_by_id: dict[int, int] = {}
    for faculty_id, department_id in department_ids_by_faculty.items():
        bind.execute(
            sa.update(class_batches)
            .where(class_batches.c.faculty_id == faculty_id, class_batches.c.department_id.is_(None))
            .values(department_id=department_id)
        )

    refreshed_class_batch_rows = bind.execute(
        sa.select(class_batches.c.id, class_batches.c.department_id)
    ).all()
    for class_batch_id, department_id in refreshed_class_batch_rows:
        if department_id is not None:
            class_batch_department_by_id[class_batch_id] = department_id

    fallback_department_id = next(iter(department_ids_by_faculty.values()), None)

    student_rows = bind.execute(
        sa.select(students.c.id, students.c.faculty_id, students.c.class_batch_id, students.c.department_id)
    ).all()
    for student_id, faculty_id, class_batch_id, department_id in student_rows:
        if department_id is not None:
            continue
        resolved_department_id = class_batch_department_by_id.get(class_batch_id)
        if resolved_department_id is None:
            resolved_department_id = department_ids_by_faculty.get(faculty_id, fallback_department_id)
        if resolved_department_id is None:
            continue
        bind.execute(
            sa.update(students)
            .where(students.c.id == student_id)
            .values(department_id=resolved_department_id)
        )

    teacher_rows = bind.execute(
        sa.select(teachers.c.id, teachers.c.faculty_id, teachers.c.department_id)
    ).all()
    for teacher_id, faculty_id, department_id in teacher_rows:
        if department_id is not None:
            continue
        resolved_department_id = department_ids_by_faculty.get(faculty_id, fallback_department_id)
        if resolved_department_id is None:
            continue
        bind.execute(
            sa.update(teachers)
            .where(teachers.c.id == teacher_id)
            .values(department_id=resolved_department_id)
        )

    with op.batch_alter_table("class_batches") as batch_op:
        batch_op.create_foreign_key(
            "fk_class_batches_department_id_departments",
            "departments",
            ["department_id"],
            ["id"],
            ondelete="CASCADE",
        )
        batch_op.alter_column("department_id", existing_type=sa.Integer(), nullable=False)
        batch_op.create_unique_constraint("uq_class_batch_department_name", ["department_id", "name"])

    with op.batch_alter_table("students") as batch_op:
        batch_op.create_foreign_key(
            "fk_students_department_id_departments",
            "departments",
            ["department_id"],
            ["id"],
            ondelete="CASCADE",
        )
        batch_op.alter_column("department_id", existing_type=sa.Integer(), nullable=False)

    with op.batch_alter_table("teachers") as batch_op:
        batch_op.create_foreign_key(
            "fk_teachers_department_id_departments",
            "departments",
            ["department_id"],
            ["id"],
            ondelete="CASCADE",
        )
        batch_op.alter_column("department_id", existing_type=sa.Integer(), nullable=False)


def downgrade() -> None:
    with op.batch_alter_table("teachers") as batch_op:
        batch_op.drop_constraint("fk_teachers_department_id_departments", type_="foreignkey")
        batch_op.drop_column("department_id")

    with op.batch_alter_table("students") as batch_op:
        batch_op.drop_constraint("fk_students_department_id_departments", type_="foreignkey")
        batch_op.drop_column("department_id")

    with op.batch_alter_table("class_batches") as batch_op:
        batch_op.drop_constraint("uq_class_batch_department_name", type_="unique")
        batch_op.drop_constraint("fk_class_batches_department_id_departments", type_="foreignkey")
        batch_op.drop_column("department_id")
        batch_op.create_unique_constraint("uq_class_batch_faculty_name", ["faculty_id", "name"])

    op.drop_index(op.f("ix_departments_id"), table_name="departments")
    op.drop_table("departments")