"""fix class numbering per faculty — make class names unique across faculty, not just department

Revision ID: ee11ff22aa33
Revises: ddee1122ff33
Create Date: 2026-05-26

Fixes the bug where creating a class in a new department within the same faculty
would restart numbering (AGR001, AGR002 again instead of AGR003, AGR004).

Changes:
  1. Renumber ALL existing class_batches so names are unique within each faculty.
     Ordering: by class batch id (creation order) within each faculty.
     A two-pass approach is used to avoid temporary constraint violations.
  2. Drop old unique constraint uq_class_batch_department_name (per-department).
  3. Add new unique constraint uq_class_batch_faculty_name (per-faculty).
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = "ee11ff22aa33"
down_revision: Union[str, None] = "ddee1122ff33"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()

    # ------------------------------------------------------------------ #
    # Step 1 – collect data                                               #
    # ------------------------------------------------------------------ #
    faculties = bind.execute(
        sa.text("SELECT id, code FROM faculties ORDER BY id")
    ).fetchall()

    # Map: faculty_id -> (prefix, [class_id, ...]) ordered by id
    faculty_class_map: dict[int, tuple[str, list[int]]] = {}
    all_class_ids: list[int] = []

    for faculty_id, faculty_code in faculties:
        prefix = (faculty_code or "CLS").strip().upper()
        rows = bind.execute(
            sa.text(
                "SELECT id FROM class_batches WHERE faculty_id = :fid ORDER BY id"
            ),
            {"fid": faculty_id},
        ).fetchall()
        class_ids = [r[0] for r in rows]
        faculty_class_map[faculty_id] = (prefix, class_ids)
        all_class_ids.extend(class_ids)

    if not all_class_ids:
        # No class batches yet — just swap the constraint and done.
        _swap_constraint(bind)
        return

    # ------------------------------------------------------------------ #
    # Step 2 – pass 1: rename everything to a guaranteed-unique temp name #
    # ------------------------------------------------------------------ #
    # Use the row's own id so the temp name is globally unique.
    for class_id in all_class_ids:
        bind.execute(
            sa.text("UPDATE class_batches SET name = :name WHERE id = :id"),
            {"name": f"__TMP{class_id}__", "id": class_id},
        )

    # ------------------------------------------------------------------ #
    # Step 3 – pass 2: assign final sequential names per faculty          #
    # ------------------------------------------------------------------ #
    for _faculty_id, (prefix, class_ids) in faculty_class_map.items():
        for seq, class_id in enumerate(class_ids, 1):
            width = max(3, len(str(len(class_ids))))
            new_name = f"{prefix}{seq:0{width}d}"
            bind.execute(
                sa.text("UPDATE class_batches SET name = :name WHERE id = :id"),
                {"name": new_name, "id": class_id},
            )

    # ------------------------------------------------------------------ #
    # Step 4 – swap unique constraint                                     #
    # ------------------------------------------------------------------ #
    _swap_constraint(bind)


def _swap_constraint(bind) -> None:  # type: ignore[no-untyped-def]
    dialect = bind.dialect.name

    if dialect == "mysql":
        # ------------------------------------------------------------------ #
        # MySQL: the old composite unique index (department_id, name) is      #
        # being used by MySQL to back the FK on class_batches.department_id.  #
        # We must create a plain index on department_id FIRST so MySQL has    #
        # something to back that FK with, otherwise DROP INDEX fails with     #
        # error 1553 "needed in a foreign key constraint".                    #
        # ------------------------------------------------------------------ #

        # 1. Create a standalone index on department_id (if not already there)
        idx_exists = bind.execute(
            sa.text(
                """
                SELECT COUNT(*) FROM information_schema.STATISTICS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME   = 'class_batches'
                  AND INDEX_NAME   = 'ix_class_batches_department_id'
                """
            )
        ).scalar()
        if not idx_exists:
            bind.execute(
                sa.text(
                    "CREATE INDEX ix_class_batches_department_id "
                    "ON class_batches (department_id)"
                )
            )

        # 2. Now it's safe to drop the old unique constraint
        exists = bind.execute(
            sa.text(
                """
                SELECT COUNT(*) FROM information_schema.TABLE_CONSTRAINTS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME   = 'class_batches'
                  AND CONSTRAINT_NAME = 'uq_class_batch_department_name'
                  AND CONSTRAINT_TYPE = 'UNIQUE'
                """
            )
        ).scalar()
        if exists:
            op.drop_constraint(
                "uq_class_batch_department_name", "class_batches", type_="unique"
            )
    else:
        # SQLite / other: attempt drop, ignore if missing
        try:
            op.drop_constraint(
                "uq_class_batch_department_name", "class_batches", type_="unique"
            )
        except Exception:
            pass

    # Add new faculty-scoped unique constraint
    op.create_unique_constraint(
        "uq_class_batch_faculty_name",
        "class_batches",
        ["faculty_id", "name"],
    )


def downgrade() -> None:
    op.drop_constraint("uq_class_batch_faculty_name", "class_batches", type_="unique")
    # Restore the old per-department unique constraint.
    # The standalone ix_class_batches_department_id index is left in place
    # (harmless, and avoids re-creating it if upgrade runs again).
    op.create_unique_constraint(
        "uq_class_batch_department_name",
        "class_batches",
        ["department_id", "name"],
    )
