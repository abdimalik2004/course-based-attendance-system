"""add_excused_to_attendancestatus

Revision ID: c1d2e3f4a5b6
Revises: 4c5d6e7f8a90
Create Date: 2026-05-23 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "c1d2e3f4a5b6"
down_revision: Union[str, Sequence[str], None] = "c3d4e5f6a7b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_OLD_ENUM = sa.Enum("PRESENT", "LATE", "ABSENT", name="attendancestatus")
_NEW_ENUM = sa.Enum("PRESENT", "LATE", "ABSENT", "EXCUSED", name="attendancestatus")


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    if dialect == "mysql":
        # MySQL supports modifying ENUM columns directly
        op.execute(
            sa.text(
                "ALTER TABLE attendance_records "
                "MODIFY COLUMN status ENUM('PRESENT','LATE','ABSENT','EXCUSED') NOT NULL"
            )
        )
    else:
        # SQLite (and others): use batch mode which recreates the table
        # This replaces the CHECK constraint with one that includes EXCUSED
        with op.batch_alter_table("attendance_records", recreate="always") as batch_op:
            batch_op.alter_column(
                "status",
                existing_type=_OLD_ENUM,
                type_=_NEW_ENUM,
                existing_nullable=False,
            )


def downgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    # Before downgrading, convert any EXCUSED records back to ABSENT
    op.execute(
        sa.text(
            "UPDATE attendance_records SET status = 'ABSENT' WHERE status = 'EXCUSED'"
        )
    )

    if dialect == "mysql":
        op.execute(
            sa.text(
                "ALTER TABLE attendance_records "
                "MODIFY COLUMN status ENUM('PRESENT','LATE','ABSENT') NOT NULL"
            )
        )
    else:
        with op.batch_alter_table("attendance_records", recreate="always") as batch_op:
            batch_op.alter_column(
                "status",
                existing_type=_NEW_ENUM,
                type_=_OLD_ENUM,
                existing_nullable=False,
            )
