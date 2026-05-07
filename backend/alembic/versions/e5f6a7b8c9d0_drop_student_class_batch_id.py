"""drop_student_class_batch_id

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-05-04 17:10:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e5f6a7b8c9d0"
down_revision: Union[str, Sequence[str], None] = "d4e5f6a7b8c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _drop_student_class_batch_fk() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    for foreign_key in inspector.get_foreign_keys("students"):
        if foreign_key.get("constrained_columns") == ["class_batch_id"] and foreign_key.get("name"):
            op.drop_constraint(foreign_key["name"], "students", type_="foreignkey")
            return


def upgrade() -> None:
    _drop_student_class_batch_fk()
    op.drop_column("students", "class_batch_id")


def downgrade() -> None:
    op.add_column(
        "students",
        sa.Column("class_batch_id", sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        "fk_students_class_batch_id_class_batches",
        "students",
        "class_batches",
        ["class_batch_id"],
        ["id"],
        ondelete="CASCADE",
    )