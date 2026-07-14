"""add excuse_requests table

Revision ID: dd4455667788
Revises: cc3344556677
Create Date: 2026-07-11 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "dd4455667788"
down_revision: Union[str, None] = "cc3344556677"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "excuse_requests",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("student_id", sa.Integer(), nullable=False),
        sa.Column("course_id", sa.Integer(), nullable=True),
        sa.Column("request_date", sa.Date(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column(
            "status",
            sa.String(length=20),
            nullable=False,
            server_default="PENDING",
        ),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(), nullable=True),
        sa.Column("reviewed_by", sa.Integer(), nullable=True),
        sa.ForeignKeyConstraint(["course_id"], ["courses.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["reviewed_by"], ["users.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["student_id"], ["students.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_excuse_requests_id", "excuse_requests", ["id"])
    op.create_index("ix_excuse_requests_student_id", "excuse_requests", ["student_id"])
    op.create_index("ix_excuse_requests_course_id", "excuse_requests", ["course_id"])
    op.create_index("ix_excuse_requests_created_at", "excuse_requests", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_excuse_requests_created_at", table_name="excuse_requests")
    op.drop_index("ix_excuse_requests_course_id", table_name="excuse_requests")
    op.drop_index("ix_excuse_requests_student_id", table_name="excuse_requests")
    op.drop_index("ix_excuse_requests_id", table_name="excuse_requests")
    op.drop_table("excuse_requests")
