"""add date_of_birth phone email to students

Revision ID: bb2233445566
Revises: aa1122334455
Create Date: 2026-07-04

Adds three optional personal-info columns to the students table:
  - date_of_birth  DATE
  - phone          VARCHAR(30)
  - email          VARCHAR(180)
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "bb2233445566"
down_revision = "aa1122334455"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    conn.execute(sa.text(
        "ALTER TABLE students "
        "ADD COLUMN date_of_birth DATE NULL, "
        "ADD COLUMN phone VARCHAR(30) NULL, "
        "ADD COLUMN email VARCHAR(180) NULL"
    ))


def downgrade() -> None:
    conn = op.get_bind()
    conn.execute(sa.text(
        "ALTER TABLE students "
        "DROP COLUMN date_of_birth, "
        "DROP COLUMN phone, "
        "DROP COLUMN email"
    ))
