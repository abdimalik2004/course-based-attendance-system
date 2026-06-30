"""merge_all_heads

Revision ID: dd3344556677
Revises: cc2233445566, zz0000000001, zz0000000002
Create Date: 2026-06-09

Merge migration — unifies all outstanding heads into a single linear chain.
"""
from __future__ import annotations

from alembic import op

revision = "dd3344556677"
down_revision = ("cc2233445566", "zz0000000001", "zz0000000002")
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
