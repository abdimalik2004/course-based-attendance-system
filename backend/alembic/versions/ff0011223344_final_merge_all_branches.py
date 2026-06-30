"""final_merge_all_branches

Revision ID: ff0011223344
Revises: dd3344556677, ee11ff22aa33
Create Date: 2026-06-09

Merges the notifications+faculty branch (dd3344556677) and the class-numbering
branch (ee11ff22aa33) into a single linear head so every future migration has
one unambiguous parent.
"""
from __future__ import annotations

from alembic import op

revision = "ff0011223344"
down_revision = ("dd3344556677", "ee11ff22aa33")
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
