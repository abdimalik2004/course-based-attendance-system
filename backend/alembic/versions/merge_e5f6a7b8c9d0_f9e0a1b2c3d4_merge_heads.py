"""merge_e5f6a7b8c9d0_f9e0a1b2c3d4_merge_heads

Revision ID: merge_e5f6a7b8c9d0_f9e0a1b2c3d4
Revises: 
Create Date: 2026-05-06 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "merge_e5f6a7b8c9d0_f9e0a1b2c3d4"
down_revision: Union[str, Sequence[str], None] = ("e5f6a7b8c9d0", "f9e0a1b2c3d4")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Merge heads: no schema changes.

    This revision intentionally does not alter the database schema. It exists
    to merge two parallel heads into a single migration head so Alembic can
    upgrade smoothly.
    """
    pass


def downgrade() -> None:
    """Downgrade not implemented for merge-only revision."""
    pass
