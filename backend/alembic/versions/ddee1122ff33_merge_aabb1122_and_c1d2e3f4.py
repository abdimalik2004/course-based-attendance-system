"""merge aabb1122cc33 and c1d2e3f4a5b6 heads

Revision ID: ddee1122ff33
Revises: aabb1122cc33, c1d2e3f4a5b6
Create Date: 2026-05-24

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "ddee1122ff33"
down_revision: Union[str, Sequence[str], None] = ("aabb1122cc33", "c1d2e3f4a5b6")
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
