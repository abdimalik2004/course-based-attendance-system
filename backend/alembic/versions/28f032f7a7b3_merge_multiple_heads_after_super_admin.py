"""merge_multiple_heads_after_super_admin

Revision ID: 28f032f7a7b3
Revises: 9c2d1e4f5a6b, a1b2c3d4e5f6, e8f9a0b1c2d3
Create Date: 2026-04-08 12:35:05.779238

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '28f032f7a7b3'
down_revision: Union[str, Sequence[str], None] = ('9c2d1e4f5a6b', 'a1b2c3d4e5f6', 'e8f9a0b1c2d3')
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
