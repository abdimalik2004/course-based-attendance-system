"""add_profile_image_url_to_users

Revision ID: ffff11112222
Revises: merge_e5f6a7b8c9d0_f9e0a1b2c3d4
Create Date: 2026-05-17 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "ffff11112222"
down_revision: Union[str, Sequence[str], None] = "merge_e5f6a7b8c9d0_f9e0a1b2c3d4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add a nullable text column to store profile image URL/path
    op.execute("ALTER TABLE users ADD COLUMN profile_image_url TEXT")


def downgrade() -> None:
    op.execute("ALTER TABLE users DROP COLUMN profile_image_url")
