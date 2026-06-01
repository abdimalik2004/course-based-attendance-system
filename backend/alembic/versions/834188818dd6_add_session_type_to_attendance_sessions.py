"""add session_type to attendance_sessions

Revision ID: 834188818dd6
Revises: ac4c9c49d46f
Create Date: 2026-05-19 10:35:55.595687

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision: str = '834188818dd6'
down_revision: Union[str, Sequence[str], None] = 'ac4c9c49d46f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'attendance_sessions',
        sa.Column(
            'session_type',
            sa.Enum('Lecture', 'Lab', 'Tutorial', name='session_type'),
            nullable=False,
            server_default=sa.text("'Lecture'"),
        ),
    )


def downgrade() -> None:
    op.drop_column('attendance_sessions', 'session_type')
