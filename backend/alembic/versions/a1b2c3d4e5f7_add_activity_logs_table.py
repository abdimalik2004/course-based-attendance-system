"""add activity_logs table

Revision ID: a1b2c3d4e5f7
Revises: 834188818dd6
Create Date: 2026-05-21

"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op


revision: str = 'a1b2c3d4e5f7'
down_revision: Union[str, Sequence[str], None] = '834188818dd6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'activity_logs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('username', sa.String(length=128), nullable=False),
        sa.Column('action', sa.String(length=255), nullable=False),
        sa.Column(
            'status',
            sa.Enum('Success', 'Failed', 'Pending', name='activity_log_status'),
            nullable=False,
        ),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index('ix_activity_logs_id', 'activity_logs', ['id'], unique=False)
    op.create_index('ix_activity_logs_user_id', 'activity_logs', ['user_id'], unique=False)
    op.create_index('ix_activity_logs_created_at', 'activity_logs', ['created_at'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_activity_logs_created_at', table_name='activity_logs')
    op.drop_index('ix_activity_logs_user_id', table_name='activity_logs')
    op.drop_index('ix_activity_logs_id', table_name='activity_logs')
    op.drop_table('activity_logs')
    op.execute("DROP TYPE IF EXISTS activity_log_status")
