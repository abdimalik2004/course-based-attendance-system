"""add_notifications_table

Revision ID: a1b2c3d4e5f6
Revises: ffff11112222
Create Date: 2026-06-09

"""
from __future__ import annotations
from alembic import op
import sqlalchemy as sa

revision = 'cc1122334455'
down_revision = 'ffff11112222'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        'notifications',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True, index=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('title', sa.String(200), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('type', sa.String(20), nullable=False, server_default='info'),
        sa.Column('is_read', sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column('link', sa.String(300), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False, index=True),
    )


def downgrade() -> None:
    op.drop_table('notifications')
