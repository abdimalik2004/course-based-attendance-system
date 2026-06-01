"""replace attendance_sessions instructor_id with teacher_id/admin_id

Revision ID: ac4c9c49d46f
Revises: 1111aabbccdd
Create Date: 2026-05-19 10:24:06.068744

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'ac4c9c49d46f'
down_revision: Union[str, Sequence[str], None] = '1111aabbccdd'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'attendance_sessions',
        sa.Column('teacher_id', sa.Integer(), nullable=True),
    )
    op.add_column(
        'attendance_sessions',
        sa.Column('admin_id', sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        'fk_attendance_sessions_teacher_id_teachers',
        'attendance_sessions',
        'teachers',
        ['teacher_id'],
        ['id'],
        ondelete='SET NULL',
    )
    op.create_foreign_key(
        'fk_attendance_sessions_admin_id_users',
        'attendance_sessions',
        'users',
        ['admin_id'],
        ['id'],
        ondelete='SET NULL',
    )
    op.drop_constraint(
        'fk_attendance_sessions_instructor_id_users',
        'attendance_sessions',
        type_='foreignkey',
    )
    op.drop_column('attendance_sessions', 'instructor_id')


def downgrade() -> None:
    op.add_column(
        'attendance_sessions',
        sa.Column('instructor_id', sa.Integer(), nullable=True),
    )
    op.create_foreign_key(
        'fk_attendance_sessions_instructor_id_users',
        'attendance_sessions',
        'users',
        ['instructor_id'],
        ['id'],
        ondelete='SET NULL',
    )
    op.drop_constraint(
        'fk_attendance_sessions_admin_id_users',
        'attendance_sessions',
        type_='foreignkey',
    )
    op.drop_constraint(
        'fk_attendance_sessions_teacher_id_teachers',
        'attendance_sessions',
        type_='foreignkey',
    )
    op.drop_column('attendance_sessions', 'admin_id')
    op.drop_column('attendance_sessions', 'teacher_id')
