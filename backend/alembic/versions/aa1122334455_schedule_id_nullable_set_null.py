"""make attendance_sessions.schedule_id nullable with SET NULL

Revision ID: aa1122334455
Revises: ff0011223344
Create Date: 2026-06-20

When a CourseSchedule is deleted we want to KEEP all AttendanceSession rows
(and their AttendanceRecord rows) so historical attendance data is preserved.

Changes:
  1. Drop the unique constraint uq_schedule_session_occurrence (the NULL-equality
     semantics in SQL make it useless once schedule_id can be NULL anyway).
  2. Drop the existing CASCADE FK on schedule_id.
  3. Make schedule_id nullable.
  4. Re-add the FK with ON DELETE SET NULL so deleting a schedule simply
     orphans the session rows rather than deleting them.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "aa1122334455"
down_revision = "ff0011223344"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Use raw SQL to avoid relying on the auto-generated FK constraint name
    # that MySQL assigns (attendance_sessions_ibfk_N).  The information_schema
    # query finds it dynamically.
    conn = op.get_bind()

    # 1. Find and drop all FKs on attendance_sessions.schedule_id
    fk_rows = conn.execute(sa.text("""
        SELECT CONSTRAINT_NAME
        FROM information_schema.KEY_COLUMN_USAGE
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = 'attendance_sessions'
          AND COLUMN_NAME = 'schedule_id'
          AND REFERENCED_TABLE_NAME = 'course_schedules'
    """)).fetchall()

    for (fk_name,) in fk_rows:
        conn.execute(sa.text(
            f"ALTER TABLE attendance_sessions DROP FOREIGN KEY `{fk_name}`"
        ))

    # 2. Drop the unique constraint that included schedule_id
    #    (ignore error if it was already dropped by a prior run)
    try:
        conn.execute(sa.text(
            "ALTER TABLE attendance_sessions "
            "DROP INDEX uq_schedule_session_occurrence"
        ))
    except Exception:
        pass

    # 3. Make schedule_id nullable
    conn.execute(sa.text(
        "ALTER TABLE attendance_sessions "
        "MODIFY COLUMN schedule_id INT NULL"
    ))

    # 4. Re-add FK with SET NULL
    conn.execute(sa.text(
        "ALTER TABLE attendance_sessions "
        "ADD CONSTRAINT fk_att_sessions_schedule_id "
        "FOREIGN KEY (schedule_id) REFERENCES course_schedules(id) "
        "ON DELETE SET NULL"
    ))


def downgrade() -> None:
    conn = op.get_bind()

    # Remove all rows where schedule_id is NULL (can't make it NOT NULL otherwise)
    conn.execute(sa.text(
        "DELETE FROM attendance_sessions WHERE schedule_id IS NULL"
    ))

    # Drop the SET NULL FK
    try:
        conn.execute(sa.text(
            "ALTER TABLE attendance_sessions "
            "DROP FOREIGN KEY fk_att_sessions_schedule_id"
        ))
    except Exception:
        pass

    # Make schedule_id NOT NULL again
    conn.execute(sa.text(
        "ALTER TABLE attendance_sessions "
        "MODIFY COLUMN schedule_id INT NOT NULL"
    ))

    # Re-add original CASCADE FK
    conn.execute(sa.text(
        "ALTER TABLE attendance_sessions "
        "ADD CONSTRAINT fk_att_sessions_schedule_id_cascade "
        "FOREIGN KEY (schedule_id) REFERENCES course_schedules(id) "
        "ON DELETE CASCADE"
    ))

    # Re-add unique constraint
    conn.execute(sa.text(
        "ALTER TABLE attendance_sessions "
        "ADD CONSTRAINT uq_schedule_session_occurrence "
        "UNIQUE (schedule_id, session_date, start_time)"
    ))
