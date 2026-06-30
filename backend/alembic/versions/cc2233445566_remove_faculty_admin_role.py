"""remove_faculty_admin_role

Revision ID: cc2233445566
Revises: cc1122334455
Create Date: 2026-06-09

Permanently deletes the FACULTY_ADMIN role and re-assigns any users that had
it to the FACULTY role, so no user account is left role-less.
"""
from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "cc2233445566"
down_revision = "cc1122334455"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()

    # 1. Get the id of FACULTY_ADMIN role
    fa_row = conn.execute(
        sa.text("SELECT id FROM roles WHERE name = 'FACULTY_ADMIN'")
    ).fetchone()
    if fa_row is None:
        # Already gone — nothing to do
        return
    fa_id = fa_row[0]

    # 2. Get (or create) the FACULTY role
    f_row = conn.execute(
        sa.text("SELECT id FROM roles WHERE name = 'FACULTY'")
    ).fetchone()
    if f_row is None:
        conn.execute(sa.text("INSERT INTO roles (name) VALUES ('FACULTY')"))
        f_row = conn.execute(
            sa.text("SELECT id FROM roles WHERE name = 'FACULTY'")
        ).fetchone()
    f_id = f_row[0]

    # 3. For every user that has FACULTY_ADMIN, assign them FACULTY if not already
    fa_users = conn.execute(
        sa.text("SELECT user_id FROM user_role_links WHERE role_id = :fa_id"),
        {"fa_id": fa_id},
    ).fetchall()

    for (user_id,) in fa_users:
        already = conn.execute(
            sa.text(
                "SELECT 1 FROM user_role_links WHERE user_id = :uid AND role_id = :fid"
            ),
            {"uid": user_id, "fid": f_id},
        ).fetchone()
        if not already:
            conn.execute(
                sa.text(
                    "INSERT INTO user_role_links (user_id, role_id) VALUES (:uid, :fid)"
                ),
                {"uid": user_id, "fid": f_id},
            )

    # 4. Remove all FACULTY_ADMIN role links
    conn.execute(
        sa.text("DELETE FROM user_role_links WHERE role_id = :fa_id"),
        {"fa_id": fa_id},
    )

    # 5. Delete the FACULTY_ADMIN role itself
    conn.execute(
        sa.text("DELETE FROM roles WHERE id = :fa_id"),
        {"fa_id": fa_id},
    )


def downgrade() -> None:
    conn = op.get_bind()
    exists = conn.execute(
        sa.text("SELECT 1 FROM roles WHERE name = 'FACULTY_ADMIN'")
    ).fetchone()
    if not exists:
        conn.execute(sa.text("INSERT INTO roles (name) VALUES ('FACULTY_ADMIN')"))
