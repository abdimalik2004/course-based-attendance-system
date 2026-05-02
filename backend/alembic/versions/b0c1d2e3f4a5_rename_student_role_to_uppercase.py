"""rename_student_role_to_uppercase

Revision ID: b0c1d2e3f4a5
Revises: a9b8c7d6e5f4
Create Date: 2026-05-02 00:00:00.000000

"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b0c1d2e3f4a5"
down_revision: Union[str, Sequence[str], None] = "a9b8c7d6e5f4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _role_exists(bind, name: str) -> bool:
    if bind.dialect.name in {"mysql", "mariadb"}:
        query = sa.text("SELECT 1 FROM roles WHERE BINARY name = :name LIMIT 1")
    else:
        query = sa.text("SELECT 1 FROM roles WHERE name = :name LIMIT 1")

    return (
        bind.execute(query, {"name": name}).first()
        is not None
    )


def upgrade() -> None:
    bind = op.get_bind()
    has_lowercase = _role_exists(bind, "Student")
    has_uppercase = _role_exists(bind, "STUDENT")

    if has_lowercase and has_uppercase:
        op.execute(sa.text("DELETE FROM roles WHERE name = 'Student'"))
    elif has_lowercase:
        op.execute(sa.text("UPDATE roles SET name = 'STUDENT' WHERE name = 'Student'"))
    elif not has_uppercase:
        op.execute(sa.text("INSERT INTO roles (name) VALUES ('STUDENT')"))


def downgrade() -> None:
    bind = op.get_bind()
    has_lowercase = _role_exists(bind, "Student")
    has_uppercase = _role_exists(bind, "STUDENT")

    if has_lowercase and has_uppercase:
        op.execute(sa.text("DELETE FROM roles WHERE name = 'STUDENT'"))
    elif has_uppercase:
        op.execute(sa.text("UPDATE roles SET name = 'Student' WHERE name = 'STUDENT'"))