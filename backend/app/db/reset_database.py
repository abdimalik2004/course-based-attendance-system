from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import inspect, text
from sqlalchemy.orm import Session

from app.core.security import get_password_hash
from app.db.models import Base, Role, User
from app.db.session import SessionLocal


@dataclass(frozen=True)
class SystemUserSeed:
    username: str
    password: str
    email: str
    role_name: str


SYSTEM_ROLE_NAMES: tuple[str, ...] = ("SUPER_ADMIN", "ACADEMIA", "FACULTY", "HR", "ADMISSIONS", "TEACHER")
SYSTEM_USERS: tuple[SystemUserSeed, ...] = (
    SystemUserSeed(
        username="admin",
        password="admin",
        email="admin@university.edu",
        role_name="SUPER_ADMIN",
    ),
    SystemUserSeed(
        username="academia",
        password="academia123",
        email="academia@university.edu",
        role_name="ACADEMIA",
    ),
    SystemUserSeed(
        username="hr",
        password="hr123",
        email="hr@university.edu",
        role_name="HR",
    ),
    SystemUserSeed(
        username="admission",
        password="admission123",
        email="admission@university.edu",
        role_name="ADMISSIONS",
    ),
)


def _base_table_names(db: Session) -> list[str]:
    inspector = inspect(db.bind)
    existing = set(inspector.get_table_names())
    return [table.name for table in Base.metadata.sorted_tables if table.name in existing]


def _clear_all_data(db: Session) -> list[str]:
    table_names = _base_table_names(db)
    dialect = db.bind.dialect.name if db.bind is not None else ""

    if dialect in {"mysql", "mariadb"}:
        db.execute(text("SET FOREIGN_KEY_CHECKS = 0"))
        try:
            for table_name in table_names:
                db.execute(text(f"TRUNCATE TABLE `{table_name}`"))
        finally:
            db.execute(text("SET FOREIGN_KEY_CHECKS = 1"))
        return table_names

    if dialect == "sqlite":
        db.execute(text("PRAGMA foreign_keys = OFF"))
        try:
            for table_name in reversed(table_names):
                db.execute(text(f'DELETE FROM "{table_name}"'))
            for table_name in table_names:
                db.execute(
                    text("DELETE FROM sqlite_sequence WHERE name = :table_name"),
                    {"table_name": table_name},
                )
        finally:
            db.execute(text("PRAGMA foreign_keys = ON"))
        return table_names

    # PostgreSQL and others: delete all rows then restart identity where supported.
    for table_name in reversed(table_names):
        db.execute(text(f'DELETE FROM "{table_name}"'))

    if dialect == "postgresql":
        for table_name in table_names:
            db.execute(text(f'ALTER SEQUENCE IF EXISTS "{table_name}_id_seq" RESTART WITH 1'))

    return table_names


def _seed_system_accounts(db: Session) -> dict[str, int]:
    roles: dict[str, Role] = {}
    for role_name in SYSTEM_ROLE_NAMES:
        role = Role(name=role_name)
        db.add(role)
        db.flush()
        roles[role_name] = role

    user_ids: dict[str, int] = {}
    for seed_user in SYSTEM_USERS:
        user = User(
            username=seed_user.username,
            email=seed_user.email,
            hashed_password=get_password_hash(seed_user.password),
            is_active=True,
            faculty_id=None,
        )
        user.roles = [roles[seed_user.role_name]]
        db.add(user)
        db.flush()
        user_ids[user.username] = user.id

    return user_ids


def reset_database_to_clean_state() -> dict[str, object]:
    db = SessionLocal()
    try:
        cleared_tables = _clear_all_data(db)
        seeded_user_ids = _seed_system_accounts(db)
        db.commit()

        total_users = db.query(User).count()
        total_roles = db.query(Role).count()

        return {
            "cleared_tables": len(cleared_tables),
            "users_created": total_users,
            "roles_created": total_roles,
            "seeded_user_ids": seeded_user_ids,
        }
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    summary = reset_database_to_clean_state()
    print(
        "Database reset complete:",
        f"cleared_tables={summary['cleared_tables']}",
        f"roles_created={summary['roles_created']}",
        f"users_created={summary['users_created']}",
        f"seeded_user_ids={summary['seeded_user_ids']}",
    )
