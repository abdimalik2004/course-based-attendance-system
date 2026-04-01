from __future__ import annotations

from app.core.security import get_password_hash, pwd_context, verify_password
from app.db.models import User
from app.db.session import SessionLocal


SEEDED_PASSWORDS: dict[str, str] = {
    "academia": "academia123",
    "facultyadmin": "faculty123",
    "teacher1": "teacher123",
}


def migrate_seeded_password_hashes() -> dict[str, int]:
    db = SessionLocal()
    migrated = 0
    skipped = 0
    not_seeded = 0

    try:
        users = db.query(User).all()
        for user in users:
            plain = SEEDED_PASSWORDS.get(user.username)
            if plain is None:
                not_seeded += 1
                continue

            if not pwd_context.needs_update(user.hashed_password):
                skipped += 1
                continue

            if not verify_password(plain, user.hashed_password):
                skipped += 1
                continue

            user.hashed_password = get_password_hash(plain)
            db.add(user)
            migrated += 1

        db.commit()
        return {"migrated": migrated, "skipped": skipped, "not_seeded": not_seeded}
    finally:
        db.close()


if __name__ == "__main__":
    result = migrate_seeded_password_hashes()
    print(
        "Password hash migration completed:",
        f"migrated={result['migrated']}",
        f"skipped={result['skipped']}",
        f"not_seeded={result['not_seeded']}",
    )
