from __future__ import annotations

from sqlalchemy import text

from app.db.models import Faculty, User
from app.db.session import SessionLocal


def reset_faculties_to_start_at_one() -> dict[str, int]:
    db = SessionLocal()
    try:
        faculty_users_deleted = (
            db.query(User)
            .filter(User.faculty_id.isnot(None))
            .delete(synchronize_session=False)
        )
        faculties_deleted = db.query(Faculty).delete(synchronize_session=False)
        db.commit()

        dialect_name = db.bind.dialect.name if db.bind is not None else ""
        if dialect_name == "mysql":
            db.execute(text("ALTER TABLE faculties AUTO_INCREMENT = 1"))
        elif dialect_name == "sqlite":
            db.execute(text("DELETE FROM sqlite_sequence WHERE name = 'faculties'"))
        db.commit()

        return {
            "faculty_users_deleted": faculty_users_deleted,
            "faculties_deleted": faculties_deleted,
        }
    finally:
        db.close()


if __name__ == "__main__":
    result = reset_faculties_to_start_at_one()
    print(
        "Faculty reset completed:",
        f"faculty_users_deleted={result['faculty_users_deleted']}",
        f"faculties_deleted={result['faculties_deleted']}",
    )
