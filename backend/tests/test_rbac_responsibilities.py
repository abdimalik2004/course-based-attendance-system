from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.security import get_current_user
from app.db.models import Base, ClassBatch, Department, Faculty, Role
from app.db.role_scoped import get_role_scoped_db
from app.db.session import get_db
from app.routers import auth, courses, schedules, students, teachers


class _Role:
    def __init__(self, name: str):
        self.name = name


class _DummyUser:
    def __init__(self, roles: list[str], faculty_id: int | None = None):
        self.roles = [_Role(role) for role in roles]
        self.faculty_id = faculty_id
        self.is_active = True


def _build_app_and_db():
    engine = create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    db = TestingSession()
    for role_name in ("SUPER_ADMIN", "ACADEMIA", "FACULTY", "FACULTY_ADMIN", "HR", "ADMISSIONS", "TEACHER", "STUDENT"):
        db.add(Role(name=role_name))

    faculty = Faculty(name="Faculty of Engineering", code="ENG")
    db.add(faculty)
    db.flush()

    department = Department(faculty_id=faculty.id, name="Architecture", code="ARCH")
    db.add(department)
    db.flush()

    class_batch = ClassBatch(faculty_id=faculty.id, department_id=department.id, name="ENG2201", year=2026)
    db.add(class_batch)
    db.commit()

    app = FastAPI()
    app.include_router(auth.router)
    app.include_router(courses.router)
    app.include_router(schedules.router)
    app.include_router(students.router)
    app.include_router(teachers.router)

    current = {"roles": ["ACADEMIA"], "faculty_id": faculty.id}

    def _override_db():
        try:
            yield db
        finally:
            pass

    def _override_current_user() -> _DummyUser:
        return _DummyUser(current["roles"], current.get("faculty_id"))

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_role_scoped_db] = _override_db
    app.dependency_overrides[get_current_user] = _override_current_user

    return TestClient(app), db, current, faculty, department, class_batch


def test_academia_can_create_course_but_faculty_cannot():
    client, db, current, _, _, class_batch = _build_app_and_db()

    current["roles"] = ["FACULTY"]
    denied = client.post("/courses", json={"faculty_id": 1, "title": "Thermodynamics"})
    assert denied.status_code == 403

    current["roles"] = ["ACADEMIA"]
    allowed = client.post("/courses", json={"faculty_id": class_batch.faculty_id, "title": "Thermodynamics"})
    assert allowed.status_code == 200

    db.close()


def test_faculty_can_schedule_course():
    client, db, current, _, _, class_batch = _build_app_and_db()

    current["roles"] = ["ACADEMIA"]
    create_course = client.post("/courses", json={"faculty_id": class_batch.faculty_id, "title": "Fluid Mechanics"})
    assert create_course.status_code == 200
    course_id = create_course.json()["id"]

    current["roles"] = ["FACULTY"]
    create_schedule = client.post(
        "/schedules",
        json={
            "course_id": course_id,
            "weekday": ["sat", "mon"],
            "start_time": "08:00:00",
            "end_time": "10:00:00",
            "grace_period_minutes": 10,
        },
    )
    assert create_schedule.status_code == 200

    db.close()


def test_hr_manages_teachers_and_admissions_manages_students():
    client, db, current, faculty, department, class_batch = _build_app_and_db()

    current["roles"] = ["ADMISSIONS"]
    denied_teacher = client.post(
        "/teachers",
        json={
            "full_name": "Dr. Role Check",
            "faculty_id": faculty.id,
            "department_id": department.id,
        },
    )
    assert denied_teacher.status_code == 403

    current["roles"] = ["HR"]
    allowed_teacher = client.post(
        "/teachers",
        json={
            "full_name": "Dr. Role Check",
            "role": "Assistant Professor",
            "status": "Inactive",
            "faculty_id": faculty.id,
            "department_id": department.id,
        },
    )
    assert allowed_teacher.status_code == 200

    current["roles"] = ["HR"]
    denied_student = client.post(
        "/students",
        json={
            "full_name": "Student Role Check",
            "faculty_id": faculty.id,
            "department_id": department.id,
        },
    )
    assert denied_student.status_code == 403

    current["roles"] = ["ADMISSIONS"]
    allowed_student = client.post(
        "/students",
        json={
            "full_name": "Student Role Check",
            "faculty_id": faculty.id,
            "department_id": department.id,
        },
    )
    assert allowed_student.status_code == 200

    db.close()


def test_super_admin_can_list_and_create_roles():
    client, db, current, _, _, _ = _build_app_and_db()

    current["roles"] = ["SUPER_ADMIN"]

    created = client.post("/auth/roles", json={"name": "LIBRARY"})
    assert created.status_code == 201
    assert created.json()["name"] == "LIBRARY"

    listed = client.get("/auth/roles")
    assert listed.status_code == 200
    assert any(role["name"] == "LIBRARY" for role in listed.json())

    db.close()
