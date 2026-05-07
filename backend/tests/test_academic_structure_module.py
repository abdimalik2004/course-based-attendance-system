from __future__ import annotations

from datetime import date

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.security import get_current_user
from app.db.models import AcademicYear, Base, ClassBatch, Course, Department, Faculty, Role
from app.db.role_scoped import get_role_scoped_db
from app.db.session import get_db
from app.routers import academic_structure


class _Role:
    def __init__(self, name: str):
        self.name = name


class _DummyUser:
    def __init__(self, roles: list[str], faculty_id: int | None = None):
        self.roles = [_Role(role) for role in roles]
        self.faculty_id = faculty_id
        self.id = 1
        self.is_active = True


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    db = TestingSession()
    try:
        for role_name in ("SUPER_ADMIN", "ACADEMIA", "FACULTY", "FACULTY_ADMIN", "HR", "ADMISSIONS", "TEACHER"):
            db.add(Role(name=role_name))
        db.commit()
        yield db
    finally:
        db.close()


@pytest.fixture()
def client(db_session):
    app = FastAPI()
    app.include_router(academic_structure.router)

    def _override_db():
        try:
            yield db_session
        finally:
            pass

    current_roles = {"roles": ["ACADEMIA"], "faculty_id": None}

    def _override_current_user() -> _DummyUser:
        return _DummyUser(current_roles["roles"], current_roles.get("faculty_id"))

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_role_scoped_db] = _override_db
    app.dependency_overrides[get_current_user] = _override_current_user

    test_client = TestClient(app)
    return test_client, current_roles


def _seed_faculty_graph(db_session):
    faculty = Faculty(name="Faculty of Computer Science", code="FCS")
    db_session.add(faculty)
    db_session.flush()

    department = Department(faculty_id=faculty.id, name="Department of Software Engineering", code="SE")
    db_session.add(department)
    db_session.flush()

    class_batch = ClassBatch(faculty_id=faculty.id, department_id=department.id, name="CIS2201", year=2026)
    db_session.add(class_batch)
    db_session.flush()

    course = Course(faculty_id=faculty.id, department_id=department.id, code="CSC401", title="AI")
    db_session.add(course)
    db_session.commit()
    return faculty, department, class_batch, course


def test_create_academic_year_and_block_second_active(client):
    api, current = client
    current["roles"] = ["ACADEMIA"]

    response = api.post(
        "/academic-structure/academic-years",
        json={
            "academic_year": "2025-2026",
            "term_name": "Semester 1",
            "start_date": "2025-09-01",
            "end_date": "2026-01-15",
            "status": "active",
        },
    )

    assert response.status_code == 200
    assert response.json()["academic_year"] == "2025-2026"
    assert response.json()["status"] == "active"

    conflict = api.post(
        "/academic-structure/academic-years",
        json={
            "academic_year": "2026-2027",
            "term_name": "Semester 1",
            "start_date": "2026-09-01",
            "end_date": "2027-01-15",
            "status": "active",
        },
    )

    assert conflict.status_code == 409


def test_course_semester_assignment_duplicate_and_filtering(client, db_session):
    faculty, department, _, course = _seed_faculty_graph(db_session)
    academic_year = AcademicYear(
        academic_year="2025-2026",
        term_name="Semester 1",
        start_date=date(2025, 9, 1),
        end_date=date(2026, 1, 15),
        status="draft",
    )
    db_session.add(academic_year)
    db_session.commit()

    api, current = client
    current["roles"] = ["ACADEMIA"]

    response = api.post(
        "/academic-structure/course-semester-assignments",
        json={
            "course_id": course.id,
            "faculty_id": faculty.id,
            "department_id": department.id,
            "semester": 1,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["course_id"] == course.id
    assert body["semester"] == 1

    duplicate = api.post(
        "/academic-structure/course-semester-assignments",
        json={
            "course_id": course.id,
            "faculty_id": faculty.id,
            "department_id": department.id,
            "semester": 1,
        },
    )
    assert duplicate.status_code == 409

    filtered = api.get(
        "/academic-structure/course-semester-assignments",
        params={"faculty_id": faculty.id, "semester": 1},
    )
    assert filtered.status_code == 200
    assert filtered.json()["total"] == 1
    assert filtered.json()["items"][0]["department_id"] == department.id


def test_class_course_assignment_crud(client, db_session):
    faculty, department, class_batch, course = _seed_faculty_graph(db_session)

    api, current = client
    current["roles"] = ["ACADEMIA"]

    response = api.post(
        "/academic-structure/class-course-assignments",
        json={
            "class_id": class_batch.id,
            "course_id": course.id,
            "faculty_id": faculty.id,
            "department_id": department.id,
        },
    )

    assert response.status_code == 200
    assignment_id = response.json()["id"]

    fetched = api.get(f"/academic-structure/class-course-assignments/{assignment_id}")
    assert fetched.status_code == 200
    assert fetched.json()["class_id"] == class_batch.id

    deleted = api.delete(f"/academic-structure/class-course-assignments/{assignment_id}")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True

    missing = api.get(f"/academic-structure/class-course-assignments/{assignment_id}")
    assert missing.status_code == 404
