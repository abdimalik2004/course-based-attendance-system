from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.security import get_current_user
from app.db.faculty_scope import FacultyScopeContext, get_optional_faculty_scope_context
from app.db.models import Base, ClassBatch, Department, Faculty, Student
from app.db.role_scoped import get_role_scoped_db
from app.routers import students


def _build_sessionmaker():
    engine = create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    testing_session = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)
    return testing_session


def _role(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name)


@pytest.fixture()
def tenant_student_app(monkeypatch):
    central_sessionmaker = _build_sessionmaker()
    tenant_sessionmaker = _build_sessionmaker()

    central_db = central_sessionmaker()
    central_db.add_all(
        [
            Faculty(name="Faculty of Engineering", code="ENG", tenant_db_name="tenant_eng"),
            Faculty(name="Faculty of Medicine", code="MED", tenant_db_name="tenant_med"),
        ]
    )
    central_db.commit()
    central_db.close()

    tenant_db = tenant_sessionmaker()
    tenant_db.add(Department(faculty_id=1, name="Architecture", code="ARCH"))
    tenant_db.flush()
    tenant_db.add(ClassBatch(faculty_id=1, department_id=1, name="ENG2201", year=2026))
    tenant_db.commit()
    tenant_db.close()

    monkeypatch.setattr("app.db.faculty_scope.SessionLocal", central_sessionmaker)
    monkeypatch.setattr("app.utils.student_numbering._candidate_numbers_from_dataset", lambda prefix: set())

    app = FastAPI()
    app.include_router(students.router)

    def _override_tenant_db():
        db = tenant_sessionmaker()
        try:
            yield db
        finally:
            db.close()

    def _override_current_user():
        return SimpleNamespace(
            roles=[_role("FACULTY_ADMIN")],
            is_active=True,
            faculty_id=1,
        )

    def _override_faculty_scope():
        return FacultyScopeContext(
            faculty_id=1,
            faculty_code="ENG",
            tenant_db_name="tenant_eng",
            tenant_db_provisioned=True,
        )

    app.dependency_overrides[get_role_scoped_db] = _override_tenant_db
    app.dependency_overrides[get_current_user] = _override_current_user
    app.dependency_overrides[get_optional_faculty_scope_context] = _override_faculty_scope

    return TestClient(app), central_sessionmaker, tenant_sessionmaker


def test_student_create_writes_only_to_tenant_db(tenant_student_app):
    client, central_sessionmaker, tenant_sessionmaker = tenant_student_app

    response = client.post(
        "/students",
        json={
            "full_name": "Tenant Student",
            "faculty_id": 1,
            "department_id": 1,
            "class_batch_id": 1,
        },
    )

    assert response.status_code == 200
    assert response.json()["student_number"] == "26ENG001"

    central_db = central_sessionmaker()
    tenant_db = tenant_sessionmaker()
    try:
        assert central_db.query(Student).count() == 0
        student = tenant_db.query(Student).filter(Student.student_number == "26ENG001").first()
        assert student is not None
        assert student.faculty_id == 1
        assert student.department_id == 1
        assert student.class_batch_id == 1
        assert student.embedding_ref == "26ENG001"
    finally:
        central_db.close()
        tenant_db.close()


def test_student_create_rejects_cross_faculty_payload(tenant_student_app):
    client, _, tenant_sessionmaker = tenant_student_app

    response = client.post(
        "/students",
        json={
            "full_name": "Cross Faculty Student",
            "faculty_id": 2,
            "department_id": 1,
            "class_batch_id": 1,
        },
    )

    assert response.status_code == 403
    assert "another faculty" in response.json()["detail"]

    tenant_db = tenant_sessionmaker()
    try:
        assert tenant_db.query(Student).count() == 0
    finally:
        tenant_db.close()


def test_student_update_rejects_cross_faculty_payload(tenant_student_app):
    client, _, tenant_sessionmaker = tenant_student_app

    tenant_db = tenant_sessionmaker()
    tenant_db.add(
        Student(
            student_number="26ENG001",
            full_name="Existing Student",
            faculty_id=1,
            department_id=1,
            class_batch_id=1,
            embedding_ref="26ENG001",
        )
    )
    tenant_db.commit()
    tenant_db.close()

    response = client.put(
        "/students/1",
        json={"faculty_id": 2},
    )

    assert response.status_code == 403
    assert "another faculty" in response.json()["detail"]