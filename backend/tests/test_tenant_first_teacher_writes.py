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
from app.db.models import Base, Department, Faculty, Teacher, User
from app.db.role_scoped import get_role_scoped_db
from app.routers import teachers


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
def tenant_teacher_app(monkeypatch):
    central_sessionmaker = _build_sessionmaker()
    tenant_sessionmaker = _build_sessionmaker()

    central_db = central_sessionmaker()
    central_db.add_all(
        [
            Faculty(name="Faculty of Engineering", code="ENG", tenant_db_name="tenant_eng"),
            Faculty(name="Faculty of Medicine", code="MED", tenant_db_name="tenant_med"),
            User(username="eng_user", hashed_password="x", is_active=True, faculty_id=1),
            User(username="med_user", hashed_password="x", is_active=True, faculty_id=2),
        ]
    )
    central_db.commit()
    central_db.close()

    tenant_db = tenant_sessionmaker()
    tenant_db.add(Department(faculty_id=1, name="Architecture", code="ARCH"))
    tenant_db.commit()
    tenant_db.close()

    monkeypatch.setattr("app.db.faculty_scope.SessionLocal", central_sessionmaker)

    app = FastAPI()
    app.include_router(teachers.router)

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


def test_teacher_create_writes_only_to_tenant_db(tenant_teacher_app):
    client, central_sessionmaker, tenant_sessionmaker = tenant_teacher_app

    response = client.post(
        "/teachers",
        json={
            "full_name": "Dr. Tenant First",
            "faculty_id": 1,
            "department_id": 1,
            "user_id": 1,
        },
    )

    assert response.status_code == 200
    assert response.json()["teacher_number"] == "ENGT001"

    central_db = central_sessionmaker()
    tenant_db = tenant_sessionmaker()
    try:
        assert central_db.query(Teacher).count() == 0
        teacher = tenant_db.query(Teacher).filter(Teacher.teacher_number == "ENGT001").first()
        assert teacher is not None
        assert teacher.faculty_id == 1
        assert teacher.department_id == 1
        assert teacher.user_id == 1
    finally:
        central_db.close()
        tenant_db.close()


def test_teacher_create_rejects_cross_faculty_payload(tenant_teacher_app):
    client, _, tenant_sessionmaker = tenant_teacher_app

    response = client.post(
        "/teachers",
        json={
            "full_name": "Dr. Cross Faculty",
            "faculty_id": 2,
            "department_id": 1,
        },
    )

    assert response.status_code == 403
    assert "another faculty" in response.json()["detail"]

    tenant_db = tenant_sessionmaker()
    try:
        assert tenant_db.query(Teacher).count() == 0
    finally:
        tenant_db.close()


def test_teacher_create_rejects_user_from_other_faculty(tenant_teacher_app):
    client, _, tenant_sessionmaker = tenant_teacher_app

    response = client.post(
        "/teachers",
        json={
            "full_name": "Dr. Wrong User",
            "faculty_id": 1,
            "department_id": 1,
            "user_id": 2,
        },
    )

    assert response.status_code == 400
    assert "current faculty" in response.json()["detail"]

    tenant_db = tenant_sessionmaker()
    try:
        assert tenant_db.query(Teacher).count() == 0
    finally:
        tenant_db.close()