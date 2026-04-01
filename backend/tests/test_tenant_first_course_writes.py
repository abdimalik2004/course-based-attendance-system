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
from app.db.models import Base, ClassBatch, Course, Department, Faculty
from app.db.role_scoped import get_role_scoped_db
from app.routers import courses


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
def tenant_course_app(monkeypatch):
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
    tenant_db.add_all(
        [
            ClassBatch(faculty_id=1, department_id=1, name="ENG2201", year=2026),
            ClassBatch(faculty_id=2, department_id=1, name="MED2201", year=2026),
        ]
    )
    tenant_db.commit()
    tenant_db.close()

    monkeypatch.setattr("app.db.faculty_scope.SessionLocal", central_sessionmaker)

    app = FastAPI()
    app.include_router(courses.router)

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


def test_course_create_writes_only_to_tenant_db(tenant_course_app):
    client, central_sessionmaker, tenant_sessionmaker = tenant_course_app

    response = client.post(
        "/courses",
        json={"class_batch_id": 1, "title": "Tenant First Course"},
    )

    assert response.status_code == 200
    assert response.json()["code"] == "ENG001"

    central_db = central_sessionmaker()
    tenant_db = tenant_sessionmaker()
    try:
        assert central_db.query(Course).count() == 0
        assert tenant_db.query(Course).filter(Course.code == "ENG001").count() == 1
    finally:
        central_db.close()
        tenant_db.close()


def test_course_create_rejects_cross_faculty_class_batch(tenant_course_app):
    client, _, tenant_sessionmaker = tenant_course_app

    response = client.post(
        "/courses",
        json={"class_batch_id": 2, "title": "Cross Faculty Course"},
    )

    assert response.status_code == 403
    assert "another faculty" in response.json()["detail"]

    tenant_db = tenant_sessionmaker()
    try:
        assert tenant_db.query(Course).count() == 0
    finally:
        tenant_db.close()


def test_course_update_rejects_cross_faculty_target_batch(tenant_course_app):
    client, _, tenant_sessionmaker = tenant_course_app

    tenant_db = tenant_sessionmaker()
    tenant_db.add(Course(class_batch_id=1, code="ENG001", title="Existing"))
    tenant_db.commit()
    tenant_db.close()

    response = client.put(
        "/courses/1",
        json={"class_batch_id": 2},
    )

    assert response.status_code == 403
    assert "another faculty" in response.json()["detail"]