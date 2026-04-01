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
from app.db.models import Base, ClassBatch, Department, Faculty, Role
from app.db.role_scoped import get_role_scoped_db
from app.routers import classes, departments


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
def tenant_first_app(monkeypatch):
    central_sessionmaker = _build_sessionmaker()
    tenant_sessionmaker = _build_sessionmaker()

    central_db = central_sessionmaker()
    central_db.add(Role(name="FACULTY_ADMIN"))
    central_db.add_all(
        [
            Faculty(name="Faculty of Engineering", code="ENG", tenant_db_name="tenant_eng"),
            Faculty(name="Faculty of Medicine", code="MED", tenant_db_name="tenant_med"),
        ]
    )
    central_db.commit()
    central_db.close()

    tenant_db = tenant_sessionmaker()
    tenant_db.add(Faculty(id=1, name="Faculty of Engineering", code="ENG"))
    tenant_db.flush()
    tenant_db.add(Department(faculty_id=1, name="Architecture", code="ARCH"))
    tenant_db.commit()
    tenant_db.close()

    monkeypatch.setattr("app.db.faculty_scope.SessionLocal", central_sessionmaker)
    monkeypatch.setattr("app.utils.organization.SessionLocal", central_sessionmaker)

    app = FastAPI()
    app.include_router(departments.router)
    app.include_router(classes.router)

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


def test_faculty_admin_department_create_writes_only_to_tenant_db(tenant_first_app):
    client, central_sessionmaker, tenant_sessionmaker = tenant_first_app

    response = client.post(
        "/departments",
        json={"faculty_id": 1, "name": "Computer Science", "code": "CIS"},
    )

    assert response.status_code == 200
    assert response.json()["code"] == "CIS"

    central_db = central_sessionmaker()
    tenant_db = tenant_sessionmaker()
    try:
        assert central_db.query(Department).count() == 0
        assert tenant_db.query(Department).filter(Department.code == "CIS").count() == 1
    finally:
        central_db.close()
        tenant_db.close()


def test_faculty_admin_department_create_rejects_other_faculty(tenant_first_app):
    client, _, tenant_sessionmaker = tenant_first_app

    response = client.post(
        "/departments",
        json={"faculty_id": 2, "name": "Computer Science", "code": "CIS"},
    )

    assert response.status_code == 403
    assert "another faculty" in response.json()["detail"]

    tenant_db = tenant_sessionmaker()
    try:
        assert tenant_db.query(Department).filter(Department.code == "CIS").count() == 0
    finally:
        tenant_db.close()


def test_faculty_admin_department_create_uses_scope_faculty_when_payload_omits_faculty_id(tenant_first_app):
    client, _, tenant_sessionmaker = tenant_first_app

    response = client.post(
        "/departments",
        json={"name": "Computer Science", "code": "CIS"},
    )

    assert response.status_code == 200
    assert response.json()["faculty_id"] == 1

    tenant_db = tenant_sessionmaker()
    try:
        department = tenant_db.query(Department).filter(Department.code == "CIS").first()
        assert department is not None
        assert department.faculty_id == 1
    finally:
        tenant_db.close()


def test_faculty_admin_department_create_materializes_missing_tenant_faculty_row(tenant_first_app):
    client, _, tenant_sessionmaker = tenant_first_app

    tenant_db = tenant_sessionmaker()
    try:
        tenant_db.query(Department).delete()
        tenant_db.query(Faculty).delete()
        tenant_db.commit()
    finally:
        tenant_db.close()

    response = client.post(
        "/departments",
        json={"name": "Computer Science", "code": "CIS"},
    )

    assert response.status_code == 200
    assert response.json()["faculty_id"] == 1

    tenant_db = tenant_sessionmaker()
    try:
        tenant_faculty = tenant_db.query(Faculty).filter(Faculty.id == 1).first()
        assert tenant_faculty is not None
        assert tenant_faculty.code == "ENG"
    finally:
        tenant_db.close()


def test_faculty_admin_class_create_uses_tenant_department_with_tenant_faculty_row(tenant_first_app):
    client, central_sessionmaker, tenant_sessionmaker = tenant_first_app

    response = client.post(
        "/classes",
        json={"faculty_id": 1, "department_id": 1, "name": "ENG2201", "year": 2026},
    )

    assert response.status_code == 200
    assert response.json()["name"] == "ENG2201"

    central_db = central_sessionmaker()
    tenant_db = tenant_sessionmaker()
    try:
        assert central_db.query(ClassBatch).count() == 0
        class_batch = tenant_db.query(ClassBatch).filter(ClassBatch.name == "ENG2201").first()
        assert class_batch is not None
        assert class_batch.department_id == 1
        assert class_batch.faculty_id == 1
    finally:
        central_db.close()
        tenant_db.close()


def test_faculty_admin_class_create_uses_scope_faculty_when_payload_omits_faculty_id(tenant_first_app):
    client, _, tenant_sessionmaker = tenant_first_app

    response = client.post(
        "/classes",
        json={"department_id": 1, "name": "ENG2201", "year": 2026},
    )

    assert response.status_code == 200
    assert response.json()["faculty_id"] == 1

    tenant_db = tenant_sessionmaker()
    try:
        class_batch = tenant_db.query(ClassBatch).filter(ClassBatch.name == "ENG2201").first()
        assert class_batch is not None
        assert class_batch.faculty_id == 1
    finally:
        tenant_db.close()


def test_faculty_admin_class_create_auto_generates_name_from_existing_sequence(tenant_first_app):
    client, _, tenant_sessionmaker = tenant_first_app

    tenant_db = tenant_sessionmaker()
    try:
        tenant_db.add(ClassBatch(faculty_id=1, department_id=1, name="ENG001", year=2026))
        tenant_db.commit()
    finally:
        tenant_db.close()

    response = client.post(
        "/classes",
        json={"department_id": 1, "year": 2027},
    )

    assert response.status_code == 200
    assert response.json()["name"] == "ENG002"