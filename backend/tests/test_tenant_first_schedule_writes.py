from __future__ import annotations

from datetime import time
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.security import get_current_user
from app.db.faculty_scope import FacultyScopeContext, get_optional_faculty_scope_context
from app.db.models import Base, ClassBatch, Course, CourseSchedule, Department, Faculty
from app.db.role_scoped import get_role_scoped_db
from app.routers import schedules


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
def tenant_schedule_app(monkeypatch):
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
    tenant_db.flush()
    tenant_db.add_all(
        [
            Course(class_batch_id=1, code="ENG001", title="Engineering Course"),
            Course(class_batch_id=2, code="MED001", title="Medical Course"),
        ]
    )
    tenant_db.commit()
    tenant_db.close()

    monkeypatch.setattr("app.db.faculty_scope.SessionLocal", central_sessionmaker)

    app = FastAPI()
    app.include_router(schedules.router)

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

    return TestClient(app), tenant_sessionmaker


def test_schedule_create_for_same_faculty_succeeds(tenant_schedule_app):
    client, tenant_sessionmaker = tenant_schedule_app

    response = client.post(
        "/schedules",
        json={
            "course_id": 1,
            "weekday": ["mon"],
            "start_time": "08:00:00",
            "end_time": "10:00:00",
            "grace_period_minutes": 10,
        },
    )

    assert response.status_code == 200

    tenant_db = tenant_sessionmaker()
    try:
        assert tenant_db.query(CourseSchedule).count() == 1
    finally:
        tenant_db.close()


def test_schedule_create_cross_faculty_course_is_rejected(tenant_schedule_app):
    client, tenant_sessionmaker = tenant_schedule_app

    response = client.post(
        "/schedules",
        json={
            "course_id": 2,
            "weekday": ["mon"],
            "start_time": "08:00:00",
            "end_time": "10:00:00",
            "grace_period_minutes": 10,
        },
    )

    assert response.status_code == 403
    assert "another faculty" in response.json()["detail"]

    tenant_db = tenant_sessionmaker()
    try:
        assert tenant_db.query(CourseSchedule).count() == 0
    finally:
        tenant_db.close()


def test_schedule_update_cross_faculty_target_course_is_rejected(tenant_schedule_app):
    client, tenant_sessionmaker = tenant_schedule_app

    tenant_db = tenant_sessionmaker()
    tenant_db.add(
        CourseSchedule(
            course_id=1,
            weekday=2,
            start_time=time(8, 0),
            end_time=time(10, 0),
            grace_period_minutes=10,
        )
    )
    tenant_db.commit()
    tenant_db.close()

    response = client.put(
        "/schedules/1",
        json={"course_id": 2},
    )

    assert response.status_code == 403
    assert "another faculty" in response.json()["detail"]


def test_schedule_delete_cross_faculty_is_rejected(tenant_schedule_app):
    client, tenant_sessionmaker = tenant_schedule_app

    tenant_db = tenant_sessionmaker()
    tenant_db.add(
        CourseSchedule(
            course_id=2,
            weekday=2,
            start_time=time(8, 0),
            end_time=time(10, 0),
            grace_period_minutes=10,
        )
    )
    tenant_db.commit()
    tenant_db.close()

    response = client.delete("/schedules/1")

    assert response.status_code == 403
    assert "another faculty" in response.json()["detail"]