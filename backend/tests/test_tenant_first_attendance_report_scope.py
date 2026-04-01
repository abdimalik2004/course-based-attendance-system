from __future__ import annotations

from datetime import date, datetime, time, timedelta
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.security import get_current_user
from app.db.faculty_scope import FacultyScopeContext, get_optional_faculty_scope_context
from app.db.models import (
    AttendanceSession,
    Base,
    ClassBatch,
    Course,
    CourseSchedule,
    Department,
    Faculty,
    SessionStatus,
)
from app.db.role_scoped import get_role_scoped_db
from app.routers import attendance, reports, sessions


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
def tenant_scope_app(monkeypatch):
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

    now = datetime.now()
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
    tenant_db.flush()
    tenant_db.add_all(
        [
            CourseSchedule(course_id=1, weekday=2, start_time=time(8, 0), end_time=time(10, 0), grace_period_minutes=10),
            CourseSchedule(course_id=2, weekday=2, start_time=time(8, 0), end_time=time(10, 0), grace_period_minutes=10),
        ]
    )
    tenant_db.flush()
    tenant_db.add_all(
        [
            AttendanceSession(
                course_id=1,
                schedule_id=1,
                session_date=date.today(),
                start_time=now - timedelta(minutes=5),
                end_time=now + timedelta(minutes=30),
                status=SessionStatus.ACTIVE,
            ),
            AttendanceSession(
                course_id=2,
                schedule_id=2,
                session_date=date.today(),
                start_time=now - timedelta(minutes=5),
                end_time=now + timedelta(minutes=30),
                status=SessionStatus.ACTIVE,
            ),
        ]
    )
    tenant_db.commit()
    tenant_db.close()

    monkeypatch.setattr("app.db.faculty_scope.SessionLocal", central_sessionmaker)
    monkeypatch.setattr("app.routers.attendance.attendance_service.process_frame", lambda **kwargs: {"ok": True})

    app = FastAPI()
    app.include_router(attendance.router)
    app.include_router(reports.router)
    app.include_router(sessions.router)

    def _override_tenant_db():
        db = tenant_sessionmaker()
        try:
            yield db
        finally:
            db.close()

    def _override_current_user():
        return SimpleNamespace(
            id=101,
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

    return TestClient(app)


def test_sessions_list_filters_to_faculty_scope(tenant_scope_app):
    response = tenant_scope_app.get("/sessions")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["course_id"] == 1


def test_sessions_active_rejects_cross_faculty_course_filter(tenant_scope_app):
    response = tenant_scope_app.get("/sessions/active", params={"course_id": 2})
    assert response.status_code == 403
    assert "another faculty" in response.json()["detail"]


def test_reports_reject_cross_faculty_course_access(tenant_scope_app):
    response = tenant_scope_app.get("/reports/course/2")
    assert response.status_code == 403
    assert "another faculty" in response.json()["detail"]


def test_attendance_frame_rejects_cross_faculty_session_access(tenant_scope_app):
    response = tenant_scope_app.post(
        "/attendance/frame",
        json={"session_id": 2, "image": "ZmFrZQ=="},
    )
    assert response.status_code == 403
    assert "another faculty" in response.json()["detail"]


def test_attendance_frame_allows_same_faculty_session_access(tenant_scope_app):
    response = tenant_scope_app.post(
        "/attendance/frame",
        json={"session_id": 1, "image": "ZmFrZQ=="},
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True