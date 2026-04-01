from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base, Faculty
from app.db.role_scoped import get_role_scoped_db


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


def _user(*, roles: list[str], faculty_id: int | None = None):
    return SimpleNamespace(
        roles=[SimpleNamespace(name=role) for role in roles],
        faculty_id=faculty_id,
    )


def test_academia_uses_central_db_even_when_tenant_routing_enabled(monkeypatch):
    central_sessionmaker = _build_sessionmaker()
    tenant_sessionmaker = _build_sessionmaker()

    monkeypatch.setattr(
        "app.db.role_scoped.settings",
        SimpleNamespace(tenant_db_runtime_routing_enabled=True),
    )
    monkeypatch.setattr("app.db.role_scoped.SessionLocal", central_sessionmaker)
    monkeypatch.setattr("app.db.role_scoped.get_tenant_sessionmaker", lambda _: tenant_sessionmaker)

    generator = get_role_scoped_db(current_user=_user(roles=["ACADEMIA"]))
    db = next(generator)

    try:
        assert db.get_bind() is central_sessionmaker.kw["bind"]
    finally:
        generator.close()


def test_faculty_admin_uses_tenant_db_when_tenant_routing_enabled(monkeypatch):
    central_sessionmaker = _build_sessionmaker()
    tenant_sessionmaker = _build_sessionmaker()

    central_db = central_sessionmaker()
    central_db.add(
        Faculty(
            id=1,
            name="Faculty of Engineering",
            code="ENG",
            tenant_db_name="tenant_eng",
            tenant_db_provisioned_at=datetime.now(timezone.utc),
        )
    )
    central_db.commit()
    central_db.close()

    monkeypatch.setattr(
        "app.db.role_scoped.settings",
        SimpleNamespace(tenant_db_runtime_routing_enabled=True),
    )
    monkeypatch.setattr("app.db.role_scoped.SessionLocal", central_sessionmaker)
    monkeypatch.setattr("app.db.role_scoped.get_tenant_sessionmaker", lambda _: tenant_sessionmaker)

    generator = get_role_scoped_db(current_user=_user(roles=["FACULTY_ADMIN"], faculty_id=1))
    db = next(generator)

    try:
        assert db.get_bind() is tenant_sessionmaker.kw["bind"]
    finally:
        generator.close()


def test_faculty_admin_does_not_fall_back_to_central_when_tenant_missing(monkeypatch):
    central_sessionmaker = _build_sessionmaker()

    central_db = central_sessionmaker()
    central_db.add(
        Faculty(
            id=1,
            name="Faculty of Engineering",
            code="ENG",
            tenant_db_name=None,
            tenant_db_provisioned_at=None,
        )
    )
    central_db.commit()
    central_db.close()

    monkeypatch.setattr(
        "app.db.role_scoped.settings",
        SimpleNamespace(tenant_db_runtime_routing_enabled=True),
    )
    monkeypatch.setattr("app.db.role_scoped.SessionLocal", central_sessionmaker)

    with pytest.raises(Exception) as exc_info:
        generator = get_role_scoped_db(current_user=_user(roles=["FACULTY_ADMIN"], faculty_id=1))
        next(generator)

    assert getattr(exc_info.value, "status_code", None) == 503
    assert "tenant database is not configured" in str(getattr(exc_info.value, "detail", "")).lower()


def test_faculty_admin_does_not_fall_back_to_central_when_tenant_unprovisioned(monkeypatch):
    central_sessionmaker = _build_sessionmaker()

    central_db = central_sessionmaker()
    central_db.add(
        Faculty(
            id=1,
            name="Faculty of Engineering",
            code="ENG",
            tenant_db_name="tenant_eng",
            tenant_db_provisioned_at=None,
        )
    )
    central_db.commit()
    central_db.close()

    monkeypatch.setattr(
        "app.db.role_scoped.settings",
        SimpleNamespace(tenant_db_runtime_routing_enabled=True),
    )
    monkeypatch.setattr("app.db.role_scoped.SessionLocal", central_sessionmaker)

    with pytest.raises(Exception) as exc_info:
        generator = get_role_scoped_db(current_user=_user(roles=["FACULTY_ADMIN"], faculty_id=1))
        next(generator)

    assert getattr(exc_info.value, "status_code", None) == 503
    assert "tenant database is not provisioned" in str(getattr(exc_info.value, "detail", "")).lower()