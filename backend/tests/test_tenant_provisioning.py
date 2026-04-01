from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base, Faculty
from app.db.provision_tenants import provision_all_faculty_tenants
from app.services.tenant_provisioning import build_tenant_db_name


def test_build_tenant_db_name_normalizes_code():
    assert build_tenant_db_name(" F-ENG  ") == "tenant_f_eng"


def test_provision_all_faculty_tenants_sets_tenant_name_when_missing(monkeypatch):
    engine = create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    db = TestingSession()
    db.add(Faculty(name="Faculty of Engineering", code="ENG"))
    db.commit()
    db.close()

    # Force the command to use this in-memory DB session.
    monkeypatch.setattr("app.db.provision_tenants.SessionLocal", TestingSession)

    summary = provision_all_faculty_tenants()
    assert summary["processed"] == 1
    assert summary["skipped"] == 1

    db = TestingSession()
    faculty = db.query(Faculty).filter(Faculty.code == "ENG").first()
    assert faculty is not None
    assert faculty.tenant_db_name == "tenant_eng"
    assert faculty.tenant_db_provisioned_at is None
    db.close()
