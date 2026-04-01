from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base, Faculty
from app.services.schedule_service import ScheduleService, TenantTickStatus


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


def test_run_once_uses_tenant_mode_when_enabled(monkeypatch):
    service = ScheduleService()
    called = {"tenant": 0, "central": 0}

    monkeypatch.setattr(
        "app.services.schedule_service.settings",
        SimpleNamespace(
            tenant_db_runtime_routing_enabled=True,
            tenant_db_scheduler_enabled=True,
            scheduler_poll_seconds=60,
            scheduler_tenant_failure_threshold=3,
            scheduler_tenant_stale_seconds=180,
        ),
    )
    monkeypatch.setattr(service, "_tick_all_tenants", lambda: called.__setitem__("tenant", called["tenant"] + 1))
    monkeypatch.setattr("app.services.schedule_service.SessionLocal", lambda: (_ for _ in ()).throw(RuntimeError("should not use central")))

    service._run_once()

    assert called["tenant"] == 1
    assert called["central"] == 0


def test_tick_all_tenants_processes_only_provisioned_tenant_dbs(monkeypatch):
    central_sessionmaker = _build_sessionmaker()
    tenant_eng_sessionmaker = _build_sessionmaker()
    tenant_med_sessionmaker = _build_sessionmaker()

    central_db = central_sessionmaker()
    central_db.add_all(
        [
            Faculty(
                name="Faculty of Engineering",
                code="ENG",
                tenant_db_name="tenant_eng",
                tenant_db_provisioned_at=datetime.now(timezone.utc),
            ),
            Faculty(
                name="Faculty of Medicine",
                code="MED",
                tenant_db_name="tenant_med",
                tenant_db_provisioned_at=datetime.now(timezone.utc),
            ),
            Faculty(
                name="Faculty of Science",
                code="SCI",
                tenant_db_name="tenant_sci",
                tenant_db_provisioned_at=None,
            ),
            Faculty(
                name="Faculty of Law",
                code="LAW",
                tenant_db_name=None,
                tenant_db_provisioned_at=None,
            ),
        ]
    )
    central_db.commit()
    central_db.close()

    monkeypatch.setattr("app.services.schedule_service.SessionLocal", central_sessionmaker)

    def _tenant_maker(name: str):
        if name == "tenant_eng":
            return tenant_eng_sessionmaker
        if name == "tenant_med":
            return tenant_med_sessionmaker
        raise AssertionError(f"unexpected tenant DB requested: {name}")

    monkeypatch.setattr("app.services.schedule_service.get_tenant_sessionmaker", _tenant_maker)

    seen_engines = []
    service = ScheduleService()

    def _capture_tick(db):
        seen_engines.append(db.get_bind())

    monkeypatch.setattr(service, "_tick", _capture_tick)

    service._tick_all_tenants()

    assert len(seen_engines) == 2
    assert tenant_eng_sessionmaker.kw["bind"] in seen_engines
    assert tenant_med_sessionmaker.kw["bind"] in seen_engines


def test_tenant_tick_report_includes_recent_failure(monkeypatch):
    central_sessionmaker = _build_sessionmaker()

    central_db = central_sessionmaker()
    central_db.add(
        Faculty(
            name="Faculty of Engineering",
            code="ENG",
            tenant_db_name="tenant_eng",
            tenant_db_provisioned_at=datetime.now(timezone.utc),
        )
    )
    central_db.commit()
    central_db.close()

    monkeypatch.setattr("app.services.schedule_service.SessionLocal", central_sessionmaker)
    monkeypatch.setattr(
        "app.services.schedule_service.settings",
        SimpleNamespace(
            tenant_db_runtime_routing_enabled=True,
            tenant_db_scheduler_enabled=True,
            scheduler_poll_seconds=60,
            scheduler_tenant_failure_threshold=3,
            scheduler_tenant_stale_seconds=180,
        ),
    )

    tenant_sessionmaker = _build_sessionmaker()
    monkeypatch.setattr("app.services.schedule_service.get_tenant_sessionmaker", lambda _: tenant_sessionmaker)

    service = ScheduleService()

    def _fail_tick(_db):
        raise RuntimeError("tick failed")

    monkeypatch.setattr(service, "_tick", _fail_tick)

    service._run_once()

    report = service.tenant_tick_report()
    assert report["mode"] == "tenant"
    assert report["tenant_mode_enabled"] is True
    assert report["unhealthy_tenant_count"] == 0
    assert len(report["tenants"]) == 1
    tenant = report["tenants"][0]
    assert tenant["faculty_code"] == "ENG"
    assert tenant["total_failures"] == 1
    assert tenant["consecutive_failures"] == 1
    assert tenant["alert_reasons"] == []
    assert tenant["is_healthy"] is True
    assert "tick failed" in tenant["last_error"]


def test_tenant_tick_report_flags_unhealthy_tenants(monkeypatch):
    monkeypatch.setattr(
        "app.services.schedule_service.settings",
        SimpleNamespace(
            tenant_db_runtime_routing_enabled=True,
            tenant_db_scheduler_enabled=True,
            scheduler_poll_seconds=60,
            scheduler_tenant_failure_threshold=2,
            scheduler_tenant_stale_seconds=180,
        ),
    )

    service = ScheduleService()
    monkeypatch.setattr(service, "is_running", lambda: True)
    now = datetime(2026, 3, 16, 11, 8, 23, tzinfo=timezone.utc)
    monkeypatch.setattr(service, "_utc_now", lambda: now)
    service._tenant_tick_status["tenant_eng"] = TenantTickStatus(
        faculty_id=1,
        faculty_code="ENG",
        tenant_db_name="tenant_eng",
        last_tick_started_at=(now - timedelta(seconds=10)).isoformat(),
        last_tick_completed_at=(now - timedelta(seconds=5)).isoformat(),
        last_success_at=(now - timedelta(seconds=120)).isoformat(),
        last_error="tick failed twice",
        total_success=5,
        total_failures=2,
        consecutive_failures=2,
    )

    report = service.tenant_tick_report()

    assert report["unhealthy_tenant_count"] == 1
    assert report["unhealthy_tenants"][0]["faculty_code"] == "ENG"
    assert report["unhealthy_tenants"][0]["alert_reasons"] == ["consecutive_failures>=2"]
    assert report["tenants"][0]["is_healthy"] is False


def test_readiness_status_reports_unhealthy_tenant_scheduler(monkeypatch):
    monkeypatch.setattr(
        "app.services.schedule_service.settings",
        SimpleNamespace(
            tenant_db_runtime_routing_enabled=True,
            tenant_db_scheduler_enabled=True,
            scheduler_poll_seconds=60,
            scheduler_tenant_failure_threshold=2,
            scheduler_tenant_stale_seconds=180,
        ),
    )

    service = ScheduleService()
    monkeypatch.setattr(service, "is_running", lambda: True)
    now = datetime(2026, 3, 16, 11, 8, 23, tzinfo=timezone.utc)
    monkeypatch.setattr(service, "_utc_now", lambda: now)
    service._tenant_tick_status["tenant_eng"] = TenantTickStatus(
        faculty_id=1,
        faculty_code="ENG",
        tenant_db_name="tenant_eng",
        last_tick_started_at=(now - timedelta(seconds=20)).isoformat(),
        last_tick_completed_at=(now - timedelta(seconds=10)).isoformat(),
        last_success_at=(now - timedelta(seconds=10)).isoformat(),
        last_error="tick failed twice",
        total_success=5,
        total_failures=2,
        consecutive_failures=2,
    )

    readiness = service.readiness_status()

    assert readiness["healthy"] is False
    assert readiness["reason"] == "Tenant scheduler unhealthy for: ENG"