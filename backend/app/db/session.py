from __future__ import annotations

from collections.abc import Generator
from threading import Lock

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings


engine_kwargs = {}
if settings.database_url.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(settings.database_url, future=True, pool_pre_ping=True, **engine_kwargs)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, class_=Session)
_tenant_sessionmakers: dict[str, sessionmaker] = {}
_tenant_lock = Lock()


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _tenant_db_url(tenant_db_name: str) -> str:
    return (
        f"mysql+pymysql://{settings.mysql_user}:{settings.mysql_password}@"
        f"{settings.mysql_host}:{settings.mysql_port}/{tenant_db_name}"
    )


def _get_tenant_sessionmaker(tenant_db_name: str) -> sessionmaker:
    with _tenant_lock:
        cached = _tenant_sessionmakers.get(tenant_db_name)
        if cached is not None:
            return cached

        tenant_engine = create_engine(_tenant_db_url(tenant_db_name), future=True, pool_pre_ping=True)
        maker = sessionmaker(bind=tenant_engine, autoflush=False, autocommit=False, class_=Session)
        _tenant_sessionmakers[tenant_db_name] = maker
        return maker


def get_tenant_sessionmaker(tenant_db_name: str) -> sessionmaker:
    return _get_tenant_sessionmaker(tenant_db_name)
