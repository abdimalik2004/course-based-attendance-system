from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import re

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from app.core.config import settings
from app.db.models import Base


@dataclass(frozen=True)
class TenantProvisionResult:
    tenant_db_name: str
    provisioned: bool
    skipped: bool = False
    reason: str | None = None


def build_tenant_db_name(faculty_code: str) -> str:
    sanitized = re.sub(r"[^a-z0-9_]", "_", (faculty_code or "").strip().lower())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    if not sanitized:
        raise ValueError("Faculty code cannot be empty when deriving tenant database name")
    return f"{settings.tenant_db_prefix}{sanitized}"


def _mysql_server_url() -> str:
    return (
        f"mysql+pymysql://{settings.mysql_user}:{settings.mysql_password}@"
        f"{settings.mysql_host}:{settings.mysql_port}/{settings.mysql_admin_db}"
    )


def _mysql_tenant_url(tenant_db_name: str) -> str:
    return (
        f"mysql+pymysql://{settings.mysql_user}:{settings.mysql_password}@"
        f"{settings.mysql_host}:{settings.mysql_port}/{tenant_db_name}"
    )


def provision_faculty_tenant_database(tenant_db_name: str) -> TenantProvisionResult:
    if not settings.tenant_db_auto_provision_enabled:
        return TenantProvisionResult(
            tenant_db_name=tenant_db_name,
            provisioned=False,
            skipped=True,
            reason="Tenant auto provisioning is disabled by configuration",
        )

    if settings.db_type.lower() != "mysql":
        return TenantProvisionResult(
            tenant_db_name=tenant_db_name,
            provisioned=False,
            skipped=True,
            reason="Current DB_TYPE is not mysql; tenant database provisioning skipped",
        )

    try:
        server_engine = create_engine(_mysql_server_url(), future=True, pool_pre_ping=True)
        with server_engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE DATABASE IF NOT EXISTS "
                    f"`{tenant_db_name}` "
                    f"CHARACTER SET {settings.tenant_db_charset} "
                    f"COLLATE {settings.tenant_db_collation}"
                )
            )

        # Phase 1: create schema in the tenant DB using current ORM metadata.
        tenant_engine = create_engine(_mysql_tenant_url(tenant_db_name), future=True, pool_pre_ping=True)
        Base.metadata.create_all(bind=tenant_engine)

        return TenantProvisionResult(
            tenant_db_name=tenant_db_name,
            provisioned=True,
            skipped=False,
            reason=f"Tenant database provisioned at {datetime.now(timezone.utc).isoformat()}",
        )
    except SQLAlchemyError as exc:
        return TenantProvisionResult(
            tenant_db_name=tenant_db_name,
            provisioned=False,
            skipped=False,
            reason=str(exc),
        )
