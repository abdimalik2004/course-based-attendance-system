from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_BACKEND_DIR = _PROJECT_ROOT / "backend"

load_dotenv(_BACKEND_DIR / ".env", override=False)
_env_profile = os.getenv("APP_ENV", "development").strip().lower()
load_dotenv(_BACKEND_DIR / f".env.{_env_profile}", override=True)


def _env_list(name: str, default: str) -> tuple[str, ...]:
    value = os.getenv(name, default)
    return tuple(item.strip() for item in value.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Course Attendance Backend")
    app_env: str = os.getenv("APP_ENV", "development")
    app_timezone: str = os.getenv("APP_TIMEZONE", "Africa/Mogadishu")
    secret_key: str = os.getenv("SECRET_KEY", "change-me-in-production")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    access_token_expire_minutes: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "120"))
    refresh_token_expire_minutes: int = int(os.getenv("REFRESH_TOKEN_EXPIRE_MINUTES", "10080"))

    db_type: str = os.getenv("DB_TYPE", "sqlite")
    mysql_host: str = os.getenv("MYSQL_HOST", "localhost")
    mysql_port: int = int(os.getenv("MYSQL_PORT", "3306"))
    mysql_user: str = os.getenv("MYSQL_USER", "root")
    mysql_password: str = os.getenv("MYSQL_PASSWORD", "")
    mysql_db: str = os.getenv("MYSQL_DB", "attendance")
    mysql_admin_db: str = os.getenv("MYSQL_ADMIN_DB", "mysql")
    sqlite_path: str = os.getenv("SQLITE_PATH", "backend/database/attendance.db")

    tenant_db_prefix: str = os.getenv("TENANT_DB_PREFIX", "tenant_")
    tenant_db_charset: str = os.getenv("TENANT_DB_CHARSET", "utf8mb4")
    tenant_db_collation: str = os.getenv("TENANT_DB_COLLATION", "utf8mb4_unicode_ci")
    tenant_db_auto_provision_enabled: bool = os.getenv("TENANT_DB_AUTO_PROVISION_ENABLED", "true").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    tenant_db_runtime_routing_enabled: bool = os.getenv("TENANT_DB_RUNTIME_ROUTING_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    tenant_db_scheduler_enabled: bool = os.getenv("TENANT_DB_SCHEDULER_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    tenant_db_allow_legacy_operational_sync: bool = os.getenv(
        "TENANT_DB_ALLOW_LEGACY_OPERATIONAL_SYNC",
        "false",
    ).strip().lower() in {
        "1",
        "true",
        "yes",
    }

    scheduler_poll_seconds: int = int(os.getenv("SCHEDULER_POLL_SECONDS", "60"))
    scheduler_tenant_failure_threshold: int = int(os.getenv("SCHEDULER_TENANT_FAILURE_THRESHOLD", "3"))
    scheduler_tenant_stale_seconds: int = int(os.getenv("SCHEDULER_TENANT_STALE_SECONDS", "180"))
    default_grace_period_minutes: int = int(os.getenv("DEFAULT_GRACE_PERIOD_MINUTES", "10"))

    face_confidence_threshold: float = float(os.getenv("FACE_CONFIDENCE_THRESHOLD", "0.60"))
    face_timeout_seconds: float = float(os.getenv("FACE_TIMEOUT_SECONDS", "2.0"))
    cors_allow_origins: tuple[str, ...] = _env_list("CORS_ALLOW_ORIGINS", "http://localhost:3000")
    cors_allow_methods: tuple[str, ...] = _env_list("CORS_ALLOW_METHODS", "*")
    cors_allow_headers: tuple[str, ...] = _env_list("CORS_ALLOW_HEADERS", "*")
    cors_allow_credentials: bool = os.getenv("CORS_ALLOW_CREDENTIALS", "true").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
    auth_rate_limit_requests: int = int(os.getenv("AUTH_RATE_LIMIT_REQUESTS", "10"))
    auth_rate_limit_window_seconds: int = int(os.getenv("AUTH_RATE_LIMIT_WINDOW_SECONDS", "60"))
    frame_rate_limit_requests: int = int(os.getenv("FRAME_RATE_LIMIT_REQUESTS", "180"))
    frame_rate_limit_window_seconds: int = int(os.getenv("FRAME_RATE_LIMIT_WINDOW_SECONDS", "60"))

    @property
    def database_url(self) -> str:
        if self.db_type.lower() == "mysql":
            return (
                f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}@"
                f"{self.mysql_host}:{self.mysql_port}/{self.mysql_db}"
            )
        sqlite_file = _PROJECT_ROOT / self.sqlite_path
        sqlite_file.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{sqlite_file.as_posix()}"


settings = Settings()
