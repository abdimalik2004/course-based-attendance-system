from __future__ import annotations

import json
import logging
import logging.config
from datetime import datetime, timezone

from app.core.config import settings


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


def configure_logging() -> None:
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "json": {
                    "()": "app.core.logging_config.JsonFormatter",
                }
            },
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "formatter": "json",
                    "level": settings.log_level,
                }
            },
            "root": {
                "handlers": ["default"],
                "level": settings.log_level,
            },
            "loggers": {
                "uvicorn": {"handlers": ["default"], "level": settings.log_level, "propagate": False},
                "uvicorn.error": {"handlers": ["default"], "level": settings.log_level, "propagate": False},
                "uvicorn.access": {"handlers": ["default"], "level": settings.log_level, "propagate": False},
            },
        }
    )
