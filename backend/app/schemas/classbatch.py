from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, field_validator


_WS_RE = re.compile(r"\s+")


def _normalize_text(value: str) -> str:
    normalized = _WS_RE.sub(" ", value.strip())
    if not normalized:
        raise ValueError("value must not be empty")
    return normalized


class ClassBatchBase(BaseModel):
    faculty_id: int
    department_id: int
    name: str
    year: int | None = None

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        return _normalize_text(value).upper()


class ClassBatchCreate(ClassBatchBase):
    name: str | None = None
    faculty_id: int | None = None

    @field_validator("name")
    @classmethod
    def normalize_optional_name_for_create(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _normalize_text(value).upper()


class ClassBatchUpdate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"faculty_id": 1, "department_id": 1, "name": "CIS2202", "year": 2027}
        }
    )

    faculty_id: int | None = None
    department_id: int | None = None
    name: str | None = None
    year: int | None = None

    @field_validator("name")
    @classmethod
    def normalize_optional_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _normalize_text(value).upper()


class ClassBatchRead(ClassBatchBase):
    model_config = ConfigDict(from_attributes=True)

    id: int


class PaginatedClassBatchRead(BaseModel):
    items: list[ClassBatchRead]
    total: int
    skip: int
    limit: int
