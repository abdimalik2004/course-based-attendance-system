from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator


_WS_RE = re.compile(r"\s+")


def _normalize_text(value: str) -> str:
    normalized = _WS_RE.sub(" ", value.strip())
    if not normalized:
        raise ValueError("value must not be empty")
    return normalized


class ClassBatchBase(BaseModel):
    faculty_id: int = Field(gt=0)
    department_id: int = Field(gt=0)
    name: str
    year: int | None = None

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        return _normalize_text(value).upper()


class ClassBatchCreate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"faculty_id": 1, "department_id": 1, "year": 2026}
        }
    )

    faculty_id: int = Field(gt=0)
    department_id: int = Field(gt=0)
    year: int | None = None


class ClassBatchUpdate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"faculty_id": 1, "department_id": 1, "year": 2027}
        }
    )

    faculty_id: int | None = Field(default=None, gt=0)
    department_id: int | None = Field(default=None, gt=0)
    year: int | None = None


class ClassBatchRead(ClassBatchBase):
    model_config = ConfigDict(from_attributes=True)

    id: int


class PaginatedClassBatchRead(BaseModel):
    items: list[ClassBatchRead]
    total: int
    skip: int
    limit: int
