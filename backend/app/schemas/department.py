from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator


_WS_RE = re.compile(r"\s+")


def _normalize_text(value: str) -> str:
    normalized = _WS_RE.sub(" ", value.strip())
    if not normalized:
        raise ValueError("value must not be empty")
    return normalized


class DepartmentBase(BaseModel):
    faculty_id: int = Field(gt=0)
    name: str
    code: str

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        return _normalize_text(value)

    @field_validator("code")
    @classmethod
    def normalize_code(cls, value: str) -> str:
        return _normalize_text(value).upper()


class DepartmentCreate(DepartmentBase):
    faculty_id: int | None = Field(default=None, gt=0)


class DepartmentUpdate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"faculty_id": 1, "name": "Department of Information Technology", "code": "IT"}
        }
    )

    faculty_id: int | None = Field(default=None, gt=0)
    name: str | None = None
    code: str | None = None

    @field_validator("name")
    @classmethod
    def normalize_optional_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _normalize_text(value)

    @field_validator("code")
    @classmethod
    def normalize_optional_code(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _normalize_text(value).upper()


class DepartmentRead(DepartmentBase):
    model_config = ConfigDict(from_attributes=True)

    id: int


class PaginatedDepartmentRead(BaseModel):
    items: list[DepartmentRead]
    total: int
    skip: int
    limit: int