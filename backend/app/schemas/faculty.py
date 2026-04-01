from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class FacultyBase(BaseModel):
    name: str
    code: str


class FacultyCreate(FacultyBase):
    pass


class FacultyUpdate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"name": "Faculty of Engineering", "code": "FOE"}
        }
    )

    name: str | None = None
    code: str | None = None


class FacultyRead(FacultyBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    tenant_db_name: str | None = None
    tenant_db_provisioned_at: datetime | None = None


class PaginatedFacultyRead(BaseModel):
    items: list[FacultyRead]
    total: int
    skip: int
    limit: int
