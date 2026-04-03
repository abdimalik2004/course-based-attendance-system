from __future__ import annotations

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


class PaginatedFacultyRead(BaseModel):
    items: list[FacultyRead]
    total: int
    skip: int
    limit: int
