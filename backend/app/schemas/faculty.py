from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class FacultyBase(BaseModel):
    name: str
    code: str
    years: int = Field(ge=3)


class FacultyCreate(FacultyBase):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "name": "Faculty of Computer Science",
                "code": "FCS",
                "years": 4,
            }
        },
    )


class FacultyUpdate(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {"name": "Faculty of Engineering", "code": "FOE", "years": 4}
        }
    )

    name: str | None = None
    code: str | None = None
    years: int | None = Field(default=None, ge=3)


class FacultyRead(FacultyBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    semesters: int


class PaginatedFacultyRead(BaseModel):
    items: list[FacultyRead]
    total: int
    skip: int
    limit: int
