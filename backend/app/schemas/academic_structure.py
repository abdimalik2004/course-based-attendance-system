from __future__ import annotations

from datetime import date, datetime
import re

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.db.models import AcademicYearStatus


_WS_RE = re.compile(r"\s+")


def _normalize_text(value: str) -> str:
    normalized = _WS_RE.sub(" ", value.strip())
    if not normalized:
        raise ValueError("value must not be empty")
    return normalized


class AcademicYearBase(BaseModel):
    academic_year: str
    term_name: str
    start_date: date
    end_date: date
    status: AcademicYearStatus = AcademicYearStatus.DRAFT

    @field_validator("academic_year")
    @classmethod
    def normalize_academic_year(cls, value: str) -> str:
        return _normalize_text(value)

    @field_validator("term_name")
    @classmethod
    def normalize_term_name(cls, value: str) -> str:
        return _normalize_text(value)

    @model_validator(mode="after")
    def validate_dates(self):
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be later than start_date")
        return self


class AcademicYearCreate(AcademicYearBase):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "academic_year": "2025-2026",
                "term_name": "Semester 1",
                "start_date": "2025-09-01",
                "end_date": "2026-01-15",
                "status": "draft",
            }
        },
    )


class AcademicYearRead(AcademicYearBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime | None = None


class PaginatedAcademicYearRead(BaseModel):
    items: list[AcademicYearRead]
    total: int
    skip: int
    limit: int


class CourseSemesterAssignmentBase(BaseModel):
    course_id: int = Field(gt=0)
    faculty_id: int = Field(gt=0)
    department_id: int = Field(gt=0)
    academic_year_id: int = Field(gt=0)


class CourseSemesterAssignmentCreate(CourseSemesterAssignmentBase):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "course_id": 1,
                "faculty_id": 1,
                "department_id": 1,
                "academic_year_id": 1,
            }
        },
    )


class CourseSemesterAssignmentRead(CourseSemesterAssignmentBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime | None = None


class PaginatedCourseSemesterAssignmentRead(BaseModel):
    items: list[CourseSemesterAssignmentRead]
    total: int
    skip: int
    limit: int


class ClassCourseAssignmentBase(BaseModel):
    class_id: int = Field(gt=0)
    course_id: int = Field(gt=0)
    faculty_id: int = Field(gt=0)
    department_id: int = Field(gt=0)


class ClassCourseAssignmentCreate(ClassCourseAssignmentBase):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "class_id": 1,
                "course_id": 1,
                "faculty_id": 1,
                "department_id": 1,
            }
        },
    )


class ClassCourseAssignmentRead(ClassCourseAssignmentBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime | None = None


class PaginatedClassCourseAssignmentRead(BaseModel):
    items: list[ClassCourseAssignmentRead]
    total: int
    skip: int
    limit: int
