from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CourseBase(BaseModel):
    faculty_id: int = Field(gt=0)
    department_id: int = Field(gt=0)
    code: str
    title: str


class CourseCreate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "faculty_id": 1,
                "department_id": 1,
                "title": "Engineering Mathematics"
            }
        }
    )

    faculty_id: int = Field(gt=0)
    department_id: int = Field(gt=0)
    title: str


class CourseUpdate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"faculty_id": 1, "department_id": 1, "title": "Advanced Backend"}
        }
    )

    faculty_id: int | None = Field(default=None, gt=0)
    department_id: int | None = Field(default=None, gt=0)
    title: str | None = None


class CourseRead(CourseBase):
    model_config = ConfigDict(from_attributes=True)

    id: int


class PaginatedCourseRead(BaseModel):
    items: list[CourseRead]
    total: int
    skip: int
    limit: int


class CourseAssignmentCreate(BaseModel):
    course_id: int = Field(gt=0)
    teacher_id: int = Field(gt=0)
    is_primary: bool = False
