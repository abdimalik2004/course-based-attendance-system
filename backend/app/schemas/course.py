from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CourseBase(BaseModel):
    class_batch_id: int = Field(gt=0)
    code: str
    title: str


class CourseCreate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "class_batch_id": 1,
                "title": "Engineering Mathematics"
            }
        }
    )

    class_batch_id: int = Field(gt=0)
    title: str


class CourseUpdate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"code": "CSC500", "title": "Advanced Backend", "class_batch_id": 1}
        }
    )

    class_batch_id: int | None = Field(default=None, gt=0)
    code: str | None = None
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
