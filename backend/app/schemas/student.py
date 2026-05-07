from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class StudentBase(BaseModel):
    student_number: str
    full_name: str
    faculty_id: int = Field(gt=0)
    department_id: int = Field(gt=0)
    embedding_ref: str | None = None


class StudentCreate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "full_name": "Abdimalik Hassan",
                "faculty_id": 1,
                "department_id": 1,
            }
        }
    )

    full_name: str
    faculty_id: int = Field(gt=0)
    department_id: int = Field(gt=0)


class StudentUpdate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "full_name": "Updated Student Name",
                "department_id": 2,
                "embedding_ref": "2201999"
            }
        }
    )

    student_number: str | None = None
    full_name: str | None = None
    faculty_id: int | None = Field(default=None, gt=0)
    department_id: int | None = Field(default=None, gt=0)
    embedding_ref: str | None = None


class StudentRead(StudentBase):
    model_config = ConfigDict(from_attributes=True)

    id: int


class PaginatedStudentRead(BaseModel):
    items: list[StudentRead]
    total: int
    skip: int
    limit: int
