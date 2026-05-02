from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.db.models import TeacherRole, TeacherStatus


class TeacherBase(BaseModel):
    teacher_number: str
    full_name: str
    role: TeacherRole
    status: TeacherStatus
    faculty_id: int = Field(gt=0)
    department_id: int = Field(gt=0)


class TeacherCreate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "full_name": "Dr. Sarah Ahmed",
                "role": "Lecturer",
                "status": "Active",
                "faculty_id": 1,
                "department_id": 1,
            }
        }
    )

    full_name: str
    role: TeacherRole = TeacherRole.LECTURER
    status: TeacherStatus = TeacherStatus.ACTIVE
    faculty_id: int = Field(gt=0)
    department_id: int = Field(gt=0)


class TeacherUpdate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "full_name": "Dr. Updated Name",
                "role": "Professor",
                "status": "Onleave",
                "faculty_id": 1,
                "department_id": 1,
            }
        }
    )

    teacher_number: str | None = None
    full_name: str | None = None
    role: TeacherRole | None = None
    status: TeacherStatus | None = None
    faculty_id: int | None = Field(default=None, gt=0)
    department_id: int | None = Field(default=None, gt=0)


class TeacherRead(TeacherBase):
    model_config = ConfigDict(from_attributes=True)

    id: int


class PaginatedTeacherRead(BaseModel):
    items: list[TeacherRead]
    total: int
    skip: int
    limit: int
