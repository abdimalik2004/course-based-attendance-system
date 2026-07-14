from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, EmailStr, Field

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
                "phone": "+252 61 234 5678",
                "email": "s.ahmed@university.edu",
                "hire_date": "2023-09-01",
            }
        }
    )

    full_name: str
    role: TeacherRole = TeacherRole.LECTURER
    status: TeacherStatus = TeacherStatus.ACTIVE
    faculty_id: int = Field(gt=0)
    department_id: int = Field(gt=0)
    phone: str | None = Field(default=None, max_length=30)
    email: EmailStr | None = None
    hire_date: date | None = None


class TeacherUpdate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "full_name": "Dr. Updated Name",
                "role": "Professor",
                "status": "On Leave",
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
    phone: str | None = Field(default=None, max_length=30)
    email: EmailStr | None = None
    hire_date: date | None = None


class TeacherRead(TeacherBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int | None = None
    linked_username: str | None = None
    phone: str | None = None
    email: str | None = None
    hire_date: date | None = None
    faculty_name: str | None = None
    department_name: str | None = None


class LinkUserPayload(BaseModel):
    """Payload for PATCH /teachers/{id}/link-user.

    Pass user_id to link an account, or null to unlink the current one.
    """
    user_id: int | None = None


class PaginatedTeacherRead(BaseModel):
    items: list[TeacherRead]
    total: int
    skip: int
    limit: int
