from __future__ import annotations

from datetime import date, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class StudentStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class StudentBase(BaseModel):
    student_number: str
    full_name: str
    faculty_id: int = Field(gt=0)
    department_id: int = Field(gt=0)
    embedding_ref: str | None = None
    status: StudentStatus = StudentStatus.PENDING


class StudentCreate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "full_name": "Abdimalik Hassan",
                "faculty_id": 1,
                "department_id": 1,
                "date_of_birth": "2000-05-14",
                "phone": "+252612345678",
                "email": "abdimalik@example.com",
            }
        }
    )

    full_name: str
    faculty_id: int = Field(gt=0)
    department_id: int = Field(gt=0)
    date_of_birth: date | None = None
    phone: str | None = None
    email: str | None = None


class StudentUpdate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "full_name": "Updated Name",
                "department_id": 2,
                "phone": "+252612345679",
            }
        }
    )

    student_number: str | None = None
    full_name: str | None = None
    faculty_id: int | None = Field(default=None, gt=0)
    department_id: int | None = Field(default=None, gt=0)
    embedding_ref: str | None = None
    status: StudentStatus | None = None
    date_of_birth: date | None = None
    phone: str | None = None
    email: str | None = None


class StudentRead(StudentBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    date_of_birth: date | None = None
    phone: str | None = None
    email: str | None = None
    # Computed from the filesystem dataset folder — always populated by the router
    face_images_count: int = 0
    # Only populated once: when login credentials are first generated
    generated_password: str | None = None


class PaginatedStudentRead(BaseModel):
    items: list[StudentRead]
    total: int
    skip: int
    limit: int


class StudentDashboardStatsRead(BaseModel):
    total_students: int
    new_admissions: int
    pending_admissions: int
    rejected_applications: int


class StudentCapturedImageRead(BaseModel):
    file_name: str
    url: str


class StudentCapturedImagesRead(BaseModel):
    student_id: int
    student_number: str
    image_count: int
    images: list[StudentCapturedImageRead]
