from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Token(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
                "token_type": "bearer",
            }
        }
    )

    access_token: str
    token_type: str = "bearer"


class TokenPair(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class TokenData(BaseModel):
    username: str


class LoginRequest(BaseModel):
    username: str
    password: str


class UserCreate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "email": "email@university.edu",
                "username": "Username",
                "password": "Password",
                "role_names": ["YOUR_ROLE"],
                "faculty_id": 0,
            }
        }
    )

    email: str
    username: str
    password: str
    role_names: list[str] = Field(min_length=1)
    faculty_id: int | None = Field(default=None, gt=0, description="Required when role_names includes FACULTY")
    teacher_id: int | None = Field(default=None, gt=0, description="Required when role_names includes TEACHER")
    student_id: int | None = Field(default=None, gt=0, description="Required when role_names includes STUDENT")

    @model_validator(mode="after")
    def _faculty_required_for_faculty_role(self) -> "UserCreate":
        normalized_roles = {role_name.strip().upper() for role_name in self.role_names if role_name.strip()}
        if "FACULTY" in normalized_roles and self.faculty_id is None:
            raise ValueError("faculty_id is required when role_names includes FACULTY")
        if "TEACHER" in normalized_roles and self.teacher_id is None:
            raise ValueError("teacher_id is required when role_names includes TEACHER")
        if "STUDENT" in normalized_roles and self.student_id is None:
            raise ValueError("student_id is required when role_names includes STUDENT")
        return self


class RoleCreate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "LIBRARY",
            }
        }
    )

    name: str = Field(min_length=1, max_length=64)


class RoleRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str


class ChangePasswordRequest(BaseModel):
    current_password: str = Field(min_length=1)
    new_password: str = Field(min_length=8, description="New password must be at least 8 characters")


class ResetDatabaseRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "confirmation": "RESET",
            }
        }
    )

    confirmation: str


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    email: str | None
    is_active: bool
    faculty_id: int | None
    teacher_id: int | None = None
    student_id: int | None = None
    student_number: str | None = None
    role_names: list[str]
    profile_image_url: str | None = None
    full_name: str | None = None
