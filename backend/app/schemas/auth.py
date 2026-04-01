from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


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
                "username": "teacher2",
                "password": "teacher123",
                "email": "teacher2@university.edu",
                "faculty_id": 1,
                "role_names": ["TEACHER"],
            }
        }
    )

    username: str
    password: str
    email: str | None = None
    faculty_id: int | None = None
    role_names: list[str] = Field(default_factory=list)


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    email: str | None
    is_active: bool
    faculty_id: int | None
