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
                "email": "email@university.edu",
                "username": "Username",
                "password": "Password",
                "role_names": ["YOUR_ROLE"],
            }
        }
    )

    email: str
    username: str
    password: str
    role_names: list[str] = Field(min_length=1)


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
    role_names: list[str]
