from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field

from app.db.models import AttendanceStatus, SessionStatus, SessionType


class AttendanceFrameRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "session_id": 12,
                "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ...",
            }
        }
    )

    session_id: int = Field(gt=0)
    image: str


class AttendanceRecordRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    student_id: int
    course_id: int
    session_id: int
    status: AttendanceStatus
    confidence: float
    recognized_at: datetime


class AttendanceSessionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    course_id: int
    teacher_id: int | None
    admin_id: int | None
    schedule_id: int | None
    session_date: date
    start_time: datetime
    end_time: datetime | None
    session_type: SessionType
    status: SessionStatus
    course_name: str | None = None
    course_code: str | None = None
    grace_period_minutes: int | None = None


class AttendanceSessionStartRequest(BaseModel):
    course_id: int = Field(gt=0)
    schedule_id: int | None = None
    session_type: SessionType = Field(default=SessionType.LECTURE)


class AttendanceSessionEndRequest(BaseModel):
    session_id: int = Field(gt=0)
