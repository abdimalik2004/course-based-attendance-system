from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field

from app.db.models import AttendanceStatus, SessionStatus


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
    instructor_id: int | None
    schedule_id: int
    session_date: date
    start_time: datetime
    end_time: datetime | None
    status: SessionStatus


class AttendanceSessionStartRequest(BaseModel):
    course_id: int = Field(gt=0)
    schedule_id: int | None = None


class AttendanceSessionEndRequest(BaseModel):
    session_id: int = Field(gt=0)
