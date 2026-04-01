from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict

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

    session_id: int
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
    schedule_id: int
    session_date: date
    start_time: datetime
    end_time: datetime
    status: SessionStatus
