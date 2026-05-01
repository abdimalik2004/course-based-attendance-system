from __future__ import annotations

from datetime import datetime, time

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.db.models import AttendanceSummaryStatus


class AttendanceCreate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "student_id": 1,
                "course_name": "Database Systems",
                "course_code": "CIS320",
                "classes_attended": 12,
                "total_classes": 15,
            }
        }
    )

    student_id: int = Field(gt=0)
    course_name: str = Field(min_length=1, max_length=200)
    course_code: str = Field(min_length=1, max_length=32)
    classes_attended: int = Field(ge=0)
    total_classes: int = Field(gt=0)

    @field_validator("course_name", "course_code")
    @classmethod
    def trim_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value cannot be empty")
        return stripped

    @model_validator(mode="after")
    def validate_attendance_bounds(self) -> "AttendanceCreate":
        if self.classes_attended > self.total_classes:
            raise ValueError("classes_attended cannot exceed total_classes")
        return self


class AttendanceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    student_id: int
    course_name: str
    course_code: str
    classes_attended: int
    total_classes: int
    attendance_percentage: float
    status: AttendanceSummaryStatus
    created_at: datetime


class ScheduleCreate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "student_id": 1,
                "course_name": "Operating Systems",
                "course_code": "CIS410",
                "weekdays": ["Mon", "Wed", "Fri"],
                "start_time": "09:00:00",
                "end_time": "10:30:00",
                "grace_period_minutes": 10,
            }
        }
    )

    student_id: int = Field(gt=0)
    course_name: str = Field(min_length=1, max_length=200)
    course_code: str = Field(min_length=1, max_length=32)
    weekdays: list[str] = Field(min_length=1)
    start_time: time
    end_time: time
    grace_period_minutes: int = Field(default=0, ge=0)

    @field_validator("course_name", "course_code")
    @classmethod
    def trim_text(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("value cannot be empty")
        return stripped

    @field_validator("weekdays")
    @classmethod
    def normalize_weekdays(cls, value: list[str]) -> list[str]:
        allowed = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}
        normalized = [item.strip().title() for item in value]
        if len(set(normalized)) != len(normalized):
            raise ValueError("weekdays must not contain duplicates")
        invalid = [item for item in normalized if item not in allowed]
        if invalid:
            raise ValueError("weekdays must contain valid weekday codes")
        return normalized

    @field_validator("end_time")
    @classmethod
    def validate_time_order(cls, value: time, info) -> time:
        start_time = info.data.get("start_time")
        if start_time is not None and value <= start_time:
            raise ValueError("end_time must be later than start_time")
        return value


class ScheduleResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    student_id: int
    course_name: str
    course_code: str
    weekdays: list[str]
    start_time: time
    end_time: time
    grace_period_minutes: int
    created_at: datetime
