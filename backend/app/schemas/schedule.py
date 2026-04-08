from __future__ import annotations

from datetime import time
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


WeekdayCode = Literal["sat", "sun", "mon", "tue", "wed", "thu", "fri"]


class CourseScheduleCreate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "course_id": 1,
                "start_time": "09:00:00",
                "end_time": "11:00:00",
                "grace_period_minutes": 10,
                "weekday": ["sat", "sun", "mon", "tue", "wed", "thu", "fri"],
            }
        }
    )

    course_id: int = Field(gt=0)
    start_time: time
    end_time: time
    grace_period_minutes: int = Field(default=10, ge=0, le=60)
    weekday: list[WeekdayCode] = Field(min_length=1)

    @field_validator("weekday")
    @classmethod
    def unique_weekdays(cls, value: list[WeekdayCode]) -> list[WeekdayCode]:
        if len(set(value)) != len(value):
            raise ValueError("weekday values must be unique")
        return value


class CourseScheduleRead(BaseModel):
    id: int
    course_id: int = Field(gt=0)
    weekday: list[WeekdayCode]
    weekday_count: int
    weekday_summary: str
    start_time: time
    end_time: time
    grace_period_minutes: int


class CourseScheduleUpdate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "weekday": ["mon"],
                "start_time": "09:00:00",
                "end_time": "11:00:00",
                "grace_period_minutes": 15
            }
        }
    )

    course_id: int | None = Field(default=None, gt=0)
    weekday: list[WeekdayCode] | None = Field(default=None, min_length=1)
    start_time: time | None = None
    end_time: time | None = None
    grace_period_minutes: int | None = Field(default=None, ge=0, le=60)

    @field_validator("weekday")
    @classmethod
    def unique_optional_weekdays(cls, value: list[WeekdayCode] | None) -> list[WeekdayCode] | None:
        if value is not None and len(set(value)) != len(value):
            raise ValueError("weekday values must be unique")
        return value


class PaginatedCourseScheduleRead(BaseModel):
    items: list[CourseScheduleRead]
    total: int
    skip: int
    limit: int
