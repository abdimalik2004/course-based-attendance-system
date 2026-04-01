from __future__ import annotations

from datetime import time

from pydantic import BaseModel, ConfigDict, Field


class CourseScheduleCreate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "course_id": 1,
                "weekday": ["sat", "sun", "mon", "tue", "wed", "thu", "fri"],
                "start_time": "09:00:00",
                "end_time": "11:00:00",
                "grace_period_minutes": 10,
            }
        }
    )

    course_id: int
    weekday: int | str | list[int | str] = Field(
        default_factory=lambda: ["sat", "sun", "mon", "tue", "wed", "thu", "fri"],
        description="Weekday as 1..7 (1=Saturday) or day code(s) like 'sat' or 'sat,sun'",
        examples=[["sat", "sun", "mon", "tue", "wed", "thu", "fri"]],
    )
    start_time: time
    end_time: time
    grace_period_minutes: int = Field(default=10, ge=0, le=60)


class CourseScheduleRead(BaseModel):
    id: int
    course_id: int
    weekday: list[str]
    weekday_count: int
    weekday_summary: str
    start_time: time
    end_time: time
    grace_period_minutes: int


class CourseScheduleUpdate(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "weekday": "mon",
                "start_time": "09:00:00",
                "end_time": "11:00:00",
                "grace_period_minutes": 15
            }
        }
    )

    course_id: int | None = None
    weekday: int | str | list[int | str] | None = Field(default=None)
    start_time: time | None = None
    end_time: time | None = None
    grace_period_minutes: int | None = Field(default=None, ge=0, le=60)


class PaginatedCourseScheduleRead(BaseModel):
    items: list[CourseScheduleRead]
    total: int
    skip: int
    limit: int
