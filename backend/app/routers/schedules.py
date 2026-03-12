from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy import and_

from app.core.security import require_roles
from app.db.models import Course, CourseSchedule
from app.db.session import get_db
from app.schemas.schedule import (
    CourseScheduleCreate,
    CourseScheduleRead,
    CourseScheduleUpdate,
    PaginatedCourseScheduleRead,
)


router = APIRouter(prefix="/schedules", tags=["schedules"])


def _validate_schedule_window(start_time, end_time) -> None:
    if start_time >= end_time:
        raise HTTPException(status_code=400, detail="start_time must be earlier than end_time")


def _ensure_no_overlap(
    db: Session,
    *,
    course_id: int,
    weekday: int,
    start_time,
    end_time,
    exclude_schedule_id: int | None = None,
) -> None:
    target_course = db.query(Course).filter(Course.id == course_id).first()
    if not target_course:
        raise HTTPException(status_code=404, detail="Course not found")

    q = (
        db.query(CourseSchedule)
        .join(Course, Course.id == CourseSchedule.course_id)
        .filter(
            Course.class_batch_id == target_course.class_batch_id,
            CourseSchedule.weekday == weekday,
            and_(CourseSchedule.start_time < end_time, CourseSchedule.end_time > start_time),
        )
    )
    if exclude_schedule_id is not None:
        q = q.filter(CourseSchedule.id != exclude_schedule_id)

    overlap = q.first()
    if overlap:
        raise HTTPException(
            status_code=400,
            detail="Schedule overlaps with another class schedule in this batch/day",
        )


@router.post("", response_model=CourseScheduleRead, dependencies=[Depends(require_roles("FACULTY_ADMIN"))])
def create_schedule(payload: CourseScheduleCreate, db: Session = Depends(get_db)):
    _validate_schedule_window(payload.start_time, payload.end_time)
    _ensure_no_overlap(
        db,
        course_id=payload.course_id,
        weekday=payload.weekday,
        start_time=payload.start_time,
        end_time=payload.end_time,
    )

    obj = CourseSchedule(**payload.model_dump())
    db.add(obj)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Schedule already exists for this course/day/time") from exc
    db.refresh(obj)
    return obj


@router.get("", response_model=PaginatedCourseScheduleRead, dependencies=[Depends(require_roles("FACULTY_ADMIN", "TEACHER", "ACADEMIA"))])
def list_schedules(
    db: Session = Depends(get_db),
    skip: int = Query(default=0, ge=0, description="Number of rows to skip", examples=[0]),
    limit: int = Query(default=50, ge=1, le=200, description="Page size", examples=[20]),
    course_id: int | None = Query(default=None, description="Filter by course id", examples=[1]),
    weekday: int | None = Query(default=None, ge=1, le=7, description="Filter by weekday (1=Saturday, 7=Friday)", examples=[1]),
):
    query = db.query(CourseSchedule)
    if course_id is not None:
        query = query.filter(CourseSchedule.course_id == course_id)
    if weekday is not None:
        query = query.filter(CourseSchedule.weekday == weekday)
    total = query.count()
    items = query.order_by(CourseSchedule.weekday, CourseSchedule.start_time).offset(skip).limit(limit).all()
    return {"items": items, "total": total, "skip": skip, "limit": limit}


@router.put("/{schedule_id}", response_model=CourseScheduleRead, dependencies=[Depends(require_roles("FACULTY_ADMIN"))])
def update_schedule(schedule_id: int, payload: CourseScheduleUpdate, db: Session = Depends(get_db)):
    obj = db.query(CourseSchedule).filter(CourseSchedule.id == schedule_id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Schedule not found")

    update_data = payload.model_dump(exclude_unset=True)
    next_course_id = update_data.get("course_id", obj.course_id)
    next_weekday = update_data.get("weekday", obj.weekday)
    next_start_time = update_data.get("start_time", obj.start_time)
    next_end_time = update_data.get("end_time", obj.end_time)

    _validate_schedule_window(next_start_time, next_end_time)
    _ensure_no_overlap(
        db,
        course_id=next_course_id,
        weekday=next_weekday,
        start_time=next_start_time,
        end_time=next_end_time,
        exclude_schedule_id=obj.id,
    )

    for field, value in update_data.items():
        setattr(obj, field, value)

    db.add(obj)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Update conflicts with existing schedule data") from exc
    db.refresh(obj)
    return obj


@router.delete("/{schedule_id}", dependencies=[Depends(require_roles("FACULTY_ADMIN"))])
def delete_schedule(schedule_id: int, db: Session = Depends(get_db)):
    obj = db.query(CourseSchedule).filter(CourseSchedule.id == schedule_id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Schedule not found")
    db.delete(obj)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Cannot delete schedule due to related records") from exc
    return {"deleted": True, "schedule_id": schedule_id}
