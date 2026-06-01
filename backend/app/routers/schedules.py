from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from sqlalchemy import and_

from app.core.security import require_roles
from app.db.faculty_scope import enforce_faculty_scope, get_optional_faculty_scope_context
from app.db.models import Course, CourseSchedule, CourseScheduleWeekday
from app.db.role_scoped import get_role_scoped_db
from app.utils.weekday_utils import (
    weekday_code,
    decode_weekday_codes,
    decode_weekday_storage,
    encode_weekday_storage,
    parse_weekday,
    parse_weekday_list,
    storage_contains_weekday,
    weekdays_intersect,
)
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


def _course_faculty_id(db: Session, course_id: int) -> int:
    target_course = db.query(Course).filter(Course.id == course_id).first()
    if not target_course:
        raise HTTPException(status_code=404, detail="Course not found")
    return target_course.faculty_id


def _ensure_no_overlap(
    db: Session,
    *,
    course_id: int,
    weekday_storage: int,
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
            Course.faculty_id == target_course.faculty_id,
            and_(CourseSchedule.start_time < end_time, CourseSchedule.end_time > start_time),
        )
    )
    if exclude_schedule_id is not None:
        q = q.filter(CourseSchedule.id != exclude_schedule_id)

    overlap = None
    for row in q.all():
        if weekdays_intersect(row.weekday, weekday_storage):
            overlap = row
            break

    if overlap is not None:
        raise HTTPException(
            status_code=400,
            detail="Schedule overlaps with another course schedule in this faculty/day",
        )


def _ensure_course_not_scheduled_twice_same_day_for_faculty(
    db: Session,
    *,
    course_id: int,
    selected_weekdays: list[int],
    exclude_schedule_id: int | None = None,
) -> None:
    target_course = db.query(Course).filter(Course.id == course_id).first()
    if not target_course:
        raise HTTPException(status_code=404, detail="Course not found")

    query = (
        db.query(CourseSchedule)
        .join(Course, Course.id == CourseSchedule.course_id)
        .filter(Course.id == target_course.id)
    )
    if exclude_schedule_id is not None:
        query = query.filter(CourseSchedule.id != exclude_schedule_id)

    selected_set = set(selected_weekdays)
    conflict_days: set[int] = set()
    existing_days: set[int] = set()

    for row in query.all():
        row_days = set(decode_weekday_storage(row.weekday))
        existing_days.update(row_days)
        conflict_days.update(row_days & selected_set)

    if not conflict_days:
        return

    if existing_days == {1, 2, 3, 4, 5, 6, 7}:
        raise HTTPException(
            status_code=400,
            detail="This course is already scheduled for this faculty on all days.",
        )

    if conflict_days == selected_set:
        raise HTTPException(
            status_code=400,
            detail="This course is already scheduled for this faculty on all selected days.",
        )

    day_codes = [weekday_code(day) for day in selected_weekdays if day in conflict_days]
    day_codes = [code for code in day_codes if code]
    day_summary = ", ".join(day_codes)
    raise HTTPException(
        status_code=400,
        detail=f"This course is already scheduled for this faculty on: {day_summary}.",
    )


def _to_read_model(obj: CourseSchedule) -> CourseScheduleRead:
    weekday_codes = decode_weekday_codes(obj.weekday)
    weekday_count = len(weekday_codes)
    return CourseScheduleRead(
        id=obj.id,
        course_id=obj.course_id,
        weekday=weekday_codes,
        weekday_count=weekday_count,
        weekday_summary=f"{weekday_count} day" if weekday_count == 1 else f"{weekday_count} days",
        start_time=obj.start_time,
        end_time=obj.end_time,
        grace_period_minutes=obj.grace_period_minutes,
    )


def _sync_schedule_weekdays(schedule: CourseSchedule, weekdays: list[int], db: Session | None = None) -> None:
    schedule.weekday = encode_weekday_storage(weekdays)
    # Clear existing rows and flush deletes to the DB *before* inserting new ones.
    # Without the flush, SQLAlchemy may INSERT new rows before DELETing the old ones,
    # which violates the unique constraint on (schedule_id, weekday).
    schedule.weekday_rows.clear()
    if db is not None:
        db.flush()
    schedule.weekday_rows = [CourseScheduleWeekday(weekday=weekday) for weekday in weekdays]


@router.post("", response_model=CourseScheduleRead, dependencies=[Depends(require_roles("FACULTY"))])
def create_schedule(
    payload: CourseScheduleCreate,
    db: Session = Depends(get_role_scoped_db),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    _validate_schedule_window(payload.start_time, payload.end_time)
    try:
        weekdays = parse_weekday_list(payload.weekday)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    weekday_storage = encode_weekday_storage(weekdays)

    if faculty_scope is not None:
        enforce_faculty_scope(_course_faculty_id(db, payload.course_id), faculty_scope)

    _ensure_course_not_scheduled_twice_same_day_for_faculty(
        db,
        course_id=payload.course_id,
        selected_weekdays=weekdays,
    )

    _ensure_no_overlap(
        db,
        course_id=payload.course_id,
        weekday_storage=weekday_storage,
        start_time=payload.start_time,
        end_time=payload.end_time,
    )

    obj = CourseSchedule(
        course_id=payload.course_id,
        start_time=payload.start_time,
        end_time=payload.end_time,
        grace_period_minutes=payload.grace_period_minutes,
    )
    _sync_schedule_weekdays(obj, weekdays)
    db.add(obj)

    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Schedule already exists for this course/day/time") from exc
    db.refresh(obj)
    return _to_read_model(obj)


@router.get("", response_model=PaginatedCourseScheduleRead, dependencies=[Depends(require_roles("FACULTY", "TEACHER", "ACADEMIA"))])
def list_schedules(
    db: Session = Depends(get_role_scoped_db),
    skip: int = Query(default=0, ge=0, description="Number of rows to skip", examples=[0]),
    limit: int = Query(default=50, ge=1, le=200, description="Page size", examples=[20]),
    course_id: int | None = Query(default=None, description="Filter by course id", examples=[1]),
    weekday: int | str | None = Query(
        default=None,
        description="Filter by weekday as 1..7 or day code (sat..fri)",
        examples=["sat"],
    ),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    query = db.query(CourseSchedule)
    if course_id is not None:
        query = query.filter(CourseSchedule.course_id == course_id)
    if faculty_scope is not None:
        if course_id is not None:
            enforce_faculty_scope(_course_faculty_id(db, course_id), faculty_scope)
        else:
            query = query.join(Course, Course.id == CourseSchedule.course_id).filter(Course.faculty_id == faculty_scope.faculty_id)
    rows = query.all()
    if weekday is not None:
        try:
            parsed_weekday = parse_weekday(weekday)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        rows = [row for row in rows if storage_contains_weekday(row.weekday, parsed_weekday)]

    rows.sort(key=lambda row: (row.start_time, row.id))
    total = len(rows)
    page = rows[skip: skip + limit]
    items = [_to_read_model(row) for row in page]
    return {"items": items, "total": total, "skip": skip, "limit": limit}


@router.put("/{schedule_id}", response_model=CourseScheduleRead, dependencies=[Depends(require_roles("FACULTY"))])
def update_schedule(
    schedule_id: int,
    payload: CourseScheduleUpdate,
    db: Session = Depends(get_role_scoped_db),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    obj = db.query(CourseSchedule).filter(CourseSchedule.id == schedule_id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Schedule not found")

    update_data = payload.model_dump(exclude_unset=True)
    next_course_id = update_data.get("course_id", obj.course_id)
    raw_next_weekday = update_data.get("weekday", obj.weekday)
    try:
        parsed_weekdays = parse_weekday_list(raw_next_weekday)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    next_weekday_storage = encode_weekday_storage(parsed_weekdays)
    next_start_time = update_data.get("start_time", obj.start_time)
    next_end_time = update_data.get("end_time", obj.end_time)

    if faculty_scope is not None:
        enforce_faculty_scope(_course_faculty_id(db, next_course_id), faculty_scope)

    _ensure_course_not_scheduled_twice_same_day_for_faculty(
        db,
        course_id=next_course_id,
        selected_weekdays=parsed_weekdays,
        exclude_schedule_id=obj.id,
    )

    _validate_schedule_window(next_start_time, next_end_time)
    _ensure_no_overlap(
        db,
        course_id=next_course_id,
        weekday_storage=next_weekday_storage,
        start_time=next_start_time,
        end_time=next_end_time,
        exclude_schedule_id=obj.id,
    )

    for field, value in update_data.items():
        if field == "weekday":
            value = next_weekday_storage
        setattr(obj, field, value)

    _sync_schedule_weekdays(obj, parsed_weekdays, db=db)

    db.add(obj)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Update conflicts with existing schedule data") from exc
    db.refresh(obj)
    return _to_read_model(obj)


@router.delete("/{schedule_id}", dependencies=[Depends(require_roles("FACULTY"))])
def delete_schedule(
    schedule_id: int,
    db: Session = Depends(get_role_scoped_db),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    obj = db.query(CourseSchedule).filter(CourseSchedule.id == schedule_id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Schedule not found")
    if faculty_scope is not None:
        enforce_faculty_scope(_course_faculty_id(db, obj.course_id), faculty_scope)
    db.delete(obj)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Cannot delete schedule due to related records") from exc
    return {"deleted": True, "schedule_id": schedule_id}
