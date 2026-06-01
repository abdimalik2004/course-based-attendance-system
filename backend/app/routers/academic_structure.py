from __future__ import annotations

from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.security import get_current_user, require_roles
from app.db.faculty_scope import enforce_faculty_scope, get_optional_faculty_scope_context
from app.db.models import AcademicYear, AcademicYearStatus, ClassCourseAssignment, CourseSemesterAssignment, User
from app.db.role_scoped import get_role_scoped_db
from app.schemas.academic_structure import (
    AcademicYearCreate,
    AcademicYearRead,
    AcademicYearUpdate,
    ClassCourseAssignmentCreate,
    ClassCourseAssignmentRead,
    CourseSemesterAssignmentCreate,
    CourseSemesterAssignmentRead,
    CourseSemesterAssignmentUpdate,
    PaginatedAcademicYearRead,
    PaginatedClassCourseAssignmentRead,
    PaginatedCourseSemesterAssignmentRead,
)
from app.utils.activity_logger import log_activity
from app.utils.db_conflicts import classify_integrity_error, integrity_error_mentions
from app.utils.organization import (
    ensure_class_batch_matches_faculty_and_department,
    ensure_department_belongs_to_faculty,
    get_class_batch_or_404,
    get_course_or_404,
    get_department_or_404,
    get_faculty_or_404,
)


router = APIRouter(prefix="/academic-structure", tags=["academic-structure"])


def _sync_semester_statuses(db: Session) -> None:
    """Auto-transition any ACTIVE semester whose end_date has already passed to INACTIVE.

    Called on every list/start-session request so no cron job is needed.
    """
    today = date.today()
    expired = (
        db.query(AcademicYear)
        .filter(
            AcademicYear.status == AcademicYearStatus.ACTIVE,
            AcademicYear.end_date < today,
        )
        .all()
    )
    if expired:
        for ay in expired:
            ay.status = AcademicYearStatus.INACTIVE
        db.commit()


def _academic_year_by_id_or_404(db: Session, academic_year_id: int) -> AcademicYear:
    academic_year = db.query(AcademicYear).filter(AcademicYear.id == academic_year_id).first()
    if academic_year is None:
        raise HTTPException(status_code=404, detail="Academic year not found")
    return academic_year


def _academic_year_duplicate_exists(
    db: Session,
    academic_year: str,
    term_name: str,
    exclude_id: int | None = None,
) -> bool:
    """Return True if a row with the same (academic_year, term_name) already exists."""
    query = db.query(AcademicYear.id).filter(
        func.lower(func.trim(AcademicYear.academic_year)) == academic_year.strip().lower(),
        func.lower(func.trim(AcademicYear.term_name)) == term_name.strip().lower(),
    )
    if exclude_id is not None:
        query = query.filter(AcademicYear.id != exclude_id)
    return db.query(query.exists()).scalar()


def _academic_year_active_exists(db: Session, exclude_id: int | None = None) -> bool:
    query = db.query(AcademicYear.id).filter(AcademicYear.status == AcademicYearStatus.ACTIVE)
    if exclude_id is not None:
        query = query.filter(AcademicYear.id != exclude_id)
    return db.query(query.exists()).scalar()


def _course_semester_assignment_duplicate_exists(
    db: Session,
    *,
    course_id: int,
    faculty_id: int,
    department_id: int,
    academic_year_id: int,
    exclude_id: int | None = None,
) -> bool:
    query = db.query(CourseSemesterAssignment.id).filter(
        CourseSemesterAssignment.course_id == course_id,
        CourseSemesterAssignment.faculty_id == faculty_id,
        CourseSemesterAssignment.department_id == department_id,
        CourseSemesterAssignment.academic_year_id == academic_year_id,
    )
    if exclude_id is not None:
        query = query.filter(CourseSemesterAssignment.id != exclude_id)
    return db.query(query.exists()).scalar()


def _academic_year_by_semester_or_404(db: Session, semester: int) -> AcademicYear:
    term_name = f"Semester {semester}".strip().lower()
    query = db.query(AcademicYear).filter(func.lower(func.trim(AcademicYear.term_name)) == term_name)
    academic_year = query.filter(AcademicYear.status == AcademicYearStatus.ACTIVE).first()
    if academic_year is None:
        academic_year = query.order_by(AcademicYear.id.desc()).first()
    if academic_year is None:
        raise HTTPException(status_code=404, detail="Academic year for requested semester not found")
    return academic_year


def _class_course_assignment_duplicate_exists(
    db: Session,
    *,
    class_id: int,
    course_id: int,
    faculty_id: int,
    department_id: int,
    exclude_id: int | None = None,
) -> bool:
    query = db.query(ClassCourseAssignment.id).filter(
        ClassCourseAssignment.class_id == class_id,
        ClassCourseAssignment.course_id == course_id,
        ClassCourseAssignment.faculty_id == faculty_id,
        ClassCourseAssignment.department_id == department_id,
    )
    if exclude_id is not None:
        query = query.filter(ClassCourseAssignment.id != exclude_id)
    return db.query(query.exists()).scalar()


@router.post("/academic-years", response_model=AcademicYearRead, dependencies=[Depends(require_roles("ACADEMIA"))])
def create_academic_year(
    payload: AcademicYearCreate,
    db: Session = Depends(get_role_scoped_db),
    current_user: User = Depends(get_current_user),
):
    if _academic_year_duplicate_exists(db, payload.academic_year, payload.term_name):
        raise HTTPException(
            status_code=409,
            detail=f"A term '{payload.term_name}' already exists for academic year '{payload.academic_year}'.",
        )
    if payload.status == AcademicYearStatus.ACTIVE and _academic_year_active_exists(db):
        raise HTTPException(status_code=409, detail="Only one academic year can be active at a time")

    obj = AcademicYear(**payload.model_dump())
    db.add(obj)
    try:
        db.commit()
    except ValueError as exc:
        db.rollback()
        message = str(exc)
        if "active" in message:
            raise HTTPException(status_code=409, detail=message) from exc
        raise HTTPException(status_code=400, detail=message) from exc
    except IntegrityError as exc:
        db.rollback()
        error_kind = classify_integrity_error(exc)
        if error_kind == "duplicate" and (
            integrity_error_mentions(exc, "uq_academic_years_year_term", "academic_years.academic_year")
            or _academic_year_duplicate_exists(db, payload.academic_year, payload.term_name)
        ):
            raise HTTPException(
                status_code=409,
                detail=f"A term '{payload.term_name}' already exists for academic year '{payload.academic_year}'.",
            ) from exc
        if error_kind == "duplicate" and _academic_year_active_exists(db):
            raise HTTPException(status_code=409, detail="Only one academic year can be active at a time") from exc
        raise HTTPException(status_code=400, detail="Academic year could not be created due to invalid data") from exc
    db.refresh(obj)
    log_activity(action=f"Academic Term Created - {obj.term_name} ({obj.academic_year})", user=current_user, db=db)
    return obj


@router.get(
    "/academic-years",
    response_model=PaginatedAcademicYearRead,
    dependencies=[Depends(require_roles("ACADEMIA", "FACULTY", "TEACHER"))],
)
def list_academic_years(
    db: Session = Depends(get_role_scoped_db),
    skip: int = Query(default=0, ge=0, description="Number of rows to skip", examples=[0]),
    limit: int = Query(default=50, ge=1, le=200, description="Page size", examples=[20]),
    academic_year: str | None = Query(default=None, description="Filter by academic year", examples=["2025-2026"]),
    status: AcademicYearStatus | None = Query(default=None, description="Filter by status"),
):
    # Auto-deactivate any semester whose end_date has passed
    _sync_semester_statuses(db)
    query = db.query(AcademicYear)
    if academic_year is not None:
        pattern = academic_year.strip().lower()
        query = query.filter(func.lower(func.trim(AcademicYear.academic_year)) == pattern)
    if status is not None:
        query = query.filter(AcademicYear.status == status)
    total = query.count()
    items = query.order_by(AcademicYear.start_date.desc(), AcademicYear.id.desc()).offset(skip).limit(limit).all()
    return {"items": items, "total": total, "skip": skip, "limit": limit}


@router.get(
    "/academic-years/{academic_year_id}",
    response_model=AcademicYearRead,
    dependencies=[Depends(require_roles("ACADEMIA", "FACULTY", "TEACHER"))],
)
def get_academic_year(academic_year_id: int, db: Session = Depends(get_role_scoped_db)):
    return _academic_year_by_id_or_404(db, academic_year_id)


@router.put(
    "/academic-years/{academic_year_id}",
    response_model=AcademicYearRead,
    dependencies=[Depends(require_roles("ACADEMIA"))],
)
def update_academic_year(
    academic_year_id: int,
    payload: AcademicYearUpdate,
    db: Session = Depends(get_role_scoped_db),
):
    obj = _academic_year_by_id_or_404(db, academic_year_id)

    # Apply only the fields that were sent
    if payload.academic_year is not None:
        obj.academic_year = payload.academic_year
    if payload.term_name is not None:
        obj.term_name = payload.term_name
    if payload.start_date is not None:
        obj.start_date = payload.start_date
    if payload.end_date is not None:
        obj.end_date = payload.end_date
    if payload.status is not None:
        # Only one semester can be ACTIVE at a time
        if payload.status == AcademicYearStatus.ACTIVE and _academic_year_active_exists(db, exclude_id=academic_year_id):
            raise HTTPException(status_code=409, detail="Only one academic year can be active at a time")
        obj.status = payload.status

    # Validate date order after applying updates
    if obj.end_date <= obj.start_date:
        raise HTTPException(status_code=400, detail="end_date must be later than start_date")

    # Check uniqueness of (academic_year, term_name)
    if _academic_year_duplicate_exists(db, obj.academic_year, obj.term_name, exclude_id=academic_year_id):
        raise HTTPException(
            status_code=409,
            detail=f"A term '{obj.term_name}' already exists for academic year '{obj.academic_year}'.",
        )

    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Could not update academic year due to a conflict") from exc
    db.refresh(obj)
    return obj


@router.delete("/academic-years/{academic_year_id}", dependencies=[Depends(require_roles("ACADEMIA"))])
def delete_academic_year(academic_year_id: int, db: Session = Depends(get_role_scoped_db)):
    obj = _academic_year_by_id_or_404(db, academic_year_id)
    db.delete(obj)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Cannot delete academic year due to related records") from exc
    return {"deleted": True, "academic_year_id": academic_year_id}


@router.post(
    "/course-semester-assignments",
    response_model=CourseSemesterAssignmentRead,
    dependencies=[Depends(require_roles("ACADEMIA"))],
)
def create_course_semester_assignment(
    payload: CourseSemesterAssignmentCreate,
    db: Session = Depends(get_role_scoped_db),
    faculty_scope=Depends(get_optional_faculty_scope_context),
    current_user: User = Depends(get_current_user),
):
    faculty = get_faculty_or_404(db, payload.faculty_id)
    if faculty_scope is not None:
        enforce_faculty_scope(faculty.id, faculty_scope)

    course = get_course_or_404(db, payload.course_id)
    department = get_department_or_404(db, payload.department_id)
    academic_year = _academic_year_by_semester_or_404(db, payload.semester)

    if course.faculty_id != faculty.id:
        raise HTTPException(status_code=400, detail="Course does not belong to faculty")
    ensure_department_belongs_to_faculty(department, faculty.id)

    if _course_semester_assignment_duplicate_exists(
        db,
        course_id=course.id,
        faculty_id=faculty.id,
        department_id=department.id,
        academic_year_id=academic_year.id,
    ):
        raise HTTPException(status_code=409, detail="Course already assigned for this semester in this faculty/department")

    obj = CourseSemesterAssignment(
        course_id=course.id,
        faculty_id=faculty.id,
        department_id=department.id,
        academic_year_id=academic_year.id,
    )
    db.add(obj)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        error_kind = classify_integrity_error(exc)
        if error_kind == "duplicate" and _course_semester_assignment_duplicate_exists(
            db,
            course_id=course.id,
            faculty_id=faculty.id,
            department_id=department.id,
            academic_year_id=academic_year.id,
        ):
            raise HTTPException(status_code=409, detail="Course already assigned for this semester in this faculty/department") from exc
        if error_kind == "foreign_key":
            raise HTTPException(status_code=400, detail="Course semester assignment references invalid data") from exc
        raise HTTPException(status_code=400, detail="Course semester assignment could not be created") from exc
    db.refresh(obj)
    log_activity(action=f"Course Assigned to Semester - {course.title} → {academic_year.term_name}", user=current_user, db=db)
    return obj


@router.get(
    "/course-semester-assignments",
    response_model=PaginatedCourseSemesterAssignmentRead,
    dependencies=[Depends(require_roles("ACADEMIA", "FACULTY", "TEACHER"))],
)
def list_course_semester_assignments(
    db: Session = Depends(get_role_scoped_db),
    skip: int = Query(default=0, ge=0, description="Number of rows to skip", examples=[0]),
    limit: int = Query(default=50, ge=1, le=200, description="Page size", examples=[20]),
    faculty_id: int | None = Query(default=None, description="Filter by faculty id", examples=[1]),
    department_id: int | None = Query(default=None, description="Filter by department id", examples=[1]),
    course_id: int | None = Query(default=None, description="Filter by course id", examples=[1]),
    semester: int | None = Query(default=None, description="Filter by semester", examples=[1]),
    faculty_scope=Depends(get_optional_faculty_scope_context),
):
    query = db.query(CourseSemesterAssignment).join(AcademicYear, AcademicYear.id == CourseSemesterAssignment.academic_year_id)
    if faculty_scope is not None:
        query = query.filter(CourseSemesterAssignment.faculty_id == faculty_scope.faculty_id)
    if faculty_id is not None:
        query = query.filter(CourseSemesterAssignment.faculty_id == faculty_id)
    if department_id is not None:
        query = query.filter(CourseSemesterAssignment.department_id == department_id)
    if course_id is not None:
        query = query.filter(CourseSemesterAssignment.course_id == course_id)
    if semester is not None:
        query = query.filter(func.lower(func.trim(AcademicYear.term_name)) == f"semester {semester}")
    total = query.count()
    items = query.order_by(CourseSemesterAssignment.created_at.desc(), CourseSemesterAssignment.id.desc()).offset(skip).limit(limit).all()
    return {"items": items, "total": total, "skip": skip, "limit": limit}


@router.get(
    "/course-semester-assignments/{assignment_id}",
    response_model=CourseSemesterAssignmentRead,
    dependencies=[Depends(require_roles("ACADEMIA", "FACULTY", "TEACHER"))],
)
def get_course_semester_assignment(assignment_id: int, db: Session = Depends(get_role_scoped_db)):
    assignment = db.query(CourseSemesterAssignment).filter(CourseSemesterAssignment.id == assignment_id).first()
    if assignment is None:
        raise HTTPException(status_code=404, detail="Course semester assignment not found")
    return assignment


@router.put(
    "/course-semester-assignments/{assignment_id}",
    response_model=CourseSemesterAssignmentRead,
    dependencies=[Depends(require_roles("ACADEMIA"))],
)
def update_course_semester_assignment(
    assignment_id: int,
    payload: CourseSemesterAssignmentUpdate,
    db: Session = Depends(get_role_scoped_db),
):
    assignment = db.query(CourseSemesterAssignment).filter(CourseSemesterAssignment.id == assignment_id).first()
    if assignment is None:
        raise HTTPException(status_code=404, detail="Course semester assignment not found")

    # Resolve the target academic year by semester number
    new_academic_year = _academic_year_by_semester_or_404(db, payload.semester)

    # Avoid a no-op commit when already assigned to that year
    if assignment.academic_year_id == new_academic_year.id:
        return assignment

    # Guard against duplicate (same course/faculty/dept already has this academic_year)
    if _course_semester_assignment_duplicate_exists(
        db,
        course_id=assignment.course_id,
        faculty_id=assignment.faculty_id,
        department_id=assignment.department_id,
        academic_year_id=new_academic_year.id,
        exclude_id=assignment_id,
    ):
        raise HTTPException(
            status_code=409,
            detail="This course is already assigned to that semester in the same faculty/department.",
        )

    assignment.academic_year_id = new_academic_year.id
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Could not update course semester assignment") from exc
    db.refresh(assignment)
    return assignment


@router.delete("/course-semester-assignments/{assignment_id}", dependencies=[Depends(require_roles("ACADEMIA"))])
def delete_course_semester_assignment(assignment_id: int, db: Session = Depends(get_role_scoped_db)):
    assignment = db.query(CourseSemesterAssignment).filter(CourseSemesterAssignment.id == assignment_id).first()
    if assignment is None:
        raise HTTPException(status_code=404, detail="Course semester assignment not found")
    db.delete(assignment)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Cannot delete course semester assignment") from exc
    return {"deleted": True, "assignment_id": assignment_id}


@router.post(
    "/class-course-assignments",
    response_model=ClassCourseAssignmentRead,
    dependencies=[Depends(require_roles("ACADEMIA"))],
)
def create_class_course_assignment(
    payload: ClassCourseAssignmentCreate,
    db: Session = Depends(get_role_scoped_db),
    faculty_scope=Depends(get_optional_faculty_scope_context),
):
    faculty = get_faculty_or_404(db, payload.faculty_id)
    if faculty_scope is not None:
        enforce_faculty_scope(faculty.id, faculty_scope)

    class_batch = get_class_batch_or_404(db, payload.class_id)
    course = get_course_or_404(db, payload.course_id)
    department = get_department_or_404(db, payload.department_id)

    ensure_class_batch_matches_faculty_and_department(
        class_batch,
        faculty_id=faculty.id,
        department_id=department.id,
    )
    if course.faculty_id != faculty.id:
        raise HTTPException(status_code=400, detail="Course does not belong to faculty")

    if _class_course_assignment_duplicate_exists(
        db,
        class_id=class_batch.id,
        course_id=course.id,
        faculty_id=faculty.id,
        department_id=department.id,
    ):
        raise HTTPException(status_code=409, detail="Class already assigned to this course in this faculty/department")

    obj = ClassCourseAssignment(
        class_id=class_batch.id,
        course_id=course.id,
        faculty_id=faculty.id,
        department_id=department.id,
    )
    db.add(obj)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        error_kind = classify_integrity_error(exc)
        if error_kind == "duplicate" and _class_course_assignment_duplicate_exists(
            db,
            class_id=class_batch.id,
            course_id=course.id,
            faculty_id=faculty.id,
            department_id=department.id,
        ):
            raise HTTPException(status_code=409, detail="Class already assigned to this course in this faculty/department") from exc
        if error_kind == "foreign_key":
            raise HTTPException(status_code=400, detail="Class course assignment references invalid data") from exc
        raise HTTPException(status_code=400, detail="Class course assignment could not be created") from exc
    db.refresh(obj)
    return obj


@router.get(
    "/class-course-assignments",
    response_model=PaginatedClassCourseAssignmentRead,
    dependencies=[Depends(require_roles("ACADEMIA", "FACULTY", "TEACHER"))],
)
def list_class_course_assignments(
    db: Session = Depends(get_role_scoped_db),
    skip: int = Query(default=0, ge=0, description="Number of rows to skip", examples=[0]),
    limit: int = Query(default=50, ge=1, le=200, description="Page size", examples=[20]),
    faculty_id: int | None = Query(default=None, description="Filter by faculty id", examples=[1]),
    department_id: int | None = Query(default=None, description="Filter by department id", examples=[1]),
    course_id: int | None = Query(default=None, description="Filter by course id", examples=[1]),
    class_id: int | None = Query(default=None, description="Filter by class id", examples=[1]),
    faculty_scope=Depends(get_optional_faculty_scope_context),
):
    query = db.query(ClassCourseAssignment)
    if faculty_scope is not None:
        query = query.filter(ClassCourseAssignment.faculty_id == faculty_scope.faculty_id)
    if faculty_id is not None:
        query = query.filter(ClassCourseAssignment.faculty_id == faculty_id)
    if department_id is not None:
        query = query.filter(ClassCourseAssignment.department_id == department_id)
    if course_id is not None:
        query = query.filter(ClassCourseAssignment.course_id == course_id)
    if class_id is not None:
        query = query.filter(ClassCourseAssignment.class_id == class_id)
    total = query.count()
    items = query.order_by(ClassCourseAssignment.created_at.desc(), ClassCourseAssignment.id.desc()).offset(skip).limit(limit).all()
    return {"items": items, "total": total, "skip": skip, "limit": limit}


@router.get(
    "/class-course-assignments/{assignment_id}",
    response_model=ClassCourseAssignmentRead,
    dependencies=[Depends(require_roles("ACADEMIA", "FACULTY", "TEACHER"))],
)
def get_class_course_assignment(assignment_id: int, db: Session = Depends(get_role_scoped_db)):
    assignment = db.query(ClassCourseAssignment).filter(ClassCourseAssignment.id == assignment_id).first()
    if assignment is None:
        raise HTTPException(status_code=404, detail="Class course assignment not found")
    return assignment


@router.delete("/class-course-assignments/{assignment_id}", dependencies=[Depends(require_roles("ACADEMIA"))])
def delete_class_course_assignment(assignment_id: int, db: Session = Depends(get_role_scoped_db)):
    assignment = db.query(ClassCourseAssignment).filter(ClassCourseAssignment.id == assignment_id).first()
    if assignment is None:
        raise HTTPException(status_code=404, detail="Class course assignment not found")
    db.delete(assignment)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Cannot delete class course assignment") from exc
    return {"deleted": True, "assignment_id": assignment_id}
