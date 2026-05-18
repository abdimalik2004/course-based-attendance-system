from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import or_, func
from sqlalchemy.exc import IntegrityError

from app.core.security import require_roles
from app.db.faculty_scope import enforce_faculty_scope, get_optional_faculty_scope_context
from app.db.models import Student, StudentAdmissionStatus
from app.db.role_scoped import get_role_scoped_db
from app.services.enrollment_service import auto_enroll_student_in_matching_courses
from app.schemas.student import (
    StudentCreate,
    StudentDashboardStatsRead,
    StudentCapturedImagesRead,
    StudentRead,
    StudentStatus,
    StudentUpdate,
    PaginatedStudentRead,
)
from app.utils.organization import (
    ensure_department_belongs_to_faculty,
    get_department_or_404,
    get_faculty_or_404,
)
from app.utils.student_numbering import next_available_student_number
from app.utils.student_numbering import student_dataset_dir as resolve_student_dataset_dir


router = APIRouter(prefix="/students", tags=["students"])


NEW_ADMISSIONS_WINDOW_DAYS = 30
_DATASET_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _count_students_by_status(db: Session, status: StudentAdmissionStatus) -> int:
    return int(db.query(func.count(Student.id)).filter(Student.status == status).scalar() or 0)


def _student_or_404(db: Session, student_id: int) -> Student:
    student = db.query(Student).filter(Student.id == student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    return student


def _student_dataset_dir(student: Student) -> Path | None:
    candidate_numbers: list[str] = []
    if student.embedding_ref:
        candidate_numbers.append(student.embedding_ref)
    candidate_numbers.append(student.student_number)

    for value in candidate_numbers:
        candidate = resolve_student_dataset_dir(value)
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _student_dataset_images(student: Student) -> list[Path]:
    dataset_dir = _student_dataset_dir(student)
    if dataset_dir is None:
        return []

    return sorted(
        [
            file_path
            for file_path in dataset_dir.iterdir()
            if file_path.is_file() and file_path.suffix.lower() in _DATASET_IMAGE_SUFFIXES
        ],
        key=lambda path: path.name.lower(),
    )


@router.get(
    "/stats",
    response_model=StudentDashboardStatsRead,
    dependencies=[Depends(require_roles("ADMISSIONS", "FACULTY", "TEACHER", "ACADEMIA", "HR"))],
)
def student_dashboard_stats(db: Session = Depends(get_role_scoped_db)):
    now_utc = datetime.now(timezone.utc)
    recent_cutoff = now_utc - timedelta(days=NEW_ADMISSIONS_WINDOW_DAYS)

    total_students = int(db.query(func.count(Student.id)).scalar() or 0)
    new_admissions = int(
        db.query(func.count(Student.id)).filter(Student.created_at >= recent_cutoff).scalar() or 0
    )
    pending_admissions = _count_students_by_status(db, StudentAdmissionStatus.PENDING)
    rejected_applications = _count_students_by_status(db, StudentAdmissionStatus.REJECTED)

    return {
        "total_students": total_students,
        "new_admissions": new_admissions,
        "pending_admissions": pending_admissions,
        "rejected_applications": rejected_applications,
    }


@router.get(
    "/{student_id}/captured-images",
    response_model=StudentCapturedImagesRead,
    dependencies=[Depends(require_roles("ADMISSIONS", "FACULTY", "TEACHER", "ACADEMIA", "HR"))],
)
def list_student_captured_images(
    student_id: int,
    db: Session = Depends(get_role_scoped_db),
):
    student = _student_or_404(db, student_id)
    image_paths = _student_dataset_images(student)
    images = [
        {
            "file_name": image_path.name,
            "url": f"/students/{student.id}/captured-images/{image_path.name}",
        }
        for image_path in image_paths
    ]

    return {
        "student_id": student.id,
        "student_number": student.student_number,
        "image_count": len(images),
        "images": images,
    }


@router.get(
    "/{student_id}/captured-images/{file_name}",
    dependencies=[Depends(require_roles("ADMISSIONS", "FACULTY", "TEACHER", "ACADEMIA", "HR"))],
)
def read_student_captured_image(
    student_id: int,
    file_name: str,
    db: Session = Depends(get_role_scoped_db),
):
    student = _student_or_404(db, student_id)
    safe_file_name = Path(file_name).name
    if safe_file_name != file_name:
        raise HTTPException(status_code=400, detail="Invalid file name")

    dataset_dir = _student_dataset_dir(student)
    if dataset_dir is None:
        raise HTTPException(status_code=404, detail="No dataset images found for this student")

    image_path = dataset_dir / safe_file_name
    if not image_path.exists() or not image_path.is_file() or image_path.suffix.lower() not in _DATASET_IMAGE_SUFFIXES:
        raise HTTPException(status_code=404, detail="Captured image not found")

    return FileResponse(image_path)


@router.post("", response_model=StudentRead, dependencies=[Depends(require_roles("ADMISSIONS"))])
def create_student(
    payload: StudentCreate,
    db: Session = Depends(get_role_scoped_db),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    if faculty_scope is None:
        faculty = get_faculty_or_404(db, payload.faculty_id)
        faculty_code = faculty.code
    else:
        enforce_faculty_scope(payload.faculty_id, faculty_scope)
        faculty_code = faculty_scope.faculty_code

    department = get_department_or_404(db, payload.department_id)
    ensure_department_belongs_to_faculty(department, payload.faculty_id)
    student_number = next_available_student_number(db, faculty_code, date.today().year)
    embedding_ref = student_number

    obj = Student(
        student_number=student_number,
        full_name=payload.full_name,
        faculty_id=payload.faculty_id,
        department_id=payload.department_id,
        embedding_ref=embedding_ref,
    )
    db.add(obj)
    try:
        db.flush()
        auto_enroll_student_in_matching_courses(db, obj)
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Student number already exists") from exc
    db.refresh(obj)
    return obj


@router.get("", response_model=PaginatedStudentRead, dependencies=[Depends(require_roles("ADMISSIONS", "FACULTY", "TEACHER", "ACADEMIA"))])
def list_students(
    db: Session = Depends(get_role_scoped_db),
    skip: int = Query(default=0, ge=0, description="Number of rows to skip", examples=[0]),
    limit: int = Query(default=50, ge=1, le=200, description="Page size", examples=[20]),
    faculty_id: int | None = Query(default=None, description="Filter by faculty id", examples=[1]),
    department_id: int | None = Query(default=None, description="Filter by department id", examples=[1]),
    status: StudentStatus | None = Query(default=None, description="Filter by admission status", examples=["pending"]),
    search: str | None = Query(default=None, description="Search by student number or full name", examples=["2201"]),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    query = db.query(Student)
    if faculty_scope is not None:
        if faculty_id is not None:
            enforce_faculty_scope(faculty_id, faculty_scope)
        else:
            faculty_id = faculty_scope.faculty_id
    if faculty_id is not None:
        query = query.filter(Student.faculty_id == faculty_id)
    if department_id is not None:
        query = query.filter(Student.department_id == department_id)
    if status is not None:
        query = query.filter(Student.status == StudentAdmissionStatus(status.value))
    if search:
        pattern = f"%{search.strip()}%"
        query = query.filter(or_(Student.full_name.ilike(pattern), Student.student_number.ilike(pattern)))
    total = query.count()
    items = query.order_by(Student.created_at.desc(), Student.id.desc()).offset(skip).limit(limit).all()
    return {"items": items, "total": total, "skip": skip, "limit": limit}


@router.put("/{student_id}", response_model=StudentRead, dependencies=[Depends(require_roles("ADMISSIONS"))])
def update_student(
    student_id: int,
    payload: StudentUpdate,
    db: Session = Depends(get_role_scoped_db),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    obj = db.query(Student).filter(Student.id == student_id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Student not found")

    target_faculty_id = payload.faculty_id if payload.faculty_id is not None else obj.faculty_id
    target_department_id = payload.department_id if payload.department_id is not None else obj.department_id

    if faculty_scope is None:
        get_faculty_or_404(db, target_faculty_id)
    else:
        enforce_faculty_scope(target_faculty_id, faculty_scope)

    department = get_department_or_404(db, target_department_id)
    ensure_department_belongs_to_faculty(department, target_faculty_id)

    payload_data = payload.model_dump(exclude_unset=True)
    for field in ("faculty_id", "department_id"):
        payload_data.pop(field, None)

    for field, value in payload_data.items():
        setattr(obj, field, value)

    obj.faculty_id = target_faculty_id
    obj.department_id = target_department_id

    db.add(obj)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Update conflicts with existing student data") from exc
    db.refresh(obj)
    return obj


@router.delete("/{student_id}", dependencies=[Depends(require_roles("ADMISSIONS"))])
def delete_student(student_id: int, db: Session = Depends(get_role_scoped_db)):
    obj = db.query(Student).filter(Student.id == student_id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Student not found")
    db.delete(obj)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Cannot delete student due to related records") from exc
    return {"deleted": True, "student_id": student_id}
