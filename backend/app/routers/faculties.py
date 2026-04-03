from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_
from sqlalchemy.exc import IntegrityError

from app.core.security import require_roles
from app.db.models import (
    AttendanceRecord,
    AttendanceSession,
    ClassBatch,
    Course,
    CourseAssignment,
    CourseSchedule,
    Department,
    Enrollment,
    Faculty,
    Student,
    Teacher,
    User,
)
from app.db.session import get_db
from app.schemas.faculty import FacultyCreate, FacultyRead, FacultyUpdate, PaginatedFacultyRead


router = APIRouter(prefix="/faculties", tags=["faculties"])


def _bulk_delete_by_ids(db: Session, model, ids: list[int]) -> int:
    if not ids:
        return 0
    return (
        db.query(model)
        .filter(model.id.in_(ids))
        .delete(synchronize_session=False)
    )


def _faculty_delete_plan(db: Session, faculty_id: int) -> dict[str, int]:
    department_ids = [row[0] for row in db.query(Department.id).filter(Department.faculty_id == faculty_id).all()]
    class_batch_ids = [row[0] for row in db.query(ClassBatch.id).filter(ClassBatch.faculty_id == faculty_id).all()]
    student_ids = [row[0] for row in db.query(Student.id).filter(Student.faculty_id == faculty_id).all()]
    teacher_ids = [row[0] for row in db.query(Teacher.id).filter(Teacher.faculty_id == faculty_id).all()]
    course_ids = [row[0] for row in db.query(Course.id).filter(Course.faculty_id == faculty_id).all()]
    schedule_ids = [row[0] for row in db.query(CourseSchedule.id).filter(CourseSchedule.course_id.in_(course_ids)).all()]
    session_ids = [row[0] for row in db.query(AttendanceSession.id).filter(AttendanceSession.course_id.in_(course_ids)).all()]

    enrollment_ids = [row[0] for row in db.query(Enrollment.id).filter(Enrollment.course_id.in_(course_ids)).all()]
    assignment_ids = [row[0] for row in db.query(CourseAssignment.id).filter(CourseAssignment.course_id.in_(course_ids)).all()]
    record_ids = [row[0] for row in db.query(AttendanceRecord.id).filter(AttendanceRecord.course_id.in_(course_ids)).all()]
    user_ids = [row[0] for row in db.query(User.id).filter(User.faculty_id == faculty_id).all()]

    return {
        "attendance_records": len(record_ids),
        "attendance_sessions": len(session_ids),
        "course_schedules": len(schedule_ids),
        "enrollments": len(enrollment_ids),
        "course_assignments": len(assignment_ids),
        "courses": len(course_ids),
        "students": len(student_ids),
        "teachers": len(teacher_ids),
        "users": len(user_ids),
        "class_batches": len(class_batch_ids),
        "departments": len(department_ids),
        "faculties": 1,
    }


def _force_delete_faculty(db: Session, faculty_id: int) -> dict[str, int]:
    department_ids = [row[0] for row in db.query(Department.id).filter(Department.faculty_id == faculty_id).all()]
    class_batch_ids = [row[0] for row in db.query(ClassBatch.id).filter(ClassBatch.faculty_id == faculty_id).all()]
    student_ids = [row[0] for row in db.query(Student.id).filter(Student.faculty_id == faculty_id).all()]
    teacher_ids = [row[0] for row in db.query(Teacher.id).filter(Teacher.faculty_id == faculty_id).all()]
    course_ids = [row[0] for row in db.query(Course.id).filter(Course.faculty_id == faculty_id).all()]
    schedule_ids = [row[0] for row in db.query(CourseSchedule.id).filter(CourseSchedule.course_id.in_(course_ids)).all()]
    session_ids = [row[0] for row in db.query(AttendanceSession.id).filter(AttendanceSession.course_id.in_(course_ids)).all()]

    enrollment_ids = [row[0] for row in db.query(Enrollment.id).filter(Enrollment.course_id.in_(course_ids)).all()]
    assignment_ids = [row[0] for row in db.query(CourseAssignment.id).filter(CourseAssignment.course_id.in_(course_ids)).all()]
    record_ids = [row[0] for row in db.query(AttendanceRecord.id).filter(AttendanceRecord.course_id.in_(course_ids)).all()]
    user_ids = [row[0] for row in db.query(User.id).filter(User.faculty_id == faculty_id).all()]

    counts = {
        "attendance_records": _bulk_delete_by_ids(db, AttendanceRecord, record_ids),
        "attendance_sessions": _bulk_delete_by_ids(db, AttendanceSession, session_ids),
        "course_schedules": _bulk_delete_by_ids(db, CourseSchedule, schedule_ids),
        "enrollments": _bulk_delete_by_ids(db, Enrollment, enrollment_ids),
        "course_assignments": _bulk_delete_by_ids(db, CourseAssignment, assignment_ids),
        "courses": _bulk_delete_by_ids(db, Course, course_ids),
        "students": _bulk_delete_by_ids(db, Student, student_ids),
        "teachers": _bulk_delete_by_ids(db, Teacher, teacher_ids),
        "users": _bulk_delete_by_ids(db, User, user_ids),
        "class_batches": _bulk_delete_by_ids(db, ClassBatch, class_batch_ids),
        "departments": _bulk_delete_by_ids(db, Department, department_ids),
    }

    db.delete(db.query(Faculty).filter(Faculty.id == faculty_id).first())
    counts["faculties"] = 1
    return counts


@router.post("", response_model=FacultyRead, dependencies=[Depends(require_roles("ACADEMIA"))])
def create_faculty(payload: FacultyCreate, db: Session = Depends(get_db)):
    faculty = Faculty(name=payload.name, code=payload.code)
    db.add(faculty)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Faculty with same name/code already exists") from exc
    db.refresh(faculty)
    return faculty


@router.get("", response_model=PaginatedFacultyRead, dependencies=[Depends(require_roles("ACADEMIA", "FACULTY"))])
def list_faculties(
    db: Session = Depends(get_db),
    skip: int = Query(default=0, ge=0, description="Number of rows to skip", examples=[0]),
    limit: int = Query(default=50, ge=1, le=200, description="Page size", examples=[20]),
    search: str | None = Query(default=None, description="Search by faculty name or code", examples=["computer"]),
):
    query = db.query(Faculty)
    if search:
        pattern = f"%{search.strip()}%"
        query = query.filter(or_(Faculty.name.ilike(pattern), Faculty.code.ilike(pattern)))
    total = query.count()
    items = query.order_by(Faculty.name).offset(skip).limit(limit).all()
    return {"items": items, "total": total, "skip": skip, "limit": limit}


@router.put("/{faculty_id}", response_model=FacultyRead, dependencies=[Depends(require_roles("ACADEMIA"))])
def update_faculty(faculty_id: int, payload: FacultyUpdate, db: Session = Depends(get_db)):
    faculty = db.query(Faculty).filter(Faculty.id == faculty_id).first()
    if not faculty:
        raise HTTPException(status_code=404, detail="Faculty not found")

    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(faculty, field, value)

    db.add(faculty)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Update conflicts with existing faculty data") from exc
    db.refresh(faculty)
    return faculty


@router.delete("/{faculty_id}", dependencies=[Depends(require_roles("ACADEMIA"))])
def delete_faculty(
    faculty_id: int,
    force: bool = Query(default=False, description="Delete faculty and related records"),
    db: Session = Depends(get_db),
):
    faculty = db.query(Faculty).filter(Faculty.id == faculty_id).first()
    if not faculty:
        raise HTTPException(status_code=404, detail="Faculty not found")

    try:
        if force:
            db.expunge(faculty)
            counts = _force_delete_faculty(db, faculty_id)
            db.commit()
            return {
                "deleted": True,
                "faculty_id": faculty_id,
                "force": True,
                "counts": counts,
            }

        db.delete(faculty)
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Cannot delete faculty due to related records") from exc
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        if force:
            raise HTTPException(status_code=409, detail="Cannot force-delete faculty due to remaining related records") from exc
        raise HTTPException(status_code=500, detail="Faculty delete failed") from exc

    return {"deleted": True, "faculty_id": faculty_id, "force": False}


@router.get("/{faculty_id}/delete-preview", dependencies=[Depends(require_roles("ACADEMIA"))])
def preview_faculty_delete(faculty_id: int, db: Session = Depends(get_db)):
    faculty = db.query(Faculty).filter(Faculty.id == faculty_id).first()
    if not faculty:
        raise HTTPException(status_code=404, detail="Faculty not found")

    plan = _faculty_delete_plan(db, faculty_id)
    return {
        "faculty_id": faculty.id,
        "name": faculty.name,
        "code": faculty.code,
        "force_required": True if any(value > 0 for key, value in plan.items() if key != "faculties") else False,
        "counts": plan,
    }
