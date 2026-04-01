from __future__ import annotations

import argparse
from dataclasses import dataclass
import warnings

from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.data_ownership import CENTRAL_PLATFORM_TABLE_KEYS, TENANT_OPERATIONAL_TABLE_KEYS
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
    Role,
    Student,
    Teacher,
    User,
    UserRoleLink,
)
from app.db.session import SessionLocal, get_tenant_sessionmaker


METADATA_TABLE_KEYS = CENTRAL_PLATFORM_TABLE_KEYS

OPERATIONAL_TABLE_KEYS = TENANT_OPERATIONAL_TABLE_KEYS


@dataclass(frozen=True)
class FacultySyncResult:
    faculty_id: int
    faculty_code: str
    tenant_db_name: str
    synced: bool
    row_count: int = 0
    skipped: bool = False
    reason: str | None = None


def _clone_row(model_cls, row):
    values = {column.name: getattr(row, column.name) for column in model_cls.__table__.columns}
    return model_cls(**values)


def _build_faculty_payload(
    central_db: Session,
    faculty: Faculty,
    *,
    include_operational_tables: bool,
) -> dict[str, list]:
    payload: dict[str, list] = {
        "faculties": [_clone_row(Faculty, faculty)],
        "roles": [],
        "users": [],
        "user_role_links": [],
        "departments": [],
        "class_batches": [],
        "teachers": [],
        "students": [],
        "courses": [],
        "course_assignments": [],
        "course_schedules": [],
        "enrollments": [],
        "attendance_sessions": [],
        "attendance_records": [],
    }

    faculty_users = (
        central_db.query(User)
        .filter(User.faculty_id == faculty.id)
        .order_by(User.id)
        .all()
    )
    role_ids = sorted({role.id for user in faculty_users for role in user.roles})
    roles = (
        central_db.query(Role)
        .filter(Role.id.in_(role_ids))
        .order_by(Role.id)
        .all()
        if role_ids
        else []
    )
    user_role_links = (
        central_db.query(UserRoleLink)
        .filter(UserRoleLink.user_id.in_([row.id for row in faculty_users]))
        .order_by(UserRoleLink.user_id, UserRoleLink.role_id)
        .all()
        if faculty_users
        else []
    )

    payload["roles"] = [_clone_row(Role, row) for row in roles]
    payload["users"] = [_clone_row(User, row) for row in faculty_users]
    payload["user_role_links"] = [_clone_row(UserRoleLink, row) for row in user_role_links]

    if not include_operational_tables:
        return payload

    departments = (
        central_db.query(Department)
        .filter(Department.faculty_id == faculty.id)
        .order_by(Department.id)
        .all()
    )
    class_batches = (
        central_db.query(ClassBatch)
        .filter(ClassBatch.faculty_id == faculty.id)
        .order_by(ClassBatch.id)
        .all()
    )
    students = (
        central_db.query(Student)
        .filter(Student.faculty_id == faculty.id)
        .order_by(Student.id)
        .all()
    )
    teachers = (
        central_db.query(Teacher)
        .filter(Teacher.faculty_id == faculty.id)
        .order_by(Teacher.id)
        .all()
    )

    class_batch_ids = [row.id for row in class_batches]
    student_ids = [row.id for row in students]
    teacher_ids = [row.id for row in teachers]

    courses = (
        central_db.query(Course)
        .filter(Course.class_batch_id.in_(class_batch_ids))
        .order_by(Course.id)
        .all()
        if class_batch_ids
        else []
    )
    course_ids = [row.id for row in courses]

    course_assignments = (
        central_db.query(CourseAssignment)
        .filter(CourseAssignment.course_id.in_(course_ids), CourseAssignment.teacher_id.in_(teacher_ids))
        .order_by(CourseAssignment.id)
        .all()
        if course_ids and teacher_ids
        else []
    )
    course_schedules = (
        central_db.query(CourseSchedule)
        .filter(CourseSchedule.course_id.in_(course_ids))
        .order_by(CourseSchedule.id)
        .all()
        if course_ids
        else []
    )
    enrollments = (
        central_db.query(Enrollment)
        .filter(Enrollment.course_id.in_(course_ids), Enrollment.student_id.in_(student_ids))
        .order_by(Enrollment.id)
        .all()
        if course_ids and student_ids
        else []
    )
    attendance_sessions = (
        central_db.query(AttendanceSession)
        .filter(AttendanceSession.course_id.in_(course_ids))
        .order_by(AttendanceSession.id)
        .all()
        if course_ids
        else []
    )
    session_ids = [row.id for row in attendance_sessions]
    attendance_records = (
        central_db.query(AttendanceRecord)
        .filter(
            AttendanceRecord.course_id.in_(course_ids),
            AttendanceRecord.student_id.in_(student_ids),
            AttendanceRecord.session_id.in_(session_ids),
        )
        .order_by(AttendanceRecord.id)
        .all()
        if course_ids and student_ids and session_ids
        else []
    )

    payload["departments"] = [_clone_row(Department, row) for row in departments]
    payload["class_batches"] = [_clone_row(ClassBatch, row) for row in class_batches]
    payload["teachers"] = [_clone_row(Teacher, row) for row in teachers]
    payload["students"] = [_clone_row(Student, row) for row in students]
    payload["courses"] = [_clone_row(Course, row) for row in courses]
    payload["course_assignments"] = [_clone_row(CourseAssignment, row) for row in course_assignments]
    payload["course_schedules"] = [_clone_row(CourseSchedule, row) for row in course_schedules]
    payload["enrollments"] = [_clone_row(Enrollment, row) for row in enrollments]
    payload["attendance_sessions"] = [_clone_row(AttendanceSession, row) for row in attendance_sessions]
    payload["attendance_records"] = [_clone_row(AttendanceRecord, row) for row in attendance_records]
    return payload


def _replace_tenant_data(
    tenant_db: Session,
    payload: dict[str, list],
    *,
    include_operational_tables: bool,
) -> int:
    delete_order = [UserRoleLink, User, Role, Faculty]
    if include_operational_tables:
        delete_order = [
            AttendanceRecord,
            AttendanceSession,
            Enrollment,
            CourseSchedule,
            CourseAssignment,
            Course,
            Student,
            Teacher,
            ClassBatch,
            Department,
            UserRoleLink,
            User,
            Role,
            Faculty,
        ]

    for model_cls in delete_order:
        tenant_db.query(model_cls).delete()

    insert_order = list(METADATA_TABLE_KEYS)
    if include_operational_tables:
        insert_order.extend(OPERATIONAL_TABLE_KEYS)

    for key in insert_order:
        rows = payload[key]
        if rows:
            tenant_db.add_all(rows)
            # Flush per table batch so foreign-key dependent rows (for example
            # user_role_links) always see parent rows inserted first.
            tenant_db.flush()

    tenant_db.commit()
    return sum(len(rows) for rows in payload.values())


def sync_faculty_tenants(
    *,
    faculty_id: int | None = None,
    faculty_code: str | None = None,
    include_unprovisioned: bool = False,
    include_operational_tables: bool = False,
    allow_legacy_operational_sync: bool | None = None,
) -> dict[str, int]:
    legacy_operational_sync_allowed = (
        settings.tenant_db_allow_legacy_operational_sync
        if allow_legacy_operational_sync is None
        else allow_legacy_operational_sync
    )
    if include_operational_tables and not legacy_operational_sync_allowed:
        raise ValueError(
            "Legacy operational tenant sync is disabled by default. "
            "Set TENANT_DB_ALLOW_LEGACY_OPERATIONAL_SYNC=true for emergency backfill only."
        )

    summary = {
        "mode": "full" if include_operational_tables else "metadata-only",
        "processed": 0,
        "synced": 0,
        "skipped": 0,
        "failed": 0,
        "rows": 0,
        "errors": [],
    }

    central_db = SessionLocal()
    try:
        query = central_db.query(Faculty)
        if faculty_id is not None:
            query = query.filter(Faculty.id == faculty_id)
        if faculty_code is not None:
            query = query.filter(Faculty.code == faculty_code.strip().upper())
        faculties = query.order_by(Faculty.id).all()

        for faculty in faculties:
            if not faculty.tenant_db_name:
                summary["processed"] += 1
                summary["skipped"] += 1
                continue
            if not include_unprovisioned and faculty.tenant_db_provisioned_at is None:
                summary["processed"] += 1
                summary["skipped"] += 1
                continue

            summary["processed"] += 1
            payload = _build_faculty_payload(
                central_db,
                faculty,
                include_operational_tables=include_operational_tables,
            )

            tenant_db = get_tenant_sessionmaker(faculty.tenant_db_name)()
            try:
                summary["rows"] += _replace_tenant_data(
                    tenant_db,
                    payload,
                    include_operational_tables=include_operational_tables,
                )
                summary["synced"] += 1
            except Exception as exc:  # noqa: BLE001
                tenant_db.rollback()
                summary["failed"] += 1
                summary["errors"].append(
                    {
                        "faculty_id": faculty.id,
                        "faculty_code": faculty.code,
                        "tenant_db_name": faculty.tenant_db_name,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            finally:
                tenant_db.close()

        return summary
    finally:
        central_db.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync faculty-scoped data from central DB into tenant DBs")
    parser.add_argument("--faculty-id", type=int, default=None, help="Sync only one faculty id")
    parser.add_argument("--faculty-code", default=None, help="Sync only one faculty code")
    parser.add_argument(
        "--include-unprovisioned",
        action="store_true",
        help="Include faculties that are not marked as provisioned",
    )
    parser.add_argument(
        "--include-operational-tables",
        action="store_true",
        help=(
            "Legacy behavior: also replace operational tenant tables from central DB. "
            "Deprecated for tenant-first mode."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.include_operational_tables:
        warnings.warn(
            "--include-operational-tables is deprecated and should be used only for emergency backfill.",
            DeprecationWarning,
            stacklevel=2,
        )
    try:
        summary = sync_faculty_tenants(
            faculty_id=args.faculty_id,
            faculty_code=args.faculty_code,
            include_unprovisioned=args.include_unprovisioned,
            include_operational_tables=args.include_operational_tables,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        "Tenant sync summary:",
        f"mode={summary['mode']}",
        f"processed={summary['processed']}",
        f"synced={summary['synced']}",
        f"skipped={summary['skipped']}",
        f"failed={summary['failed']}",
        f"rows={summary['rows']}",
    )
    if summary["errors"]:
        for item in summary["errors"]:
            print(
                "Tenant sync error:",
                f"faculty_id={item['faculty_id']}",
                f"faculty_code={item['faculty_code']}",
                f"tenant_db_name={item['tenant_db_name']}",
                f"error_type={item['error_type']}",
                f"error={item['error']}",
            )


if __name__ == "__main__":
    main()
