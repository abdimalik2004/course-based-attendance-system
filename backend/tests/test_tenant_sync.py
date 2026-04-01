from __future__ import annotations

from datetime import date, datetime, time, timezone

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import (
    AttendanceRecord,
    AttendanceSession,
    AttendanceStatus,
    Base,
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
    Role,
    UserRoleLink,
)
from app.db.sync_tenants import sync_faculty_tenants


def _build_sessionmaker():
    engine = create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)
    return TestingSession


def test_sync_faculty_tenants_copies_faculty_scoped_rows(monkeypatch):
    central_sessionmaker = _build_sessionmaker()
    tenant_sessionmaker = _build_sessionmaker()

    central_db = central_sessionmaker()
    faculty_a = Faculty(
        name="Faculty of Engineering",
        code="ENG",
        tenant_db_name="tenant_eng",
        tenant_db_provisioned_at=datetime.now(timezone.utc),
    )
    faculty_b = Faculty(
        name="Faculty of Medicine",
        code="MED",
        tenant_db_name="tenant_med",
        tenant_db_provisioned_at=datetime.now(timezone.utc),
    )
    central_db.add_all([faculty_a, faculty_b])
    central_db.flush()

    dept_a = Department(faculty_id=faculty_a.id, name="Architecture", code="ARCH")
    dept_b = Department(faculty_id=faculty_b.id, name="Surgery", code="SURG")
    central_db.add_all([dept_a, dept_b])
    central_db.flush()

    batch_a = ClassBatch(faculty_id=faculty_a.id, department_id=dept_a.id, name="ENG2201", year=2026)
    batch_b = ClassBatch(faculty_id=faculty_b.id, department_id=dept_b.id, name="MED2201", year=2026)
    central_db.add_all([batch_a, batch_b])
    central_db.flush()

    teacher_user = User(username="eng_teacher", hashed_password="x", is_active=True, faculty_id=faculty_a.id)
    central_db.add(teacher_user)
    central_db.flush()

    teacher_a = Teacher(
        teacher_number="ENGT001",
        full_name="Teacher A",
        faculty_id=faculty_a.id,
        department_id=dept_a.id,
        user_id=teacher_user.id,
    )
    teacher_b = Teacher(
        teacher_number="MEDT001",
        full_name="Teacher B",
        faculty_id=faculty_b.id,
        department_id=dept_b.id,
        user_id=None,
    )
    central_db.add_all([teacher_a, teacher_b])
    central_db.flush()

    student_a = Student(
        student_number="26ENG001",
        full_name="Student A",
        faculty_id=faculty_a.id,
        department_id=dept_a.id,
        class_batch_id=batch_a.id,
        embedding_ref="26ENG001",
    )
    student_b = Student(
        student_number="26MED001",
        full_name="Student B",
        faculty_id=faculty_b.id,
        department_id=dept_b.id,
        class_batch_id=batch_b.id,
        embedding_ref="26MED001",
    )
    central_db.add_all([student_a, student_b])
    central_db.flush()

    course_a = Course(class_batch_id=batch_a.id, code="ENG001", title="Statics")
    course_b = Course(class_batch_id=batch_b.id, code="MED001", title="Anatomy")
    central_db.add_all([course_a, course_b])
    central_db.flush()

    central_db.add(CourseAssignment(course_id=course_a.id, teacher_id=teacher_a.id, is_primary=True))
    central_db.add(Enrollment(student_id=student_a.id, course_id=course_a.id))
    schedule = CourseSchedule(
        course_id=course_a.id,
        weekday="mon",
        start_time=time(8, 0),
        end_time=time(10, 0),
        grace_period_minutes=10,
    )
    central_db.add(schedule)
    central_db.flush()

    session = AttendanceSession(
        course_id=course_a.id,
        schedule_id=schedule.id,
        session_date=date(2026, 3, 15),
        start_time=datetime(2026, 3, 15, 8, 0, 0),
        end_time=datetime(2026, 3, 15, 10, 0, 0),
        status="ACTIVE",
    )
    central_db.add(session)
    central_db.flush()

    central_db.add(
        AttendanceRecord(
            student_id=student_a.id,
            course_id=course_a.id,
            session_id=session.id,
            status=AttendanceStatus.PRESENT,
            confidence=0.95,
        )
    )
    central_db.commit()
    central_db.close()

    monkeypatch.setattr("app.db.sync_tenants.SessionLocal", central_sessionmaker)
    monkeypatch.setattr(
        "app.db.sync_tenants.get_tenant_sessionmaker",
        lambda tenant_db_name: tenant_sessionmaker,
    )

    summary = sync_faculty_tenants(
        faculty_code="ENG",
        include_operational_tables=True,
        allow_legacy_operational_sync=True,
    )

    assert summary["processed"] == 1
    assert summary["synced"] == 1
    assert summary["failed"] == 0
    assert summary["rows"] > 0

    tenant_db = tenant_sessionmaker()
    try:
        assert tenant_db.query(Faculty).count() == 1
        assert tenant_db.query(Faculty).first().code == "ENG"
        assert tenant_db.query(Department).count() == 1
        assert tenant_db.query(ClassBatch).count() == 1
        assert tenant_db.query(User).count() == 1
        assert tenant_db.query(Teacher).count() == 1
        assert tenant_db.query(Student).count() == 1
        assert tenant_db.query(Course).count() == 1
        assert tenant_db.query(CourseAssignment).count() == 1
        assert tenant_db.query(Enrollment).count() == 1
        assert tenant_db.query(CourseSchedule).count() == 1
        assert tenant_db.query(AttendanceSession).count() == 1
        assert tenant_db.query(AttendanceRecord).count() == 1
        assert tenant_db.query(Student).first().student_number == "26ENG001"
    finally:
        tenant_db.close()


def test_sync_faculty_tenants_metadata_only_skips_operational_rows(monkeypatch):
    central_sessionmaker = _build_sessionmaker()
    tenant_sessionmaker = _build_sessionmaker()

    central_db = central_sessionmaker()
    faculty = Faculty(
        name="Faculty of Engineering",
        code="ENG",
        tenant_db_name="tenant_eng",
        tenant_db_provisioned_at=datetime.now(timezone.utc),
    )
    central_db.add(faculty)
    central_db.flush()
    central_db.add(Department(faculty_id=faculty.id, name="Architecture", code="ARCH"))
    central_db.commit()
    central_db.close()

    monkeypatch.setattr("app.db.sync_tenants.SessionLocal", central_sessionmaker)
    monkeypatch.setattr(
        "app.db.sync_tenants.get_tenant_sessionmaker",
        lambda tenant_db_name: tenant_sessionmaker,
    )

    summary = sync_faculty_tenants(faculty_code="ENG")

    assert summary["mode"] == "metadata-only"
    assert summary["processed"] == 1
    assert summary["synced"] == 1

    tenant_db = tenant_sessionmaker()
    try:
        assert tenant_db.query(Faculty).count() == 1
        assert tenant_db.query(Department).count() == 0
    finally:
        tenant_db.close()


def test_sync_faculty_tenants_skips_unprovisioned_faculty(monkeypatch):
    central_sessionmaker = _build_sessionmaker()
    central_db = central_sessionmaker()
    central_db.add(Faculty(name="Faculty of Science", code="SCI", tenant_db_name="tenant_sci", tenant_db_provisioned_at=None))
    central_db.commit()
    central_db.close()

    monkeypatch.setattr("app.db.sync_tenants.SessionLocal", central_sessionmaker)

    summary = sync_faculty_tenants(faculty_code="SCI")

    assert summary["processed"] == 1
    assert summary["synced"] == 0
    assert summary["skipped"] == 1
    assert summary["failed"] == 0


def test_sync_faculty_tenants_syncs_user_role_links_in_metadata_mode(monkeypatch):
    central_sessionmaker = _build_sessionmaker()
    tenant_sessionmaker = _build_sessionmaker()

    central_db = central_sessionmaker()
    faculty = Faculty(
        name="Faculty of Business",
        code="BUS",
        tenant_db_name="tenant_bus",
        tenant_db_provisioned_at=datetime.now(timezone.utc),
    )
    central_db.add(faculty)
    central_db.flush()

    role = Role(name="FACULTY_ADMIN")
    user = User(
        username="busadmin",
        hashed_password="x",
        is_active=True,
        faculty_id=faculty.id,
    )
    central_db.add_all([role, user])
    central_db.flush()
    central_db.add(UserRoleLink(user_id=user.id, role_id=role.id))
    central_db.commit()
    central_db.close()

    monkeypatch.setattr("app.db.sync_tenants.SessionLocal", central_sessionmaker)
    monkeypatch.setattr(
        "app.db.sync_tenants.get_tenant_sessionmaker",
        lambda tenant_db_name: tenant_sessionmaker,
    )

    summary = sync_faculty_tenants(faculty_code="BUS")

    assert summary["mode"] == "metadata-only"
    assert summary["synced"] == 1
    assert summary["failed"] == 0

    tenant_db = tenant_sessionmaker()
    try:
        assert tenant_db.query(User).count() == 1
        assert tenant_db.query(Role).count() == 1
        assert tenant_db.query(UserRoleLink).count() == 1
    finally:
        tenant_db.close()


def test_sync_faculty_tenants_blocks_legacy_operational_sync_by_default(monkeypatch):
    central_sessionmaker = _build_sessionmaker()
    central_db = central_sessionmaker()
    central_db.add(
        Faculty(
            name="Faculty of Engineering",
            code="ENG",
            tenant_db_name="tenant_eng",
            tenant_db_provisioned_at=datetime.now(timezone.utc),
        )
    )
    central_db.commit()
    central_db.close()

    monkeypatch.setattr("app.db.sync_tenants.SessionLocal", central_sessionmaker)

    try:
        sync_faculty_tenants(faculty_code="ENG", include_operational_tables=True)
        assert False, "Expected legacy operational sync to be blocked"
    except ValueError as exc:
        assert "TENANT_DB_ALLOW_LEGACY_OPERATIONAL_SYNC=true" in str(exc)
