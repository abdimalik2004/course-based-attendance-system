from __future__ import annotations

import base64
from datetime import datetime, timedelta, time

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import (
    AttendanceRecord,
    AttendanceSession,
    AttendanceStatus,
    Base,
    ClassBatch,
    Course,
    CourseSchedule,
    Department,
    Enrollment,
    Faculty,
    SessionStatus,
    Student,
)
from app.services.attendance_service import AttendanceService
from app.services.schedule_service import ScheduleService
from app.utils.datetime_utils import schedule_weekday_from_datetime
from app.utils.weekday_utils import weekday_code


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    db = TestingSession()
    try:
        yield db
    finally:
        db.close()


def _seed_core(db):
    faculty = Faculty(name="FCS", code="FCS")
    db.add(faculty)
    db.flush()

    department = Department(faculty_id=faculty.id, name="Department of Information Technology", code="IT")
    db.add(department)
    db.flush()

    batch = ClassBatch(faculty_id=faculty.id, department_id=department.id, name="CIS2201", year=2026)
    db.add(batch)
    db.flush()

    course = Course(faculty_id=faculty.id, department_id=department.id, code="CSC401", title="AI")
    db.add(course)
    db.flush()

    schedule = CourseSchedule(
        course_id=course.id,
        weekday=weekday_code(schedule_weekday_from_datetime(datetime.now())),
        start_time=time(8, 0),
        end_time=time(10, 0),
        grace_period_minutes=10,
    )
    db.add(schedule)
    db.flush()

    student = Student(
        student_number="2201001",
        full_name="Student One",
        faculty_id=faculty.id,
        department_id=department.id,
        embedding_ref="2201001",
    )
    db.add(student)
    db.flush()

    db.add(Enrollment(student_id=student.id, course_id=course.id))
    db.commit()
    return faculty, batch, course, schedule, student


def _fake_jpeg_b64() -> str:
    # Decoding is monkeypatched in tests, so payload only needs to be base64-like.
    return base64.b64encode(b"fake-image").decode("ascii")


def test_prevent_duplicate_attendance(db_session, monkeypatch):
    _, _, course, schedule, student = _seed_core(db_session)

    now = datetime.now()
    session = AttendanceSession(
        course_id=course.id,
        schedule_id=schedule.id,
        session_date=now.date(),
        start_time=now - timedelta(minutes=1),
        end_time=now + timedelta(minutes=30),
        status=SessionStatus.ACTIVE,
    )
    db_session.add(session)
    db_session.commit()

    monkeypatch.setattr(
        "app.services.attendance_service.decode_base64_image",
        lambda _: np.zeros((8, 8, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "app.services.attendance_service.face_service.recognize_student",
        lambda _: {"matched": True, "student_number": student.student_number, "confidence": 0.92, "processing_time": 0.2},
    )

    svc = AttendanceService()
    first = svc.process_frame(db_session, session.id, _fake_jpeg_b64())
    second = svc.process_frame(db_session, session.id, _fake_jpeg_b64())

    assert first["ok"] is True
    assert second["ok"] is True
    assert second["message"] == "Attendance already marked"
    assert (
        db_session.query(AttendanceRecord)
        .filter(AttendanceRecord.session_id == session.id, AttendanceRecord.student_id == student.id)
        .count()
        == 1
    )


def test_present_and_late_cutoff_logic(db_session, monkeypatch):
    _, _, course, schedule, student = _seed_core(db_session)
    monkeypatch.setattr(
        "app.services.attendance_service.decode_base64_image",
        lambda _: np.zeros((8, 8, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "app.services.attendance_service.face_service.recognize_student",
        lambda _: {"matched": True, "student_number": student.student_number, "confidence": 0.95, "processing_time": 0.2},
    )

    now = datetime.now()
    present_session = AttendanceSession(
        course_id=course.id,
        schedule_id=schedule.id,
        session_date=now.date(),
        start_time=now - timedelta(minutes=2),
        end_time=now + timedelta(minutes=30),
        status=SessionStatus.ACTIVE,
    )
    db_session.add(present_session)
    db_session.commit()

    svc = AttendanceService()
    result_present = svc.process_frame(db_session, present_session.id, _fake_jpeg_b64())
    present_record = db_session.query(AttendanceRecord).filter(AttendanceRecord.id == result_present["record_id"]).first()
    assert present_record.status == AttendanceStatus.PRESENT

    student2 = Student(
        student_number="2201002",
        full_name="Student Two",
        faculty_id=student.faculty_id,
        department_id=student.department_id,
        embedding_ref="2201002",
    )
    db_session.add(student2)
    db_session.flush()
    db_session.add(Enrollment(student_id=student2.id, course_id=course.id))
    db_session.commit()

    monkeypatch.setattr(
        "app.services.attendance_service.face_service.recognize_student",
        lambda _: {"matched": True, "student_number": student2.student_number, "confidence": 0.89, "processing_time": 0.2},
    )

    late_session = AttendanceSession(
        course_id=course.id,
        schedule_id=schedule.id,
        session_date=now.date(),
        start_time=now - timedelta(minutes=20),
        end_time=now + timedelta(minutes=10),
        status=SessionStatus.ACTIVE,
    )
    db_session.add(late_session)
    db_session.commit()

    result_late = svc.process_frame(db_session, late_session.id, _fake_jpeg_b64())
    late_record = db_session.query(AttendanceRecord).filter(AttendanceRecord.id == result_late["record_id"]).first()
    assert late_record.status == AttendanceStatus.LATE


def test_close_session_marks_absent(db_session):
    _, _, course, schedule, student = _seed_core(db_session)

    student2 = Student(
        student_number="2201003",
        full_name="Student Three",
        faculty_id=student.faculty_id,
        department_id=student.department_id,
        embedding_ref="2201003",
    )
    db_session.add(student2)
    db_session.flush()
    db_session.add(Enrollment(student_id=student2.id, course_id=course.id))

    now = datetime.now()
    session = AttendanceSession(
        course_id=course.id,
        schedule_id=schedule.id,
        session_date=now.date(),
        start_time=now - timedelta(hours=2),
        end_time=now - timedelta(minutes=1),
        status=SessionStatus.ACTIVE,
    )
    db_session.add(session)
    db_session.flush()

    db_session.add(
        AttendanceRecord(
            student_id=student.id,
            course_id=course.id,
            session_id=session.id,
            status=AttendanceStatus.PRESENT,
            confidence=0.9,
        )
    )
    db_session.commit()

    svc = AttendanceService()
    created = svc.close_session_and_mark_absent(db_session, session)

    assert created == 1
    records = db_session.query(AttendanceRecord).filter(AttendanceRecord.session_id == session.id).all()
    statuses = {r.student_id: r.status for r in records}
    assert statuses[student.id] == AttendanceStatus.PRESENT
    assert statuses[student2.id] == AttendanceStatus.ABSENT
    assert session.status == SessionStatus.CLOSED


def test_scheduler_tick_idempotent_session_creation(db_session):
    _, _, course, schedule, _ = _seed_core(db_session)

    now = datetime.now()
    schedule.start_time = (now - timedelta(minutes=1)).time().replace(microsecond=0)
    schedule.end_time = (now + timedelta(minutes=15)).time().replace(microsecond=0)
    db_session.commit()

    svc = ScheduleService()
    svc._tick(db_session)
    svc._tick(db_session)

    sessions = db_session.query(AttendanceSession).filter(AttendanceSession.course_id == course.id).all()
    assert sessions == []
