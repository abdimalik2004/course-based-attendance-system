from __future__ import annotations

from datetime import datetime, timedelta, time

import numpy as np

from app.db.models import (
    AttendanceSession,
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
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.utils.datetime_utils import schedule_weekday_from_datetime
from app.utils.weekday_utils import weekday_code


def _build_db():
    engine = create_engine("sqlite+pysqlite:///:memory:", future=True)
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)
    return TestingSession()


def _seed(db):
    faculty = Faculty(name="FCS", code="FCS")
    db.add(faculty)
    db.flush()
    department = Department(faculty_id=faculty.id, name="Department of Information Technology", code="IT")
    db.add(department)
    db.flush()
    batch = ClassBatch(faculty_id=faculty.id, department_id=department.id, name="CIS2201")
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
    session = AttendanceSession(
        course_id=course.id,
        schedule_id=schedule.id,
        session_date=datetime.now().date(),
        start_time=datetime.now() - timedelta(minutes=1),
        end_time=datetime.now() + timedelta(minutes=30),
        status=SessionStatus.ACTIVE,
    )
    db.add(session)
    db.commit()
    return session.id


def test_timeout_response_when_processing_exceeds_budget(monkeypatch):
    db = _build_db()
    try:
        session_id = _seed(db)
        monkeypatch.setattr(
            "app.services.attendance_service.decode_base64_image",
            lambda _: np.zeros((8, 8, 3), dtype=np.uint8),
        )
        monkeypatch.setattr(
            "app.services.attendance_service.face_service.recognize_student",
            lambda _: {"matched": False, "reason": "timeout", "processing_time": 2.31},
        )

        svc = AttendanceService()
        result = svc.process_frame(db, session_id, "ZmFrZQ==")

        assert result["ok"] is False
        assert "timeout" in result["message"].lower()
        assert result["processing_time"] > 2.0
    finally:
        db.close()
