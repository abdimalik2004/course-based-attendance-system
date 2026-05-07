from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from app.db.models import (
    AttendanceRecord,
    AttendanceSession,
    AttendanceStatus,
    Course,
    CourseSchedule,
    Enrollment,
    SessionStatus,
    Student,
    User,
)
from app.services.face_service import face_service
from app.utils.datetime_utils import current_local_datetime
from app.utils.image_decode import decode_base64_image


class AttendanceService:
    def process_frame(self, db: Session, session_id: int, image_b64: str) -> dict:
        if session_id <= 0:
            return {"ok": False, "message": "Invalid session id"}

        session = db.query(AttendanceSession).filter(AttendanceSession.id == session_id).first()
        if not session:
            return {"ok": False, "message": "Session not found"}
        if session.course_id <= 0:
            return {"ok": False, "message": "Session has invalid course reference"}
        if session.status != SessionStatus.ACTIVE:
            return {"ok": False, "message": "Session is not active"}

        now = current_local_datetime()

        frame = decode_base64_image(image_b64)
        recognition = face_service.recognize_student(frame)
        if not recognition.get("matched"):
            if recognition.get("reason") == "timeout":
                return {
                    "ok": False,
                    "message": "Recognition timeout exceeded 2.0 seconds",
                    "processing_time": recognition.get("processing_time"),
                }
            return {
                "ok": True,
                "message": "No valid face matched",
                "processing_time": recognition.get("processing_time"),
            }

        student = (
            db.query(Student)
            .filter(Student.student_number == recognition["student_number"])
            .first()
        )
        if not student:
            return {"ok": False, "message": "Recognized student not registered"}
        if (
            student.id <= 0
            or student.faculty_id <= 0
            or student.department_id <= 0
        ):
            return {"ok": False, "message": "Recognized student has invalid references"}

        if student.faculty_id != session.course.faculty_id:
            return {"ok": False, "message": "Student does not belong to course faculty"}

        enrolled = (
            db.query(Enrollment)
            .filter(
                Enrollment.student_id == student.id,
                Enrollment.course_id == session.course_id,
                Enrollment.student_id > 0,
                Enrollment.course_id > 0,
            )
            .first()
        )
        if not enrolled:
            return {"ok": False, "message": "Student not enrolled in this course"}

        existing = (
            db.query(AttendanceRecord)
            .filter(
                AttendanceRecord.student_id == student.id,
                AttendanceRecord.course_id == session.course_id,
                AttendanceRecord.session_id == session.id,
                AttendanceRecord.student_id > 0,
                AttendanceRecord.course_id > 0,
                AttendanceRecord.session_id > 0,
            )
            .first()
        )
        if existing:
            return {"ok": True, "message": "Attendance already marked", "record_id": existing.id}

        grace_cutoff = session.start_time + timedelta(minutes=session.schedule.grace_period_minutes)
        status = AttendanceStatus.PRESENT if now <= grace_cutoff else AttendanceStatus.LATE

        record = AttendanceRecord(
            student_id=student.id,
            course_id=session.course_id,
            session_id=session.id,
            status=status,
            confidence=recognition.get("confidence", 0.0),
        )
        db.add(record)
        db.commit()
        db.refresh(record)

        return {
            "ok": True,
            "message": "Attendance recorded",
            "record_id": record.id,
            "status": record.status,
            "student_number": student.student_number,
            "confidence": record.confidence,
            "processing_time": recognition.get("processing_time"),
        }

    def start_session(self, db: Session, course_id: int, schedule_id: int | None, instructor_id: int) -> dict:
        if course_id <= 0 or instructor_id <= 0:
            return {"ok": False, "message": "Invalid session start payload"}

        course = db.query(Course).filter(Course.id == course_id).with_for_update().first()
        if not course:
            return {"ok": False, "message": "Course not found"}

        schedule_query = db.query(CourseSchedule).filter(CourseSchedule.course_id == course_id).with_for_update()
        if schedule_id is not None:
            schedule_query = schedule_query.filter(CourseSchedule.id == schedule_id)
        schedule = schedule_query.order_by(CourseSchedule.id.asc()).first()
        if not schedule:
            return {"ok": False, "message": "Schedule not found for course"}

        instructor = db.query(User).filter(User.id == instructor_id, User.is_active.is_(True)).first()
        if not instructor:
            return {"ok": False, "message": "Instructor not found"}

        existing_active_session = (
            db.query(AttendanceSession)
            .filter(
                AttendanceSession.course_id == course_id,
                AttendanceSession.status == SessionStatus.ACTIVE,
            )
            .first()
        )
        if existing_active_session:
            return {
                "ok": True,
                "message": "Attendance session already active",
                "session": existing_active_session,
                "created": False,
            }

        now = current_local_datetime()
        session = AttendanceSession(
            course_id=course_id,
            instructor_id=instructor_id,
            schedule_id=schedule_id,
            session_date=now.date(),
            start_time=now,
            end_time=None,
            status=SessionStatus.ACTIVE,
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        return {"ok": True, "message": "Attendance session started", "session": session, "created": True}

    def end_session(self, db: Session, session_id: int) -> dict:
        if session_id <= 0:
            return {"ok": False, "message": "Invalid session id"}

        session = db.query(AttendanceSession).filter(AttendanceSession.id == session_id).first()
        if not session:
            return {"ok": False, "message": "Session not found"}

        if session.status == SessionStatus.CLOSED:
            return {"ok": True, "message": "Attendance session already closed", "session": session, "absences": 0, "ended": False}

        if session.end_time is None:
            session.end_time = current_local_datetime()

        absences = self.close_session_and_mark_absent(db, session)
        db.refresh(session)
        return {"ok": True, "message": "Attendance session closed", "session": session, "absences": absences, "ended": True}

    def close_session_and_mark_absent(self, db: Session, session: AttendanceSession) -> int:
        if session.status == SessionStatus.CLOSED:
            return 0

        if session.end_time is None:
            session.end_time = current_local_datetime()

        enrolled_students = (
            db.query(Student)
            .join(Enrollment, Enrollment.student_id == Student.id)
            .filter(
                Enrollment.course_id == session.course_id,
                Student.id > 0,
                Student.faculty_id > 0,
                Enrollment.student_id > 0,
                Enrollment.course_id > 0,
            )
            .all()
        )

        present_ids = {
            record.student_id
            for record in db.query(AttendanceRecord)
            .filter(AttendanceRecord.session_id == session.id, AttendanceRecord.student_id > 0)
            .all()
        }

        created = 0
        for student in enrolled_students:
            if student.id in present_ids:
                continue
            db.add(
                AttendanceRecord(
                    student_id=student.id,
                    course_id=session.course_id,
                    session_id=session.id,
                    status=AttendanceStatus.ABSENT,
                    confidence=0.0,
                    recognized_at=session.end_time,
                )
            )
            created += 1

        session.status = SessionStatus.CLOSED
        db.commit()
        return created


attendance_service = AttendanceService()
