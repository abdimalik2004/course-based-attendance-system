from __future__ import annotations

from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from app.db.models import (
    AttendanceRecord,
    AttendanceSession,
    AttendanceStatus,
    Enrollment,
    SessionStatus,
    Student,
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
        if now < session.start_time or now > session.end_time:
            return {"ok": False, "message": "Outside session window"}

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
            or student.class_batch_id <= 0
        ):
            return {"ok": False, "message": "Recognized student has invalid references"}

        if student.class_batch_id != session.course.class_batch_id:
            return {"ok": False, "message": "Student does not belong to class"}

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

    def close_session_and_mark_absent(self, db: Session, session: AttendanceSession) -> int:
        if session.status == SessionStatus.CLOSED:
            return 0

        enrolled_students = (
            db.query(Student)
            .join(Enrollment, Enrollment.student_id == Student.id)
            .filter(
                Enrollment.course_id == session.course_id,
                Student.id > 0,
                Student.class_batch_id > 0,
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
