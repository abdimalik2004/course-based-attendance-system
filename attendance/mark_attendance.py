from datetime import datetime

from attendance.rules import derive_attendance_status
from database.db import Database
from utils.logging import get_logger
from utils.tts import speak_async


logger = get_logger(__name__)


def mark_attendance(db: Database, student_id: str, course_id: str, confidence: float, session_key: str):
    if db.attendance_exists(student_id, course_id, session_key):
        return False
    course = db.get_course(course_id)
    status = derive_attendance_status(course, datetime.now())
    db.record_attendance(student_id, course_id, confidence, session_key, status)
    speak_async("Thank you", rate=170)
    logger.info(
        "Attendance marked: student_id=%s course_id=%s status=%s confidence=%.2f session=%s",
        student_id,
        course_id,
        status,
        confidence,
        session_key,
    )
    return True
