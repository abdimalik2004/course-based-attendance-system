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
    SessionType,
    Student,
    StudentAdmissionStatus,
    Teacher,
    User,
)
from app.services.face_service import face_service
from app.utils.datetime_utils import current_local_datetime, schedule_weekday_from_datetime
from app.utils.weekday_utils import decode_weekday_storage
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
        try:
            recognition = face_service.recognize_student(frame)
        except RuntimeError as exc:
            return {"ok": False, "status": "error", "message": str(exc)}

        if not recognition.get("matched"):
            reason = recognition.get("reason", "not_recognized")
            pt = recognition.get("processing_time")
            if reason == "no_face":
                # Nothing in the frame — client should silently retry
                return {"ok": False, "status": "no_face", "message": "No face detected", "processing_time": pt}
            if reason == "low_light":
                # Face detected but frame is too dark to produce a usable embedding
                return {"ok": False, "status": "low_light", "message": "Improve lighting", "processing_time": pt}
            if reason in ("below_threshold",):
                # Face detected but recognition quality too low — likely partial face,
                # mask, dark sunglasses, hand over mouth/nose, or scarf
                return {
                    "ok": False,
                    "status": "partial_face",
                    "message": "Full face required — remove mask or sunglasses",
                    "confidence": recognition.get("confidence"),
                    "processing_time": pt,
                }
            if reason == "timeout":
                # Treat timeout the same as no_face — silently retry
                return {"ok": False, "status": "no_face", "message": "Recognition timeout", "processing_time": pt}
            # not_recognized — face visible and quality OK but student not enrolled / unknown
            return {"ok": False, "status": "not_recognized", "message": "Face not matched", "processing_time": pt}

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

        # Only approved students may take attendance.
        # Return the same response as an unrecognised face so the scanner UI
        # (and any network observer) cannot tell the difference — the student
        # is simply treated as unknown until their admission is approved.
        if student.status != StudentAdmissionStatus.APPROVED:
            return {
                "ok": False,
                "status": "not_recognized",
                "message": "Face not matched",
            }

        if student.faculty_id != session.course.faculty_id:
            return {"ok": False, "message": "Student does not belong to course faculty"}

        # Department-level guard: a student from Animal Science must not be able
        # to mark attendance for a Plant Protection session (or any other
        # department inside the same faculty).  Both the student and the course
        # carry a department_id so the check is a direct comparison.
        if (
            student.department_id
            and session.course.department_id
            and student.department_id != session.course.department_id
        ):
            return {
                "ok": False,
                "status": "not_recognized",
                "message": "Face not matched",
            }

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
            _first = student.full_name.split()[0] if student.full_name else student.student_number
            return {
                "ok": True,
                "status": "already_marked",
                "message": "Attendance already marked",
                "record_id": existing.id,
                "student_number": student.student_number,
                "full_name": student.full_name or student.student_number,
                "first_name": _first,
            }

        grace_minutes = session.schedule.grace_period_minutes if session.schedule is not None else 0
        grace_cutoff = session.start_time + timedelta(minutes=grace_minutes)
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

        _first = student.full_name.split()[0] if student.full_name else student.student_number
        return {
            "ok": True,
            "status": "success",
            "message": "Attendance recorded",
            "record_id": record.id,
            "attendance_status": record.status.value,  # PRESENT or LATE
            "student_number": student.student_number,
            "full_name": student.full_name or student.student_number,
            "first_name": _first,
            "confidence": record.confidence,
            "processing_time": recognition.get("processing_time"),
        }

    def start_session(self, db: Session, course_id: int, schedule_id: int | None, session_type: "SessionType", actor_id: int) -> dict:
        if course_id <= 0 or actor_id <= 0:
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

        actor = db.query(User).filter(User.id == actor_id, User.is_active.is_(True)).first()
        if not actor:
            return {"ok": False, "message": "Actor not found"}

        teacher = db.query(Teacher).filter(Teacher.user_id == actor_id).first()
        teacher_id = teacher.id if teacher is not None else None
        admin_id = None

        # Enforce that a teacher is assigned to this course
        if teacher_id is not None:
            from app.db.models import CourseAssignment
            assigned = (
                db.query(CourseAssignment)
                .filter(
                    CourseAssignment.teacher_id == teacher_id,
                    CourseAssignment.course_id == course_id,
                )
                .first()
            )
            if not assigned:
                return {"ok": False, "message": "You are not assigned to this course"}

        if teacher_id is None:
            role_names = {role.name for role in actor.roles}
            _admin_roles = {"SUPER_ADMIN", "ACADEMIA", "FACULTY", "HR", "ADMISSIONS"}
            if role_names & _admin_roles:
                admin_id = actor_id
            else:
                return {"ok": False, "message": "Only a teacher or admin can start attendance sessions"}

        # ── Schedule day/time enforcement ─────────────────────────────────────
        now = current_local_datetime()

        # 1. Check that today is one of the scheduled days
        current_weekday_int = schedule_weekday_from_datetime(now)
        schedule_weekdays = set(decode_weekday_storage(schedule.weekday))
        if schedule_weekdays and current_weekday_int not in schedule_weekdays:
            _day_names = {
                1: "Saturday", 2: "Sunday", 3: "Monday",
                4: "Tuesday", 5: "Wednesday", 6: "Thursday", 7: "Friday",
            }
            scheduled_days = ", ".join(
                _day_names[d] for d in sorted(schedule_weekdays) if d in _day_names
            )
            return {
                "ok": False,
                "error_code": "WRONG_DAY",
                "message": (
                    f"This course is not scheduled for today. "
                    f"It runs on: {scheduled_days}."
                ),
            }

        # 2. Check that the current time is within the scheduled window
        current_time = now.time()
        if current_time < schedule.start_time:
            start_str = schedule.start_time.strftime("%H:%M")
            return {
                "ok": False,
                "error_code": "TOO_EARLY",
                "message": (
                    f"This session cannot be started yet. "
                    f"The scheduled time begins at {start_str}."
                ),
            }
        if current_time > schedule.end_time:
            end_str = schedule.end_time.strftime("%H:%M")
            return {
                "ok": False,
                "error_code": "TOO_LATE",
                "message": (
                    f"The scheduled time for this course has passed. "
                    f"Sessions for this slot close at {end_str}."
                ),
            }
        # ─────────────────────────────────────────────────────────────────────

        today = now.date()

        # Block re-starting a session that was already run and closed today.
        already_closed_today = (
            db.query(AttendanceSession)
            .filter(
                AttendanceSession.course_id == course_id,
                AttendanceSession.status == SessionStatus.CLOSED,
                AttendanceSession.session_date == today,
            )
            .first()
        )
        if already_closed_today:
            return {
                "ok": False,
                "error_code": "SESSION_ALREADY_COMPLETED",
                "message": (
                    "A session for this course was already held and closed today. "
                    "It cannot be started again on the same day."
                ),
            }

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

        session = AttendanceSession(
            course_id=course_id,
            teacher_id=teacher_id,
            admin_id=admin_id,
            schedule_id=schedule.id,  # use the resolved schedule, not the raw (possibly-None) parameter
            session_date=now.date(),
            start_time=now,
            end_time=None,
            session_type=session_type,
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
