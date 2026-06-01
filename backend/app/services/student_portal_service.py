from __future__ import annotations

import logging

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

from app.db.models import (
    AttendanceRecord,
    AttendanceSession,
    AttendanceStatus,
    ClassBatch,
    ClassCourseAssignment,
    Course,
    CourseSchedule,
    Enrollment,
    SessionStatus,
    Student,
    StudentAttendance,
    StudentSchedule,
    User,
)
from app.schemas.student_portal import AttendanceCreate, ScheduleCreate


# Map lowercase stored weekday abbreviations → display labels used by the frontend
_WEEKDAY_DISPLAY = {
    "sun": "Sun", "mon": "Mon", "tue": "Tue",
    "wed": "Wed", "thu": "Thu", "fri": "Fri", "sat": "Sat",
}


def _weekday_to_display(raw: str) -> list[str]:
    """Convert a stored weekday string (e.g. 'mon' or 'mon,wed,fri') to a display list."""
    parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
    return [_WEEKDAY_DISPLAY.get(p, p.capitalize()) for p in parts]


class StudentPortalService:
    # ------------------------------------------------------------------
    # /me/ endpoints — live data scoped to the logged-in student
    # ------------------------------------------------------------------

    def _find_student_for_user(self, db: Session, user: User) -> Student | None:
        """
        Look up the Student record that belongs to this User.
        Preferred: use the student_id FK stored directly on the User row.
        Fallback: match Student.student_number == User.username (legacy convention).
        Returns None (not 404) so callers can return an empty list gracefully.
        """
        if user.student_id is not None:
            return (
                db.query(Student)
                .filter(Student.id == user.student_id)
                .first()
            )
        # Legacy fallback: username was set to the student number at account creation
        return (
            db.query(Student)
            .filter(Student.student_number == user.username)
            .first()
        )

    def get_my_attendance(self, db: Session, user: User) -> list[dict]:
        """
        Return per-course attendance summary for the logged-in student.
        Reads live data from AttendanceRecord — NOT the stale StudentAttendance snapshot.
        """
        student = self._find_student_for_user(db, user)
        if student is None:
            return []

        rows = (
            db.query(AttendanceRecord, Course)
            .join(Course, Course.id == AttendanceRecord.course_id)
            .filter(AttendanceRecord.student_id == student.id)
            .all()
        )

        # Aggregate by course
        course_map: dict[int, dict] = {}
        for record, course in rows:
            if course.id not in course_map:
                course_map[course.id] = {
                    "id": course.id,
                    "course_name": course.title,
                    "course_code": course.code,
                    "classes_attended": 0,
                    "classes_absent": 0,
                    "classes_excused": 0,
                    "total_classes": 0,
                    "last_updated": None,
                }
            entry = course_map[course.id]
            entry["total_classes"] += 1
            if record.status in (AttendanceStatus.PRESENT, AttendanceStatus.LATE):
                entry["classes_attended"] += 1
            elif record.status == AttendanceStatus.ABSENT:
                entry["classes_absent"] += 1
            elif record.status == AttendanceStatus.EXCUSED:
                entry["classes_excused"] += 1
            if record.recognized_at and (
                entry["last_updated"] is None
                or record.recognized_at > entry["last_updated"]
            ):
                entry["last_updated"] = record.recognized_at

        result = []
        for entry in course_map.values():
            total = entry["total_classes"]
            attended = entry["classes_attended"]
            pct = round((attended / total) * 100, 1) if total > 0 else 0.0

            if pct >= 85:
                display_status = "Good"
            elif pct >= 70:
                display_status = "Warning"
            else:
                display_status = "Low"

            result.append({
                "id": entry["id"],
                "course_name": entry["course_name"],
                "course_code": entry["course_code"],
                "classes_attended": attended,
                "classes_absent": entry["classes_absent"],
                "classes_excused": entry["classes_excused"],
                "total_classes": total,
                "attendance_percentage": pct,
                "status": display_status,
                "created_at": entry["last_updated"].isoformat() if entry["last_updated"] else None,
            })

        return sorted(result, key=lambda r: r["course_name"])

    def get_my_schedule(self, db: Session, user: User) -> list[dict]:
        """
        Return the course schedule for the logged-in student.
        Reads live data from Enrollment → Course → CourseSchedule.
        Also resolves the class batch name for each course (scoped to the
        student's own department so the correct class is shown).
        """
        student = self._find_student_for_user(db, user)
        if student is None:
            logger.info("get_my_schedule: no Student record found for user_id=%s username=%s", user.id, user.username)
            return []

        logger.info(
            "get_my_schedule: student id=%s dept_id=%s faculty_id=%s",
            student.id, student.department_id, student.faculty_id,
        )

        rows = (
            db.query(CourseSchedule, Course)
            .join(Course, Course.id == CourseSchedule.course_id)
            .join(Enrollment, Enrollment.course_id == Course.id)
            .filter(Enrollment.student_id == student.id)
            .order_by(Course.title, CourseSchedule.start_time)
            .all()
        )

        logger.info("get_my_schedule: found %d schedule rows", len(rows))

        # Build a map: course_id → class batch name.
        # Priority order:
        #   1. ClassCourseAssignment scoped to the student's department
        #   2. ClassCourseAssignment for the course (any department)
        #   3. Any ClassBatch in the student's own department (no formal assignment needed)
        #   4. Any ClassBatch in the student's faculty (broadest catch-all)
        course_ids = list({course.id for _, course in rows})
        logger.info("get_my_schedule: course_ids=%s", course_ids)
        class_name_map: dict[int, str] = {}
        if course_ids:
            def _fetch_assignments(extra_filter=None):
                q = (
                    db.query(ClassCourseAssignment, ClassBatch)
                    .join(ClassBatch, ClassBatch.id == ClassCourseAssignment.class_id)
                    .filter(ClassCourseAssignment.course_id.in_(course_ids))
                )
                if extra_filter is not None:
                    q = q.filter(extra_filter)
                return q.all()

            # 1. Try department-scoped ClassCourseAssignment first (most specific).
            #    Each course should map to exactly one class for this student.
            dept_assignments = (
                _fetch_assignments(ClassBatch.department_id == student.department_id)
                if student.department_id
                else []
            )
            logger.info("get_my_schedule: step-1 dept-scoped assignments=%d", len(dept_assignments))
            for assignment, class_batch in dept_assignments:
                # Only set once — first dept-scoped match wins per course.
                if assignment.course_id not in class_name_map:
                    class_name_map[assignment.course_id] = class_batch.name

            # 2. For courses still missing, fall back to any ClassCourseAssignment.
            #    Take only the FIRST match per course — a student belongs to one class,
            #    so concatenating multiple assignments would show incorrect data.
            missing_after_step1 = [cid for cid in course_ids if cid not in class_name_map]
            if missing_after_step1:
                fallback_assignments = _fetch_assignments(
                    ClassCourseAssignment.course_id.in_(missing_after_step1)
                )
                logger.info("get_my_schedule: step-2 fallback assignments=%d for %d missing courses",
                            len(fallback_assignments), len(missing_after_step1))
                for assignment, class_batch in fallback_assignments:
                    # First match per course wins — do NOT concatenate.
                    if assignment.course_id not in class_name_map:
                        class_name_map[assignment.course_id] = class_batch.name

            # 3. For any course still missing a class name, use the first ClassBatch in
            #    the student's department. Only one name — never concatenate.
            missing_cids = [cid for cid in course_ids if cid not in class_name_map]
            if missing_cids and student.department_id:
                dept_batch = (
                    db.query(ClassBatch)
                    .filter(ClassBatch.department_id == student.department_id)
                    .order_by(ClassBatch.name)
                    .first()
                )
                logger.info("get_my_schedule: step-3 dept batch=%s for %d missing courses",
                            dept_batch.name if dept_batch else None, len(missing_cids))
                if dept_batch:
                    for cid in missing_cids:
                        class_name_map[cid] = dept_batch.name

            # 4. Last resort: first ClassBatch in the student's faculty.
            missing_cids = [cid for cid in course_ids if cid not in class_name_map]
            if missing_cids and student.faculty_id:
                faculty_batch = (
                    db.query(ClassBatch)
                    .filter(ClassBatch.faculty_id == student.faculty_id)
                    .order_by(ClassBatch.name)
                    .first()
                )
                logger.info("get_my_schedule: step-4 faculty batch=%s for %d missing courses",
                            faculty_batch.name if faculty_batch else None, len(missing_cids))
                if faculty_batch:
                    for cid in missing_cids:
                        class_name_map[cid] = faculty_batch.name

        logger.info("get_my_schedule: final class_name_map=%s", class_name_map)

        # Fetch all active sessions for these courses in one query so we can
        # mark which courses currently have an ongoing session.
        active_course_ids: set[int] = set()
        if course_ids:
            active_rows = (
                db.query(AttendanceSession.course_id)
                .filter(
                    AttendanceSession.course_id.in_(course_ids),
                    AttendanceSession.status == SessionStatus.ACTIVE,
                )
                .all()
            )
            active_course_ids = {row.course_id for row in active_rows}

        return [
            {
                "id": schedule.id,
                "course_id": course.id,
                "course_name": course.title,
                "course_code": course.code,
                "weekdays": _weekday_to_display(schedule.weekday),
                "start_time": schedule.start_time.strftime("%H:%M"),
                "end_time": schedule.end_time.strftime("%H:%M"),
                "grace_period_minutes": schedule.grace_period_minutes,
                "class_name": class_name_map.get(course.id),
                "has_active_session": course.id in active_course_ids,
            }
            for schedule, course in rows
        ]

    # ------------------------------------------------------------------
    # Legacy snapshot-based methods (kept for backward compatibility)
    # ------------------------------------------------------------------

    def _load_student(self, db: Session, student_id: int) -> User:
        student = db.query(User).filter(User.id == student_id).first()
        if student is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")
        return student

    def list_attendance(self, db: Session, student_id: int) -> list[StudentAttendance]:
        self._load_student(db, student_id)
        return (
            db.query(StudentAttendance)
            .filter(StudentAttendance.student_id == student_id)
            .order_by(StudentAttendance.created_at.desc(), StudentAttendance.id.desc())
            .all()
        )

    def get_attendance(self, db: Session, attendance_id: int) -> StudentAttendance:
        attendance = db.query(StudentAttendance).filter(StudentAttendance.id == attendance_id).first()
        if attendance is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Attendance record not found")
        return attendance

    def create_attendance(self, db: Session, payload: AttendanceCreate) -> StudentAttendance:
        self._load_student(db, payload.student_id)

        attendance = StudentAttendance(
            student_id=payload.student_id,
            course_name=payload.course_name.strip(),
            course_code=payload.course_code.strip(),
            classes_attended=payload.classes_attended,
            total_classes=payload.total_classes,
        )
        db.add(attendance)
        db.commit()
        db.refresh(attendance)
        return attendance

    def list_schedules(self, db: Session, student_id: int) -> list[StudentSchedule]:
        self._load_student(db, student_id)
        return (
            db.query(StudentSchedule)
            .filter(StudentSchedule.student_id == student_id)
            .order_by(StudentSchedule.created_at.desc(), StudentSchedule.id.desc())
            .all()
        )

    def get_schedule(self, db: Session, schedule_id: int) -> StudentSchedule:
        schedule = db.query(StudentSchedule).filter(StudentSchedule.id == schedule_id).first()
        if schedule is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Schedule not found")
        return schedule

    def create_schedule(self, db: Session, payload: ScheduleCreate) -> StudentSchedule:
        self._load_student(db, payload.student_id)

        schedule = StudentSchedule(
            student_id=payload.student_id,
            course_name=payload.course_name.strip(),
            course_code=payload.course_code.strip(),
            weekdays=payload.weekdays,
            start_time=payload.start_time,
            end_time=payload.end_time,
            grace_period_minutes=payload.grace_period_minutes,
        )
        db.add(schedule)
        db.commit()
        db.refresh(schedule)
        return schedule


student_portal_service = StudentPortalService()
