from __future__ import annotations

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.db.models import StudentAttendance, StudentSchedule, User
from app.schemas.student_portal import AttendanceCreate, ScheduleCreate


class StudentPortalService:
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
