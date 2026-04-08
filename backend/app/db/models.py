from __future__ import annotations

import enum
from datetime import date, datetime, time

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Text,
    Time,
    UniqueConstraint,
    func,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


_POSITIVE_ID_FIELDS: dict[str, tuple[str, ...]] = {
    "user_role_links": ("user_id", "role_id"),
    "roles": ("id",),
    "organizational_units": ("id",),
    "faculties": ("id",),
    "departments": ("id", "faculty_id"),
    "class_batches": ("id", "faculty_id", "department_id"),
    "users": ("id", "faculty_id"),
    "students": ("id", "faculty_id", "department_id", "class_batch_id"),
    "teachers": ("id", "faculty_id", "department_id", "user_id"),
    "courses": ("id", "faculty_id"),
    "course_assignments": ("id", "course_id", "teacher_id"),
    "enrollments": ("id", "student_id", "course_id"),
    "course_schedules": ("id", "course_id"),
    "course_schedule_weekdays": ("id", "schedule_id", "weekday"),
    "attendance_sessions": ("id", "course_id", "schedule_id"),
    "attendance_records": ("id", "student_id", "course_id", "session_id"),
}


def _validate_positive_id_fields(_mapper, _connection, target) -> None:
    table_name = target.__tablename__
    for field_name in _POSITIVE_ID_FIELDS.get(table_name, ()):  # pragma: no branch - tiny tuple loop
        value = getattr(target, field_name, None)
        if value is None:
            continue
        if isinstance(value, bool) or int(value) <= 0:
            raise ValueError(f"{table_name}.{field_name} must be a positive integer")


class AttendanceStatus(str, enum.Enum):
    PRESENT = "PRESENT"
    LATE = "LATE"
    ABSENT = "ABSENT"


class SessionStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    CLOSED = "CLOSED"


class UserRoleLink(Base):
    __tablename__ = "user_role_links"

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    role_id: Mapped[int] = mapped_column(ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True)


class Role(Base):
    __tablename__ = "roles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)

    users: Mapped[list["User"]] = relationship(
        secondary="user_role_links", back_populates="roles", lazy="selectin"
    )


class OrganizationalUnit(Base):
    __tablename__ = "organizational_units"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    code: Mapped[str] = mapped_column(String(32), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)


class Faculty(Base):
    __tablename__ = "faculties"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(150), unique=True, nullable=False)
    code: Mapped[str] = mapped_column(String(30), unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    departments: Mapped[list["Department"]] = relationship(back_populates="faculty", cascade="all, delete-orphan")
    class_batches: Mapped[list["ClassBatch"]] = relationship(back_populates="faculty", cascade="all, delete-orphan")
    courses: Mapped[list["Course"]] = relationship(back_populates="faculty")
    students: Mapped[list["Student"]] = relationship(back_populates="faculty")
    teachers: Mapped[list["Teacher"]] = relationship(back_populates="faculty")


class Department(Base):
    __tablename__ = "departments"
    __table_args__ = (
        UniqueConstraint("faculty_id", "name", name="uq_department_faculty_name"),
        UniqueConstraint("faculty_id", "code", name="uq_department_faculty_code"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    faculty_id: Mapped[int] = mapped_column(ForeignKey("faculties.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(150), nullable=False)
    code: Mapped[str] = mapped_column(String(30), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    faculty: Mapped[Faculty] = relationship(back_populates="departments")
    class_batches: Mapped[list["ClassBatch"]] = relationship(back_populates="department", cascade="all, delete-orphan")
    students: Mapped[list["Student"]] = relationship(back_populates="department")
    teachers: Mapped[list["Teacher"]] = relationship(back_populates="department")


class ClassBatch(Base):
    __tablename__ = "class_batches"
    __table_args__ = (UniqueConstraint("department_id", "name", name="uq_class_batch_department_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    faculty_id: Mapped[int] = mapped_column(ForeignKey("faculties.id", ondelete="CASCADE"), nullable=False)
    department_id: Mapped[int] = mapped_column(ForeignKey("departments.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(80), nullable=False)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)

    faculty: Mapped[Faculty] = relationship(back_populates="class_batches")
    department: Mapped[Department] = relationship(back_populates="class_batches")
    students: Mapped[list["Student"]] = relationship(back_populates="class_batch", cascade="all, delete-orphan")


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(80), unique=True, nullable=False, index=True)
    email: Mapped[str | None] = mapped_column(String(150), unique=True, nullable=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    faculty_id: Mapped[int | None] = mapped_column(ForeignKey("faculties.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    roles: Mapped[list[Role]] = relationship(secondary="user_role_links", back_populates="users", lazy="selectin")

    @property
    def role_names(self) -> list[str]:
        return [role.name for role in self.roles]


class Student(Base):
    __tablename__ = "students"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    student_number: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    full_name: Mapped[str] = mapped_column(String(180), nullable=False)
    faculty_id: Mapped[int] = mapped_column(ForeignKey("faculties.id", ondelete="CASCADE"), nullable=False)
    department_id: Mapped[int] = mapped_column(ForeignKey("departments.id", ondelete="CASCADE"), nullable=False)
    class_batch_id: Mapped[int] = mapped_column(ForeignKey("class_batches.id", ondelete="CASCADE"), nullable=False)
    embedding_ref: Mapped[str | None] = mapped_column(String(255), nullable=True)

    faculty: Mapped[Faculty] = relationship(back_populates="students")
    department: Mapped[Department] = relationship(back_populates="students")
    class_batch: Mapped[ClassBatch] = relationship(back_populates="students")
    enrollments: Mapped[list["Enrollment"]] = relationship(back_populates="student", cascade="all, delete-orphan")
    attendance_records: Mapped[list["AttendanceRecord"]] = relationship(back_populates="student")


class Teacher(Base):
    __tablename__ = "teachers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    teacher_number: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    full_name: Mapped[str] = mapped_column(String(180), nullable=False)
    faculty_id: Mapped[int] = mapped_column(ForeignKey("faculties.id", ondelete="CASCADE"), nullable=False)
    department_id: Mapped[int] = mapped_column(ForeignKey("departments.id", ondelete="CASCADE"), nullable=False)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    faculty: Mapped[Faculty] = relationship(back_populates="teachers")
    department: Mapped[Department] = relationship(back_populates="teachers")
    course_assignments: Mapped[list["CourseAssignment"]] = relationship(back_populates="teacher")


class Course(Base):
    __tablename__ = "courses"
    __table_args__ = (
        UniqueConstraint("faculty_id", "code", name="uq_course_faculty_code"),
        UniqueConstraint("faculty_id", "normalized_title", name="uq_course_faculty_normalized_title"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    faculty_id: Mapped[int] = mapped_column(ForeignKey("faculties.id", ondelete="CASCADE"), nullable=False, index=True)
    code: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    normalized_title: Mapped[str] = mapped_column(String(200), nullable=False)

    faculty: Mapped[Faculty] = relationship(back_populates="courses")
    assignments: Mapped[list["CourseAssignment"]] = relationship(back_populates="course", cascade="all, delete-orphan")
    enrollments: Mapped[list["Enrollment"]] = relationship(back_populates="course", cascade="all, delete-orphan")
    schedules: Mapped[list["CourseSchedule"]] = relationship(back_populates="course", cascade="all, delete-orphan")
    sessions: Mapped[list["AttendanceSession"]] = relationship(back_populates="course")


def normalize_course_title(value: str) -> str:
    return value.strip().lower()


@event.listens_for(Course, "before_insert")
def _sync_course_uniqueness_columns_before_insert(_mapper, _connection, target: Course) -> None:
    if target.faculty_id is None or int(target.faculty_id) <= 0:
        raise ValueError("courses.faculty_id must be a positive integer")
    target.normalized_title = normalize_course_title(target.title)


@event.listens_for(Course, "before_update")
def _sync_course_uniqueness_columns_before_update(_mapper, _connection, target: Course) -> None:
    if target.faculty_id is None or int(target.faculty_id) <= 0:
        raise ValueError("courses.faculty_id must be a positive integer")
    target.normalized_title = normalize_course_title(target.title)


class CourseAssignment(Base):
    __tablename__ = "course_assignments"
    __table_args__ = (UniqueConstraint("course_id", "teacher_id", name="uq_course_teacher_assignment"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    course_id: Mapped[int] = mapped_column(ForeignKey("courses.id", ondelete="CASCADE"), nullable=False)
    teacher_id: Mapped[int] = mapped_column(ForeignKey("teachers.id", ondelete="CASCADE"), nullable=False)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    course: Mapped[Course] = relationship(back_populates="assignments")
    teacher: Mapped[Teacher] = relationship(back_populates="course_assignments")


class Enrollment(Base):
    __tablename__ = "enrollments"
    __table_args__ = (UniqueConstraint("student_id", "course_id", name="uq_student_course_enrollment"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id", ondelete="CASCADE"), nullable=False)
    course_id: Mapped[int] = mapped_column(ForeignKey("courses.id", ondelete="CASCADE"), nullable=False)

    student: Mapped[Student] = relationship(back_populates="enrollments")
    course: Mapped[Course] = relationship(back_populates="enrollments")


class CourseSchedule(Base):
    __tablename__ = "course_schedules"
    __table_args__ = (
        UniqueConstraint("course_id", "weekday", "start_time", name="uq_course_weekday_time"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    course_id: Mapped[int] = mapped_column(ForeignKey("courses.id", ondelete="CASCADE"), nullable=False)
    weekday: Mapped[str] = mapped_column(String(64), nullable=False)  # e.g. "sat" or "sat,sun,mon"
    start_time: Mapped[time] = mapped_column(Time, nullable=False)
    end_time: Mapped[time] = mapped_column(Time, nullable=False)
    grace_period_minutes: Mapped[int] = mapped_column(Integer, default=10, nullable=False)

    course: Mapped[Course] = relationship(back_populates="schedules")
    weekday_rows: Mapped[list["CourseScheduleWeekday"]] = relationship(
        back_populates="schedule",
        cascade="all, delete-orphan",
    )
    sessions: Mapped[list["AttendanceSession"]] = relationship(back_populates="schedule")


class CourseScheduleWeekday(Base):
    __tablename__ = "course_schedule_weekdays"
    __table_args__ = (UniqueConstraint("schedule_id", "weekday", name="uq_schedule_weekday"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    schedule_id: Mapped[int] = mapped_column(ForeignKey("course_schedules.id", ondelete="CASCADE"), nullable=False)
    weekday: Mapped[int] = mapped_column(Integer, nullable=False)

    schedule: Mapped[CourseSchedule] = relationship(back_populates="weekday_rows")


class AttendanceSession(Base):
    __tablename__ = "attendance_sessions"
    __table_args__ = (
        UniqueConstraint("schedule_id", "session_date", "start_time", name="uq_schedule_session_occurrence"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    course_id: Mapped[int] = mapped_column(ForeignKey("courses.id", ondelete="CASCADE"), nullable=False)
    schedule_id: Mapped[int] = mapped_column(ForeignKey("course_schedules.id", ondelete="CASCADE"), nullable=False)
    session_date: Mapped[date] = mapped_column(Date, nullable=False)
    start_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    end_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    status: Mapped[SessionStatus] = mapped_column(Enum(SessionStatus), default=SessionStatus.ACTIVE, nullable=False)

    course: Mapped[Course] = relationship(back_populates="sessions")
    schedule: Mapped[CourseSchedule] = relationship(back_populates="sessions")
    records: Mapped[list["AttendanceRecord"]] = relationship(back_populates="session", cascade="all, delete-orphan")


class AttendanceRecord(Base):
    __tablename__ = "attendance_records"
    __table_args__ = (
        UniqueConstraint("student_id", "course_id", "session_id", name="uq_student_course_session_attendance"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("students.id", ondelete="CASCADE"), nullable=False)
    course_id: Mapped[int] = mapped_column(ForeignKey("courses.id", ondelete="CASCADE"), nullable=False)
    session_id: Mapped[int] = mapped_column(ForeignKey("attendance_sessions.id", ondelete="CASCADE"), nullable=False)
    status: Mapped[AttendanceStatus] = mapped_column(Enum(AttendanceStatus), nullable=False)
    confidence: Mapped[float] = mapped_column(nullable=False, default=0.0)
    recognized_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    raw_payload: Mapped[str | None] = mapped_column(Text, nullable=True)

    student: Mapped[Student] = relationship(back_populates="attendance_records")
    session: Mapped[AttendanceSession] = relationship(back_populates="records")


for _model in (
    UserRoleLink,
    Role,
    OrganizationalUnit,
    Faculty,
    Department,
    ClassBatch,
    User,
    Student,
    Teacher,
    Course,
    CourseAssignment,
    Enrollment,
    CourseSchedule,
    CourseScheduleWeekday,
    AttendanceSession,
    AttendanceRecord,
):
    event.listen(_model, "before_insert", _validate_positive_id_fields)
    event.listen(_model, "before_update", _validate_positive_id_fields)
