from __future__ import annotations

import enum
from typing import Optional
from datetime import date, datetime, time

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    Enum,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    Time,
    UniqueConstraint,
    func,
    event,
    select,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


_POSITIVE_ID_FIELDS: dict[str, tuple[str, ...]] = {
    "user_role_links": ("user_id", "role_id"),
    "roles": ("id",),
    "organizational_units": ("id",),
    "faculties": ("id",),
    "academic_years": ("id",),
    "departments": ("id", "faculty_id"),
    "class_batches": ("id", "faculty_id", "department_id"),
    "users": ("id", "faculty_id", "student_id"),
    "students": ("id", "faculty_id", "department_id"),
    "teachers": ("id", "faculty_id", "department_id", "user_id"),
    "courses": ("id", "faculty_id"),
    "course_semester_assignments": ("id", "course_id", "faculty_id", "department_id", "academic_year_id"),
    "class_course_assignments": ("id", "class_id", "course_id", "faculty_id", "department_id"),
    "course_assignments": ("id", "course_id", "teacher_id"),
    "enrollments": ("id", "student_id", "course_id"),
    "course_schedules": ("id", "course_id"),
    "course_schedule_weekdays": ("id", "schedule_id", "weekday"),
    "attendance_sessions": ("id", "course_id", "teacher_id", "admin_id", "schedule_id"),
    "attendance_records": ("id", "student_id", "course_id", "session_id"),
    "student_attendance": ("id", "student_id"),
    "student_schedule": ("id", "student_id"),
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
    EXCUSED = "EXCUSED"


class AttendanceSummaryStatus(str, enum.Enum):
    GOOD = "Good"
    WARNING = "Warning"
    LOW = "Low"


class SessionStatus(str, enum.Enum):
    ACTIVE = "ACTIVE"
    CLOSED = "CLOSED"


class SessionType(str, enum.Enum):
    LECTURE = "Lecture"
    LAB = "Lab"
    TUTORIAL = "Tutorial"


class AcademicYearStatus(str, enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    DRAFT = "draft"


class TeacherRole(str, enum.Enum):
    PROFESSOR = "Professor"
    ASSOCIATE_PROFESSOR = "Associate Professor"
    ASSISTANT_PROFESSOR = "Assistant Professor"
    LECTURER = "Lecturer"


class TeacherStatus(str, enum.Enum):
    ACTIVE = "Active"
    ONLEAVE = "Onleave"
    INACTIVE = "Inactive"


class StudentAdmissionStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


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
    __table_args__ = (
        CheckConstraint("years >= 3", name="ck_faculties_years_minimum"),
        CheckConstraint("semesters = years * 2", name="ck_faculties_semesters_match_years"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(150), unique=True, nullable=False)
    code: Mapped[str] = mapped_column(String(30), unique=True, nullable=False)
    years: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("4"))
    semesters: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("8"))
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    departments: Mapped[list["Department"]] = relationship(back_populates="faculty", cascade="all, delete-orphan")
    class_batches: Mapped[list["ClassBatch"]] = relationship(back_populates="faculty", cascade="all, delete-orphan")
    courses: Mapped[list["Course"]] = relationship(back_populates="faculty")
    course_semester_assignments: Mapped[list["CourseSemesterAssignment"]] = relationship(back_populates="faculty")
    class_course_assignments: Mapped[list["ClassCourseAssignment"]] = relationship(back_populates="faculty")
    students: Mapped[list["Student"]] = relationship(back_populates="faculty")
    teachers: Mapped[list["Teacher"]] = relationship(back_populates="faculty")


def _sync_faculty_duration(_mapper, _connection, target: Faculty) -> None:
    years = 4 if target.years is None else target.years
    if isinstance(years, bool):
        raise ValueError("faculties.years must be a positive integer")

    years_value = int(years)
    if years_value < 3:
        raise ValueError("faculties.years must be at least 3")

    target.years = years_value
    target.semesters = years_value * 2


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
    courses: Mapped[list["Course"]] = relationship(back_populates="department", cascade="all, delete-orphan")


class ClassBatch(Base):
    __tablename__ = "class_batches"
    __table_args__ = (UniqueConstraint("faculty_id", "name", name="uq_class_batch_faculty_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    faculty_id: Mapped[int] = mapped_column(ForeignKey("faculties.id", ondelete="CASCADE"), nullable=False)
    department_id: Mapped[int] = mapped_column(ForeignKey("departments.id", ondelete="CASCADE"), nullable=False)
    name: Mapped[str] = mapped_column(String(80), nullable=False)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)

    faculty: Mapped[Faculty] = relationship(back_populates="class_batches")
    department: Mapped[Department] = relationship(back_populates="class_batches")
    class_course_assignments: Mapped[list["ClassCourseAssignment"]] = relationship(back_populates="class_batch")


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(80), unique=True, nullable=False, index=True)
    email: Mapped[str | None] = mapped_column(String(150), unique=True, nullable=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    profile_image_url: Mapped[str | None] = mapped_column(String(255), nullable=True)
    faculty_id: Mapped[int | None] = mapped_column(ForeignKey("faculties.id"), nullable=True)
    student_id: Mapped[int | None] = mapped_column(ForeignKey("students.id", ondelete="SET NULL"), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    roles: Mapped[list[Role]] = relationship(secondary="user_role_links", back_populates="users", lazy="selectin")
    attendance_entries: Mapped[list["StudentAttendance"]] = relationship(back_populates="student")
    schedule_entries: Mapped[list["StudentSchedule"]] = relationship(back_populates="student")
    teacher: Mapped[Optional["Teacher"]] = relationship(back_populates="user", uselist=False)
    linked_student: Mapped[Optional["Student"]] = relationship(foreign_keys=[student_id], uselist=False)

    @property
    def role_names(self) -> list[str]:
        return [role.name for role in self.roles]

    @property
    def teacher_id(self) -> int | None:
        return self.teacher.id if self.teacher is not None else None

    @property
    def student_number(self) -> str | None:
        return self.linked_student.student_number if self.linked_student is not None else None

    @property
    def full_name(self) -> str | None:
        if self.teacher is not None:
            return self.teacher.full_name
        if self.linked_student is not None:
            return self.linked_student.full_name
        return None


class Student(Base):
    __tablename__ = "students"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    student_number: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    full_name: Mapped[str] = mapped_column(String(180), nullable=False)
    faculty_id: Mapped[int] = mapped_column(ForeignKey("faculties.id", ondelete="CASCADE"), nullable=False)
    department_id: Mapped[int] = mapped_column(ForeignKey("departments.id", ondelete="CASCADE"), nullable=False)
    embedding_ref: Mapped[str | None] = mapped_column(String(255), nullable=True)
    status: Mapped[StudentAdmissionStatus] = mapped_column(
        Enum(
            StudentAdmissionStatus,
            name="student_admission_status",
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        default=StudentAdmissionStatus.PENDING,
        nullable=False,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    faculty: Mapped[Faculty] = relationship(back_populates="students")
    department: Mapped[Department] = relationship(back_populates="students")
    enrollments: Mapped[list["Enrollment"]] = relationship(back_populates="student", cascade="all, delete-orphan")
    attendance_records: Mapped[list["AttendanceRecord"]] = relationship(back_populates="student", cascade="all, delete-orphan")


class Teacher(Base):
    __tablename__ = "teachers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    teacher_number: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    full_name: Mapped[str] = mapped_column(String(180), nullable=False)
    role: Mapped[TeacherRole] = mapped_column(
        Enum(
            TeacherRole,
            name="teacher_role",
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        default=TeacherRole.LECTURER,
        nullable=False,
    )
    status: Mapped[TeacherStatus] = mapped_column(
        Enum(
            TeacherStatus,
            name="teacher_status",
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        default=TeacherStatus.ACTIVE,
        nullable=False,
    )
    faculty_id: Mapped[int] = mapped_column(ForeignKey("faculties.id", ondelete="CASCADE"), nullable=False)
    department_id: Mapped[int] = mapped_column(ForeignKey("departments.id", ondelete="CASCADE"), nullable=False)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)

    faculty: Mapped[Faculty] = relationship(back_populates="teachers")
    department: Mapped[Department] = relationship(back_populates="teachers")
    course_assignments: Mapped[list["CourseAssignment"]] = relationship(back_populates="teacher")
    user: Mapped[Optional["User"]] = relationship(back_populates="teacher", foreign_keys=[user_id])


class Course(Base):
    __tablename__ = "courses"
    __table_args__ = (
        UniqueConstraint("faculty_id", "code", name="uq_course_faculty_code"),
        UniqueConstraint("faculty_id", "normalized_title", name="uq_course_faculty_normalized_title"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    faculty_id: Mapped[int] = mapped_column(ForeignKey("faculties.id", ondelete="CASCADE"), nullable=False, index=True)
    department_id: Mapped[int] = mapped_column(ForeignKey("departments.id", ondelete="CASCADE"), nullable=False)
    code: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    normalized_title: Mapped[str] = mapped_column(String(200), nullable=False)

    faculty: Mapped[Faculty] = relationship(back_populates="courses")
    department: Mapped[Department] = relationship(back_populates="courses")
    course_semester_assignments: Mapped[list["CourseSemesterAssignment"]] = relationship(back_populates="course", cascade="all, delete-orphan")
    class_course_assignments: Mapped[list["ClassCourseAssignment"]] = relationship(back_populates="course", cascade="all, delete-orphan")
    assignments: Mapped[list["CourseAssignment"]] = relationship(back_populates="course", cascade="all, delete-orphan")
    enrollments: Mapped[list["Enrollment"]] = relationship(back_populates="course", cascade="all, delete-orphan")
    schedules: Mapped[list["CourseSchedule"]] = relationship(back_populates="course", cascade="all, delete-orphan")
    sessions: Mapped[list["AttendanceSession"]] = relationship(back_populates="course")


class AcademicYear(Base):
    __tablename__ = "academic_years"
    __table_args__ = (
        UniqueConstraint("academic_year", "term_name", name="uq_academic_years_year_term"),
        CheckConstraint("end_date > start_date", name="ck_academic_years_date_order"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    academic_year: Mapped[str] = mapped_column(String(32), nullable=False)
    term_name: Mapped[str] = mapped_column(String(64), nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    status: Mapped[AcademicYearStatus] = mapped_column(
        Enum(
            AcademicYearStatus,
            name="academic_year_status",
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        default=AcademicYearStatus.DRAFT,
        nullable=False,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    course_semester_assignments: Mapped[list["CourseSemesterAssignment"]] = relationship(
        back_populates="academic_year",
        cascade="all, delete-orphan",
    )


class CourseSemesterAssignment(Base):
    __tablename__ = "course_semester_assignments"
    __table_args__ = (
        UniqueConstraint(
            "course_id",
            "faculty_id",
            "department_id",
            "academic_year_id",
            name="uq_course_semester_assignment",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    course_id: Mapped[int] = mapped_column(ForeignKey("courses.id", ondelete="CASCADE"), nullable=False, index=True)
    faculty_id: Mapped[int] = mapped_column(ForeignKey("faculties.id", ondelete="CASCADE"), nullable=False, index=True)
    department_id: Mapped[int] = mapped_column(ForeignKey("departments.id", ondelete="CASCADE"), nullable=False, index=True)
    academic_year_id: Mapped[int] = mapped_column(
        ForeignKey("academic_years.id"),
        nullable=False,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    course: Mapped[Course] = relationship(back_populates="course_semester_assignments")
    faculty: Mapped[Faculty] = relationship(back_populates="course_semester_assignments")
    department: Mapped[Department] = relationship()
    academic_year: Mapped[AcademicYear] = relationship(back_populates="course_semester_assignments")

    @property
    def semester(self) -> int:
        term_name = getattr(self.academic_year, "term_name", "") or ""
        digits = "".join(character for character in term_name if character.isdigit())
        if digits:
            return int(digits)
        return 1


class ClassCourseAssignment(Base):
    __tablename__ = "class_course_assignments"
    __table_args__ = (
        UniqueConstraint("class_id", "course_id", "faculty_id", "department_id", name="uq_class_course_assignment"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    class_id: Mapped[int] = mapped_column(ForeignKey("class_batches.id", ondelete="CASCADE"), nullable=False, index=True)
    course_id: Mapped[int] = mapped_column(ForeignKey("courses.id", ondelete="CASCADE"), nullable=False, index=True)
    faculty_id: Mapped[int] = mapped_column(ForeignKey("faculties.id", ondelete="CASCADE"), nullable=False, index=True)
    department_id: Mapped[int] = mapped_column(ForeignKey("departments.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    class_batch: Mapped[ClassBatch] = relationship(back_populates="class_course_assignments")
    course: Mapped[Course] = relationship(back_populates="class_course_assignments")
    faculty: Mapped[Faculty] = relationship(back_populates="class_course_assignments")
    department: Mapped[Department] = relationship()


def normalize_course_title(value: str) -> str:
    return value.strip().lower()


def _validate_academic_year_record(target: AcademicYear) -> None:
    if not isinstance(target.start_date, date) or not isinstance(target.end_date, date):
        raise ValueError("academic_years.start_date and academic_years.end_date are required")
    if target.end_date <= target.start_date:
        raise ValueError("academic_years.end_date must be later than start_date")


def _calculate_attendance_percentage(classes_attended: int, total_classes: int) -> float:
    if total_classes <= 0:
        raise ValueError("attendance.total_classes must be greater than 0")
    if classes_attended < 0:
        raise ValueError("attendance.classes_attended must be non-negative")
    if classes_attended > total_classes:
        raise ValueError("attendance.classes_attended cannot exceed total_classes")
    return round((classes_attended / total_classes) * 100, 2)


def _attendance_status_from_percentage(percentage: float) -> AttendanceSummaryStatus:
    if percentage >= 75:
        return AttendanceSummaryStatus.GOOD
    if percentage >= 50:
        return AttendanceSummaryStatus.WARNING
    return AttendanceSummaryStatus.LOW


def _normalize_weekdays(weekdays: list[str] | None) -> list[str]:
    allowed_weekdays = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}
    if not isinstance(weekdays, list) or not weekdays:
        raise ValueError("schedule.weekdays must be a non-empty list")

    normalized: list[str] = []
    seen: set[str] = set()
    for weekday in weekdays:
        if not isinstance(weekday, str):
            raise ValueError("schedule.weekdays must contain strings")
        candidate = weekday.strip().title()
        if candidate not in allowed_weekdays:
            raise ValueError("schedule.weekdays must contain valid weekday codes")
        if candidate in seen:
            raise ValueError("schedule.weekdays must not contain duplicates")
        seen.add(candidate)
        normalized.append(candidate)
    return normalized


def _validate_student_attendance_record(target: StudentAttendance) -> None:
    percentage = _calculate_attendance_percentage(target.classes_attended, target.total_classes)
    target.attendance_percentage = percentage
    target.status = _attendance_status_from_percentage(percentage)


def _validate_student_schedule_record(target: StudentSchedule) -> None:
    if target.end_time <= target.start_time:
        raise ValueError("schedule.end_time must be later than start_time")
    target.weekdays = _normalize_weekdays(target.weekdays)
    if target.grace_period_minutes < 0:
        raise ValueError("schedule.grace_period_minutes must be non-negative")


def _validate_single_active_academic_year(connection, target: AcademicYear) -> None:
    if target.status != AcademicYearStatus.ACTIVE:
        return

    query = select(func.count()).select_from(AcademicYear.__table__).where(
        AcademicYear.__table__.c.status == AcademicYearStatus.ACTIVE.value,
    )
    if target.id is not None:
        query = query.where(AcademicYear.__table__.c.id != target.id)
    active_count = int(connection.execute(query).scalar_one())
    if active_count > 0:
        raise ValueError("only one academic year can be active at a time")


@event.listens_for(AcademicYear, "before_insert")
def _validate_academic_year_before_insert(_mapper, connection, target: AcademicYear) -> None:
    _validate_academic_year_record(target)
    _validate_single_active_academic_year(connection, target)


@event.listens_for(AcademicYear, "before_update")
def _validate_academic_year_before_update(_mapper, connection, target: AcademicYear) -> None:
    _validate_academic_year_record(target)
    _validate_single_active_academic_year(connection, target)


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


class StudentAttendance(Base):
    __tablename__ = "student_attendance"
    __table_args__ = (
        CheckConstraint("classes_attended >= 0", name="ck_student_attendance_classes_attended_nonnegative"),
        CheckConstraint("total_classes > 0", name="ck_student_attendance_total_classes_positive"),
        CheckConstraint(
            "classes_attended <= total_classes",
            name="ck_student_attendance_classes_attended_not_greater_than_total",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    course_name: Mapped[str] = mapped_column(String(200), nullable=False)
    course_code: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    classes_attended: Mapped[int] = mapped_column(Integer, nullable=False)
    total_classes: Mapped[int] = mapped_column(Integer, nullable=False)
    attendance_percentage: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    status: Mapped[AttendanceSummaryStatus] = mapped_column(
        Enum(
            AttendanceSummaryStatus,
            name="attendance_summary_status",
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    student: Mapped[User] = relationship(back_populates="attendance_entries")


class StudentSchedule(Base):
    __tablename__ = "student_schedule"
    __table_args__ = (
        CheckConstraint("grace_period_minutes >= 0", name="ck_student_schedule_grace_period_nonnegative"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    student_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    course_name: Mapped[str] = mapped_column(String(200), nullable=False)
    course_code: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    weekdays: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    start_time: Mapped[time] = mapped_column(Time, nullable=False)
    end_time: Mapped[time] = mapped_column(Time, nullable=False)
    grace_period_minutes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    student: Mapped[User] = relationship(back_populates="schedule_entries")


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
    teacher_id: Mapped[int | None] = mapped_column(ForeignKey("teachers.id", ondelete="SET NULL"), nullable=True)
    admin_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    schedule_id: Mapped[int] = mapped_column(ForeignKey("course_schedules.id", ondelete="CASCADE"), nullable=False)
    session_date: Mapped[date] = mapped_column(Date, nullable=False)
    start_time: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    end_time: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    session_type: Mapped[SessionType] = mapped_column(
        Enum(
            SessionType,
            name="session_type",
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        default=SessionType.LECTURE,
        nullable=False,
    )
    status: Mapped[SessionStatus] = mapped_column(Enum(SessionStatus), default=SessionStatus.ACTIVE, nullable=False)

    course: Mapped[Course] = relationship(back_populates="sessions")
    teacher: Mapped[Teacher | None] = relationship(foreign_keys=[teacher_id])
    admin: Mapped[User | None] = relationship(foreign_keys=[admin_id])
    schedule: Mapped[CourseSchedule] = relationship(back_populates="sessions")
    records: Mapped[list["AttendanceRecord"]] = relationship(back_populates="session", cascade="all, delete-orphan")

    @property
    def course_name(self) -> str | None:
        return self.course.title if self.course is not None else None

    @property
    def course_code(self) -> str | None:
        return self.course.code if self.course is not None else None

    @property
    def grace_period_minutes(self) -> int | None:
        return self.schedule.grace_period_minutes if self.schedule is not None else None


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


class ActivityLogStatus(str, enum.Enum):
    SUCCESS = "Success"
    FAILED = "Failed"
    PENDING = "Pending"


class ActivityLog(Base):
    __tablename__ = "activity_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    username: Mapped[str] = mapped_column(String(128), nullable=False, default="System")
    action: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[ActivityLogStatus] = mapped_column(
        # values_callable tells SQLAlchemy to use the enum VALUES ("Success",
        # "Failed", "Pending") for storage and lookup — not the enum names
        # ("SUCCESS", "FAILED", "PENDING") which is the SQLAlchemy 2.x default.
        # This matches what is already stored in the database.
        Enum(
            ActivityLogStatus,
            name="activity_log_status",
            native_enum=False,
            values_callable=lambda obj: [e.value for e in obj],
        ),
        nullable=False,
        default=ActivityLogStatus.SUCCESS,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False, index=True)

    user: Mapped[Optional["User"]] = relationship(foreign_keys=[user_id])


class SystemSetting(Base):
    """Key-value store for system-wide configuration settings."""

    __tablename__ = "system_settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)
    value: Mapped[str] = mapped_column(Text, nullable=False, default="")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now(), nullable=False
    )


@event.listens_for(StudentAttendance, "before_insert")
def _validate_student_attendance_before_insert(_mapper, _connection, target: StudentAttendance) -> None:
    _validate_student_attendance_record(target)


@event.listens_for(StudentAttendance, "before_update")
def _validate_student_attendance_before_update(_mapper, _connection, target: StudentAttendance) -> None:
    _validate_student_attendance_record(target)


@event.listens_for(StudentSchedule, "before_insert")
def _validate_student_schedule_before_insert(_mapper, _connection, target: StudentSchedule) -> None:
    _validate_student_schedule_record(target)


@event.listens_for(StudentSchedule, "before_update")
def _validate_student_schedule_before_update(_mapper, _connection, target: StudentSchedule) -> None:
    _validate_student_schedule_record(target)


for _model in (
    UserRoleLink,
    Role,
    OrganizationalUnit,
    Faculty,
    AcademicYear,
    Department,
    ClassBatch,
    User,
    Student,
    Teacher,
    Course,
    CourseSemesterAssignment,
    ClassCourseAssignment,
    CourseAssignment,
    Enrollment,
    StudentAttendance,
    StudentSchedule,
    CourseSchedule,
    CourseScheduleWeekday,
    AttendanceSession,
    AttendanceRecord,
):
    event.listen(_model, "before_insert", _validate_positive_id_fields)
    event.listen(_model, "before_update", _validate_positive_id_fields)

event.listen(Faculty, "before_insert", _sync_faculty_duration)
event.listen(Faculty, "before_update", _sync_faculty_duration)
