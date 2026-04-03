from __future__ import annotations

from datetime import time

from app.core.security import get_password_hash
from app.db.models import (
    ClassBatch,
    Course,
    CourseAssignment,
    CourseSchedule,
    Department,
    Enrollment,
    Faculty,
    OrganizationalUnit,
    Role,
    Student,
    Teacher,
    User,
)
from app.db.session import SessionLocal


def _get_or_create_role(db, name: str) -> Role:
    role = db.query(Role).filter(Role.name == name).first()
    if role:
        return role
    role = Role(name=name)
    db.add(role)
    db.flush()
    return role


def _get_or_create_org_unit(db, code: str, name: str) -> OrganizationalUnit:
    unit = db.query(OrganizationalUnit).filter(OrganizationalUnit.code == code).first()
    if unit:
        unit.name = name
        db.add(unit)
        db.flush()
        return unit

    unit = OrganizationalUnit(code=code, name=name)
    db.add(unit)
    db.flush()
    return unit


def _get_or_create_user(db, username: str, password: str, role_names: list[str], faculty_id: int | None = None) -> User:
    user = db.query(User).filter(User.username == username).first()
    roles = db.query(Role).filter(Role.name.in_(role_names)).all()
    if user:
        # Keep seeded credentials in sync with the current hash scheme.
        user.hashed_password = get_password_hash(password)
        user.faculty_id = faculty_id
        user.is_active = True
        user.roles = roles
        db.add(user)
        db.flush()
        return user

    user = User(
        username=username,
        hashed_password=get_password_hash(password),
        is_active=True,
        faculty_id=faculty_id,
    )
    user.roles = roles
    db.add(user)
    db.flush()
    return user


def _get_or_create_department(db, faculty_id: int, name: str, code: str) -> Department:
    department = (
        db.query(Department)
        .filter(Department.faculty_id == faculty_id, Department.code == code)
        .first()
    )
    if department:
        department.name = name
        db.add(department)
        db.flush()
        return department

    department = Department(faculty_id=faculty_id, name=name, code=code)
    db.add(department)
    db.flush()
    return department


def _get_or_create_faculty(db, name: str, code: str) -> Faculty:
    by_code = db.query(Faculty).filter(Faculty.code == code).first()
    if by_code:
        if by_code.name != name:
            by_code.name = name
            db.add(by_code)
            db.flush()
        return by_code

    by_name = db.query(Faculty).filter(Faculty.name == name).first()
    if by_name:
        if by_name.code != code:
            code_taken = db.query(Faculty.id).filter(Faculty.code == code, Faculty.id != by_name.id).first()
            if code_taken is None:
                by_name.code = code
                db.add(by_name)
                db.flush()
        return by_name

    faculty = Faculty(name=name, code=code)
    db.add(faculty)
    db.flush()
    return faculty


def seed_demo_data() -> None:
    db = SessionLocal()
    try:
        for code, name in [
            ("ACADEMIA", "Academia"),
            ("FACULTIES", "Faculties"),
            ("HR", "HR"),
            ("ADMISSIONS", "Admissions"),
        ]:
            _get_or_create_org_unit(db, code, name)

        for role_name in ["ACADEMIA", "FACULTY", "HR", "ADMISSIONS", "TEACHER"]:
            _get_or_create_role(db, role_name)
        db.flush()

        faculty = _get_or_create_faculty(db, "Faculty of Computer Science", "FCS")

        department = _get_or_create_department(db, faculty.id, "Department of Information Technology", "IT")

        engineering = _get_or_create_faculty(db, "Faculty of Engineering", "ENG")
        _get_or_create_department(db, engineering.id, "Department of Architecture", "ARCH")

        class_batch = (
            db.query(ClassBatch)
            .filter(ClassBatch.faculty_id == faculty.id, ClassBatch.department_id == department.id, ClassBatch.name == "CIS2201")
            .first()
        )
        if not class_batch:
            class_batch = ClassBatch(
                faculty_id=faculty.id,
                department_id=department.id,
                name="CIS2201",
                year=2026,
            )
            db.add(class_batch)
            db.flush()

        academia_user = _get_or_create_user(db, "academia", "academia123", ["ACADEMIA"])
        faculty_admin_user = _get_or_create_user(
            db,
            "facultyadmin",
            "faculty123",
            ["FACULTY"],
            faculty_id=faculty.id,
        )

        teacher_user = _get_or_create_user(
            db,
            "teacher1",
            "teacher123",
            ["TEACHER"],
            faculty_id=faculty.id,
        )

        hr_user = _get_or_create_user(
            db,
            "hr",
            "hr123",
            ["HR"],
        )

        admissions_user = _get_or_create_user(
            db,
            "admission",
            "admission123",
            ["ADMISSIONS"],
        )

        teacher = db.query(Teacher).filter(Teacher.teacher_number == "T-1001").first()
        if not teacher:
            teacher = Teacher(
                teacher_number="T-1001",
                full_name="Dr. Sarah Ahmed",
                faculty_id=faculty.id,
                department_id=department.id,
                user_id=teacher_user.id,
            )
            db.add(teacher)
            db.flush()

        course = (
            db.query(Course)
            .filter(Course.class_batch_id == class_batch.id, Course.code == "CSC401")
            .first()
        )
        if not course:
            course = Course(
                class_batch_id=class_batch.id,
                code="CSC401",
                title="Artificial Intelligence",
            )
            db.add(course)
            db.flush()

        assignment = (
            db.query(CourseAssignment)
            .filter(CourseAssignment.course_id == course.id, CourseAssignment.teacher_id == teacher.id)
            .first()
        )
        if not assignment:
            db.add(CourseAssignment(course_id=course.id, teacher_id=teacher.id, is_primary=True))

        schedule = (
            db.query(CourseSchedule)
            .filter(
                CourseSchedule.course_id == course.id,
                CourseSchedule.weekday == "mon",
                CourseSchedule.start_time == time(hour=8, minute=0),
            )
            .first()
        )
        if not schedule:
            db.add(
                CourseSchedule(
                    course_id=course.id,
                    weekday="mon",
                    start_time=time(hour=8, minute=0),
                    end_time=time(hour=10, minute=0),
                    grace_period_minutes=10,
                )
            )

        students_data = [
            ("2201001", "Abdimalik Hassan"),
            ("2201002", "Aisha Noor"),
            ("2201003", "Mohamed Ali"),
        ]
        student_rows: list[Student] = []
        for student_number, full_name in students_data:
            student = db.query(Student).filter(Student.student_number == student_number).first()
            if not student:
                student = Student(
                    student_number=student_number,
                    full_name=full_name,
                    faculty_id=faculty.id,
                    department_id=department.id,
                    class_batch_id=class_batch.id,
                    embedding_ref=student_number,
                )
                db.add(student)
                db.flush()
            student_rows.append(student)

        for student in student_rows:
            enr = (
                db.query(Enrollment)
                .filter(Enrollment.student_id == student.id, Enrollment.course_id == course.id)
                .first()
            )
            if not enr:
                db.add(Enrollment(student_id=student.id, course_id=course.id))

        db.commit()
        print("Seed completed.")
        print(
            "Users: academia/academia123, facultyadmin/faculty123, "
            "teacher1/teacher123, hr/hr123, admission/admission123"
        )
        print(f"Faculty: {faculty.name} | Class: {class_batch.name} | Course: {course.code}")
        _ = academia_user, faculty_admin_user, hr_user, admissions_user
    finally:
        db.close()


if __name__ == "__main__":
    seed_demo_data()
