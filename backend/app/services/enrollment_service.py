from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models import Course, Enrollment, Student


def auto_enroll_student_in_matching_courses(db: Session, student: Student) -> int:
    """Enroll a student into all courses that belong to their own department."""
    if student.id <= 0 or student.faculty_id <= 0 or student.department_id <= 0:
        return 0

    course_rows = (
        db.query(Course.id)
        .filter(
            Course.faculty_id == student.faculty_id,
            Course.department_id == student.department_id,
            Course.id > 0,
        )
        .all()
    )
    course_ids = [course_id for (course_id,) in course_rows]
    if not course_ids:
        return 0

    existing_rows = (
        db.query(Enrollment.course_id)
        .filter(
            Enrollment.student_id == student.id,
            Enrollment.course_id.in_(course_ids),
            Enrollment.student_id > 0,
            Enrollment.course_id > 0,
        )
        .all()
    )
    existing_course_ids = {course_id for (course_id,) in existing_rows}

    created = 0
    for course_id in course_ids:
        if course_id in existing_course_ids:
            continue
        db.add(Enrollment(student_id=student.id, course_id=course_id))
        created += 1
    return created


def auto_enroll_existing_students_for_course(db: Session, course: Course) -> int:
    """Enroll existing students from the same department into a new course."""
    if course.id <= 0 or course.faculty_id <= 0 or course.department_id <= 0:
        return 0

    student_rows = (
        db.query(Student.id)
        .filter(
            Student.faculty_id == course.faculty_id,
            Student.department_id == course.department_id,
            Student.id > 0,
        )
        .all()
    )
    student_ids = [student_id for (student_id,) in student_rows]
    if not student_ids:
        return 0

    existing_rows = (
        db.query(Enrollment.student_id)
        .filter(
            Enrollment.course_id == course.id,
            Enrollment.student_id.in_(student_ids),
            Enrollment.student_id > 0,
            Enrollment.course_id > 0,
        )
        .all()
    )
    existing_student_ids = {student_id for (student_id,) in existing_rows}

    created = 0
    for student_id in student_ids:
        if student_id in existing_student_ids:
            continue
        db.add(Enrollment(student_id=student_id, course_id=course.id))
        created += 1
    return created