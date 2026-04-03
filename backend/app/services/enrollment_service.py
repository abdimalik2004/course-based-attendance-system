from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models import ClassBatch, Course, Enrollment, Student


def auto_enroll_student_in_matching_courses(db: Session, student: Student) -> int:
    """Enroll a student into all courses in the same faculty and department."""
    if student.id <= 0 or student.faculty_id <= 0 or student.department_id <= 0:
        return 0

    course_rows = (
        db.query(Course.id)
        .join(ClassBatch, ClassBatch.id == Course.class_batch_id)
        .filter(
            Course.faculty_id == student.faculty_id,
            ClassBatch.department_id == student.department_id,
            Course.id > 0,
            Course.class_batch_id > 0,
            Course.faculty_id > 0,
            ClassBatch.id > 0,
            ClassBatch.department_id > 0,
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
    """Enroll existing students in matching faculty and department into a new course."""
    if course.id <= 0 or course.class_batch_id <= 0 or course.faculty_id <= 0:
        return 0

    class_batch = db.query(ClassBatch).filter(ClassBatch.id == course.class_batch_id).first()
    if class_batch is None:
        return 0
    if class_batch.id <= 0 or class_batch.department_id <= 0:
        return 0

    student_rows = (
        db.query(Student.id)
        .filter(
            Student.faculty_id == course.faculty_id,
            Student.department_id == class_batch.department_id,
            Student.id > 0,
            Student.faculty_id > 0,
            Student.department_id > 0,
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