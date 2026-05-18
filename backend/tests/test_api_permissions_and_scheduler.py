from __future__ import annotations

from datetime import date, datetime, time, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.security import get_current_user
from app.db.models import (
    AttendanceRecord,
    AttendanceStatus,
    AttendanceSession,
    Base,
    ClassBatch,
    Course,
    CourseSchedule,
    Department,
    Enrollment,
    Faculty,
    Role,
    SessionStatus,
    Student,
    Teacher,
    User,
)
from app.db.role_scoped import get_role_scoped_db
from app.db.session import get_db
from app.routers import classes, courses, departments, faculties, reports, schedules, sessions, students, teachers
from app.services.schedule_service import ScheduleService
from app.utils.datetime_utils import schedule_weekday_from_datetime
from app.utils import student_numbering
from app.utils.weekday_utils import weekday_code


class _Role:
    def __init__(self, name: str):
        self.name = name


class _DummyUser:
    def __init__(self, roles: list[str], faculty_id: int | None = None):
        self.roles = [_Role(role) for role in roles]
        self.faculty_id = faculty_id
        self.id = 1
        self.is_active = True


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite+pysqlite://",
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    db = TestingSession()
    try:
        # Seed role rows used by ORM relationships.
        for role_name in ("SUPER_ADMIN", "ACADEMIA", "FACULTY", "FACULTY_ADMIN", "HR", "ADMISSIONS", "TEACHER", "STUDENT"):
            db.add(Role(name=role_name))
        db.commit()
        yield db
    finally:
        db.close()


@pytest.fixture()
def client(db_session):
    app = FastAPI()
    app.include_router(faculties.router)
    app.include_router(departments.router)
    app.include_router(classes.router)
    app.include_router(courses.router)
    app.include_router(schedules.router)
    app.include_router(reports.router)
    app.include_router(students.router)
    app.include_router(teachers.router)
    app.include_router(sessions.router)

    def _override_db():
        try:
            yield db_session
        finally:
            pass

    current_roles = {"roles": ["ACADEMIA"], "faculty_id": 1, "user_id": 1}

    def _override_current_user() -> _DummyUser:
        user = _DummyUser(current_roles["roles"], current_roles.get("faculty_id"))
        user.id = current_roles.get("user_id", 1)
        return user

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[get_role_scoped_db] = _override_db
    app.dependency_overrides[get_current_user] = _override_current_user

    test_client = TestClient(app)
    return test_client, current_roles


def _seed_department(db_session, faculty: Faculty, *, name: str = "Department of Information Technology", code: str = "IT"):
    department = Department(faculty_id=faculty.id, name=name, code=code)
    db_session.add(department)
    db_session.flush()
    return department


def _seed_course_graph(db_session):
    faculty = Faculty(name="Faculty A", code="FA")
    db_session.add(faculty)
    db_session.flush()

    department = _seed_department(db_session, faculty, name="Department A", code="DA")

    class_batch = ClassBatch(faculty_id=faculty.id, department_id=department.id, name="CIS2201", year=2026)
    db_session.add(class_batch)
    db_session.flush()

    course = Course(faculty_id=faculty.id, department_id=department.id, code="CSC401", title="AI")
    db_session.add(course)
    db_session.commit()
    return faculty, department, class_batch, course


def test_permission_blocks_non_academia_faculty_create(client):
    api, current = client
    current["roles"] = ["TEACHER"]

    response = api.post("/faculties", json={"name": "Science", "code": "SCI", "years": 4})

    assert response.status_code == 403
    assert "Role required" in response.json()["detail"]


def test_permission_allows_academia_faculty_create(client):
    api, current = client
    current["roles"] = ["ACADEMIA"]

    response = api.post("/faculties", json={"name": "Science", "code": "SCI", "years": 4})

    assert response.status_code == 200
    assert response.json()["code"] == "SCI"
    assert response.json()["years"] == 4
    assert response.json()["semesters"] == 8


@pytest.mark.parametrize("roles", [["HR"], ["ADMIN"], ["ADMISSIONS"]])
def test_permission_allows_faculty_list_for_staff_roles(client, roles):
    api, current = client
    current["roles"] = roles

    response = api.get("/faculties")

    assert response.status_code == 200
    body = response.json()
    assert "items" in body
    assert "total" in body


@pytest.mark.parametrize("roles", [["HR"], ["ADMIN"], ["ADMISSIONS"]])
def test_permission_allows_department_list_for_staff_roles(client, roles):
    api, current = client
    current["roles"] = roles

    response = api.get("/departments")

    assert response.status_code == 200
    body = response.json()
    assert "items" in body
    assert "total" in body


@pytest.mark.parametrize("roles", [["HR"], ["ADMIN"], ["ADMISSIONS"]])
def test_permission_allows_course_list_for_staff_roles(client, roles):
    api, current = client
    current["roles"] = roles

    response = api.get("/courses")

    assert response.status_code == 200
    body = response.json()
    assert "items" in body
    assert "total" in body


def test_delete_faculty_without_force_returns_expected_response(client, db_session):
    faculty = Faculty(name="Faculty of Computer Science", code="FCS")
    db_session.add(faculty)
    db_session.flush()
    _seed_department(db_session, faculty, name="Department of IT", code="IT")
    db_session.add(User(username="faculty_blocked", hashed_password="x", is_active=True, faculty_id=faculty.id))
    db_session.commit()

    api, current = client
    current["roles"] = ["ACADEMIA"]

    response = api.delete(f"/faculties/{faculty.id}")

    assert response.status_code == 200
    body = response.json()
    assert body["deleted"] is True
    assert body["force"] is False


def test_preview_faculty_delete_shows_related_counts(client, db_session):
    faculty = Faculty(name="Faculty of Engineering", code="ENG")
    db_session.add(faculty)
    db_session.flush()

    department = _seed_department(db_session, faculty, name="Department of Architecture", code="ARCH")
    class_batch = ClassBatch(faculty_id=faculty.id, department_id=department.id, name="ENG2201", year=2026)
    db_session.add(class_batch)
    db_session.flush()

    db_session.add(Course(faculty_id=faculty.id, department_id=department.id, code="ENG001", title="Statics"))
    db_session.add(User(username="faculty_preview", hashed_password="x", is_active=True, faculty_id=faculty.id))
    db_session.commit()

    api, current = client
    current["roles"] = ["ACADEMIA"]

    response = api.get(f"/faculties/{faculty.id}/delete-preview")

    assert response.status_code == 200
    body = response.json()
    assert body["faculty_id"] == faculty.id
    assert body["force_required"] is True
    assert body["counts"]["departments"] == 1
    assert body["counts"]["class_batches"] == 1
    assert body["counts"]["courses"] == 1
    assert body["counts"]["users"] == 1


def test_delete_faculty_with_related_records_succeeds_with_force(client, db_session):
    faculty = Faculty(name="Faculty of Engineering", code="ENG")
    db_session.add(faculty)
    db_session.flush()

    department = _seed_department(db_session, faculty, name="Department of Architecture", code="ARCH")
    class_batch = ClassBatch(faculty_id=faculty.id, department_id=department.id, name="ENG2201", year=2026)
    db_session.add(class_batch)
    db_session.flush()

    db_session.add(Course(faculty_id=faculty.id, department_id=department.id, code="ENG001", title="Statics"))
    user = User(username="faculty_linked", hashed_password="x", is_active=True, faculty_id=faculty.id)
    db_session.add(user)
    db_session.commit()
    linked_user_id = user.id

    api, current = client
    current["roles"] = ["ACADEMIA"]

    response = api.delete(f"/faculties/{faculty.id}", params={"force": True})

    assert response.status_code == 200
    body = response.json()
    assert body["deleted"] is True
    assert body["force"] is True
    assert body["counts"]["faculties"] == 1
    assert body["counts"]["departments"] == 1
    assert body["counts"]["class_batches"] == 1
    assert body["counts"]["courses"] == 1
    assert body["counts"]["students"] == 0
    assert body["counts"]["teachers"] == 0
    assert body["counts"]["users"] == 1

    assert db_session.query(Faculty).filter(Faculty.id == faculty.id).first() is None
    assert db_session.query(Department).filter(Department.faculty_id == faculty.id).count() == 0
    assert db_session.query(User).filter(User.id == linked_user_id).first() is None


def test_create_department_under_faculty(client, db_session):
    faculty = Faculty(name="Faculty of Computer Science", code="FCS")
    db_session.add(faculty)
    db_session.commit()

    api, current = client
    current["roles"] = ["ACADEMIA"]
    current["faculty_id"] = faculty.id

    response = api.post(
        "/departments",
        json={
            "faculty_id": faculty.id,
            "name": "Department of Information Technology",
            "code": "IT",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["faculty_id"] == faculty.id
    assert body["code"] == "IT"


def test_create_department_allows_same_code_in_different_faculties(client, db_session):
    faculty_a = Faculty(name="Faculty A", code="FA")
    faculty_b = Faculty(name="Faculty B", code="FB")
    db_session.add_all([faculty_a, faculty_b])
    db_session.commit()

    api, current = client
    current["roles"] = ["ACADEMIA"]

    response_a = api.get(
        "/departments",
        params={"faculty_id": faculty_a.id},
    )
    response_b = api.get(
        "/departments",
        params={"faculty_id": faculty_b.id},
    )

    assert response_a.status_code == 200
    assert response_b.status_code == 200


def test_create_department_rejects_case_and_whitespace_duplicate(client, db_session):
    faculty = Faculty(name="Faculty of Computer Science", code="FCS")
    db_session.add(faculty)
    db_session.flush()
    _seed_department(db_session, faculty, name="Department of Information Technology", code="IT")
    db_session.commit()

    api, current = client
    current["roles"] = ["ACADEMIA"]
    current["faculty_id"] = faculty.id

    response = api.post(
        "/departments",
        json={
            "faculty_id": faculty.id,
            "name": "  department   of information technology  ",
            "code": "  it  ",
        },
    )

    assert response.status_code == 409
    assert "already exists" in response.json()["detail"].lower()


def test_create_department_requires_faculty_id_without_scope(client):
    api, current = client
    current["roles"] = ["ACADEMIA"]
    current["faculty_id"] = 1

    response = api.post(
        "/departments",
        json={"name": "Department of IT", "code": "IT"},
    )

    assert response.status_code == 400


def test_create_department_blocks_faculty_write(client, db_session):
    faculty = Faculty(name="Faculty of Science", code="SCI")
    db_session.add(faculty)
    db_session.commit()

    api, current = client
    current["roles"] = ["FACULTY"]

    response = api.post(
        "/departments",
        json={
            "faculty_id": faculty.id,
            "name": "Department of Statistics",
            "code": "STAT",
        },
    )

    assert response.status_code == 403


def test_create_course_auto_generates_code_from_faculty(client, db_session):
    faculty = Faculty(name="Faculty of Engineering", code="ENG")
    db_session.add(faculty)
    db_session.flush()

    department = _seed_department(db_session, faculty, name="Department of Architecture", code="ARCH")

    class_batch = ClassBatch(faculty_id=faculty.id, department_id=department.id, name="ENG2201", year=2026)
    db_session.add(class_batch)
    db_session.commit()

    api, current = client
    current["roles"] = ["ACADEMIA"]

    response = api.post(
        "/courses",
        json={
            "faculty_id": faculty.id,
            "title": "Thermodynamics",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == "ENG001"


def test_create_course_auto_generation_increments_sequence(client, db_session):
    faculty = Faculty(name="Faculty of Engineering", code="ENG")
    db_session.add(faculty)
    db_session.flush()

    department = _seed_department(db_session, faculty, name="Department of Architecture", code="ARCH")

    class_batch = ClassBatch(faculty_id=faculty.id, department_id=department.id, name="ENG2201", year=2026)
    db_session.add(class_batch)
    db_session.flush()

    db_session.add(Course(faculty_id=faculty.id, department_id=department.id, code="ENG001", title="Existing Course"))
    db_session.commit()

    api, current = client
    current["roles"] = ["ACADEMIA"]

    response = api.post(
        "/courses",
        json={
            "faculty_id": faculty.id,
            "title": "Fluid Mechanics",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == "ENG002"


def test_create_course_auto_enrolls_existing_students_in_same_faculty(client, db_session):
    faculty = Faculty(name="Faculty of Engineering", code="ENG")
    db_session.add(faculty)
    db_session.flush()

    department_arch = _seed_department(db_session, faculty, name="Department of Architecture", code="ARCH")
    department_civil = _seed_department(db_session, faculty, name="Department of Civil", code="CIV")

    batch_arch = ClassBatch(faculty_id=faculty.id, department_id=department_arch.id, name="ENG2201", year=2026)
    batch_civil = ClassBatch(faculty_id=faculty.id, department_id=department_civil.id, name="ENG2202", year=2026)
    db_session.add_all([batch_arch, batch_civil])
    db_session.flush()

    student_match = Student(
        student_number="26ENG001",
        full_name="Match Student",
        faculty_id=faculty.id,
        department_id=department_arch.id,
        embedding_ref="26ENG001",
    )
    student_other_department = Student(
        student_number="26ENG002",
        full_name="Other Department Student",
        faculty_id=faculty.id,
        department_id=department_civil.id,
        embedding_ref="26ENG002",
    )

    other_faculty = Faculty(name="Faculty of Computing", code="CMP")
    db_session.add(other_faculty)
    db_session.flush()
    other_department = _seed_department(db_session, other_faculty, name="Department of Systems", code="SYS")
    other_batch = ClassBatch(faculty_id=other_faculty.id, department_id=other_department.id, name="CMP2201", year=2026)
    db_session.add(other_batch)
    db_session.flush()
    student_other_faculty = Student(
        student_number="26CMP001",
        full_name="Other Faculty Student",
        faculty_id=other_faculty.id,
        department_id=other_department.id,
        embedding_ref="26CMP001",
    )

    db_session.add_all([student_match, student_other_department, student_other_faculty])
    db_session.commit()

    api, current = client
    current["roles"] = ["ACADEMIA"]

    response = api.post(
        "/courses",
        json={
            "faculty_id": faculty.id,
            "title": "Structural Analysis",
        },
    )

    assert response.status_code == 200
    course_id = response.json()["id"]

    enrollment_rows = db_session.query(Enrollment.student_id).filter(Enrollment.course_id == course_id).all()
    enrolled_student_ids = {student_id for (student_id,) in enrollment_rows}
    assert enrolled_student_ids == {student_match.id, student_other_department.id}


def test_create_course_uses_faculty_prefix_for_code_generation(client, db_session):
    faculty = Faculty(name="Faculty of Computer Science", code="FCS")
    db_session.add(faculty)
    db_session.flush()

    department = _seed_department(db_session, faculty, name="Department of Information Technology", code="IT")
    class_batch = ClassBatch(faculty_id=faculty.id, department_id=department.id, name="CIS2201", year=2026)
    db_session.add(class_batch)
    db_session.commit()

    api, current = client
    current["roles"] = ["ACADEMIA"]

    response = api.post(
        "/courses",
        json={
            "faculty_id": faculty.id,
            "department_id": department.id,
            "title": "Compiler Design",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["code"] == "FCS001"


def test_create_course_rejects_duplicate_normalized_title_within_same_faculty(client, db_session):
    faculty = Faculty(name="Faculty of Engineering", code="ENG")
    db_session.add(faculty)
    db_session.flush()

    architecture = _seed_department(db_session, faculty, name="Architecture", code="ARCH")
    civil = _seed_department(db_session, faculty, name="Civil", code="CIV")

    arch_batch = ClassBatch(faculty_id=faculty.id, department_id=architecture.id, name="ARCH2201", year=2026)
    civil_batch = ClassBatch(faculty_id=faculty.id, department_id=civil.id, name="CIV2201", year=2026)
    db_session.add_all([arch_batch, civil_batch])
    db_session.flush()

    db_session.add(Course(faculty_id=faculty.id, department_id=department.id, code="ENG001", title="Thermodynamics"))
    db_session.commit()

    api, current = client
    current["roles"] = ["ACADEMIA"]

    response = api.post(
        "/courses",
        json={
            "faculty_id": faculty.id,
            "title": "  thermodynamics  ",
        },
    )

    assert response.status_code == 409
    assert "title already exists" in response.json()["detail"].lower()


def test_create_course_allows_same_title_across_different_faculties(client, db_session):
    faculty_eng = Faculty(name="Faculty of Engineering", code="ENG")
    faculty_med = Faculty(name="Faculty of Medicine", code="MED")
    db_session.add_all([faculty_eng, faculty_med])
    db_session.flush()

    dep_eng = _seed_department(db_session, faculty_eng, name="Architecture", code="ARCH")
    dep_med = _seed_department(db_session, faculty_med, name="Medicine", code="MEDD")

    batch_eng = ClassBatch(faculty_id=faculty_eng.id, department_id=dep_eng.id, name="ENG2201", year=2026)
    batch_med = ClassBatch(faculty_id=faculty_med.id, department_id=dep_med.id, name="MED2201", year=2026)
    db_session.add_all([batch_eng, batch_med])
    db_session.flush()

    db_session.add(Course(faculty_id=faculty_eng.id, department_id=other_department.id, code="ENG001", title="Physics"))
    db_session.commit()

    api, current = client
    current["roles"] = ["ACADEMIA"]

    response = api.post(
        "/courses",
        json={
            "faculty_id": faculty_med.id,
            "title": "  physics ",
        },
    )

    assert response.status_code == 200


def test_report_range_rejects_inverted_dates(client, db_session):
    _, _, _, course = _seed_course_graph(db_session)
    api, current = client
    current["roles"] = ["TEACHER"]

    response = api.get(
        f"/reports/course/{course.id}/range",
        params={"start_date": "2026-02-10", "end_date": "2026-02-01"},
    )

    assert response.status_code == 400
    assert "end_date" in response.json()["detail"]


def _seed_mixed_attendance_for_reports(db_session):
    faculty, department, class_batch, course = _seed_course_graph(db_session)

    student = Student(
        student_number="26CIS007",
        full_name="Ahmed Abdirahman",
        faculty_id=faculty.id,
        department_id=department.id,
        embedding_ref="26CIS007",
    )
    db_session.add(student)
    db_session.flush()

    db_session.add(Enrollment(student_id=student.id, course_id=course.id))

    schedule = CourseSchedule(
        course_id=course.id,
        weekday=weekday_code(schedule_weekday_from_datetime(datetime.now())),
        start_time=time(8, 0),
        end_time=time(10, 0),
        grace_period_minutes=10,
    )
    db_session.add(schedule)
    db_session.flush()

    base_day = datetime(2026, 3, 19, 9, 0, 0)

    present_session = AttendanceSession(
        course_id=course.id,
        schedule_id=schedule.id,
        session_date=base_day.date(),
        start_time=base_day,
        end_time=base_day + timedelta(hours=1),
        status=SessionStatus.CLOSED,
    )
    late_session = AttendanceSession(
        course_id=course.id,
        schedule_id=schedule.id,
        session_date=(base_day + timedelta(days=1)).date(),
        start_time=base_day + timedelta(days=1),
        end_time=base_day + timedelta(days=1, hours=1),
        status=SessionStatus.CLOSED,
    )
    absent_session = AttendanceSession(
        course_id=course.id,
        schedule_id=schedule.id,
        session_date=(base_day + timedelta(days=2)).date(),
        start_time=base_day + timedelta(days=2),
        end_time=base_day + timedelta(days=2, hours=1),
        status=SessionStatus.CLOSED,
    )
    db_session.add_all([present_session, late_session, absent_session])
    db_session.flush()

    db_session.add_all(
        [
            AttendanceRecord(
                student_id=student.id,
                course_id=course.id,
                session_id=present_session.id,
                status=AttendanceStatus.PRESENT,
                confidence=0.93,
            ),
            AttendanceRecord(
                student_id=student.id,
                course_id=course.id,
                session_id=late_session.id,
                status=AttendanceStatus.LATE,
                confidence=0.88,
            ),
            AttendanceRecord(
                student_id=student.id,
                course_id=course.id,
                session_id=absent_session.id,
                status=AttendanceStatus.ABSENT,
                confidence=0.0,
            ),
        ]
    )
    db_session.commit()
    return course, student, present_session, late_session, absent_session


def test_course_report_counts_late_as_present(client, db_session):
    course, _, _, _, _ = _seed_mixed_attendance_for_reports(db_session)
    api, current = client
    current["roles"] = ["TEACHER"]

    response = api.get(f"/reports/course/{course.id}")

    assert response.status_code == 200
    body = response.json()
    assert body["total_records"] == 3
    assert body["present"] == 2
    assert body["late"] == 1
    assert body["absent"] == 1


def test_course_report_range_counts_late_as_present(client, db_session):
    course, _, present_session, _, absent_session = _seed_mixed_attendance_for_reports(db_session)
    api, current = client
    current["roles"] = ["TEACHER"]

    response = api.get(
        f"/reports/course/{course.id}/range",
        params={
            "start_date": present_session.session_date.isoformat(),
            "end_date": absent_session.session_date.isoformat(),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["total_records"] == 3
    assert body["present"] == 2
    assert body["late"] == 1
    assert body["absent"] == 1


def test_course_report_students_counts_late_as_present(client, db_session):
    course, student, present_session, _, absent_session = _seed_mixed_attendance_for_reports(db_session)
    api, current = client
    current["roles"] = ["TEACHER"]

    response = api.get(
        f"/reports/course/{course.id}/students",
        params={
            "start_date": present_session.session_date.isoformat(),
            "end_date": absent_session.session_date.isoformat(),
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert len(body["students"]) == 1
    row = body["students"][0]
    assert row["student_id"] == student.id
    assert row["present"] == 2
    assert row["late"] == 1
    assert row["absent"] == 1
    assert row["total"] == 3


def test_course_report_sessions_counts_late_as_present(client, db_session):
    course, _, present_session, late_session, absent_session = _seed_mixed_attendance_for_reports(db_session)
    api, current = client
    current["roles"] = ["TEACHER"]

    response = api.get(
        f"/reports/course/{course.id}/sessions",
        params={
            "start_date": present_session.session_date.isoformat(),
            "end_date": absent_session.session_date.isoformat(),
        },
    )

    assert response.status_code == 200
    body = response.json()

    by_session_id = {item["session_id"]: item for item in body["sessions"]}
    assert by_session_id[present_session.id]["present"] == 1
    assert by_session_id[present_session.id]["late"] == 0
    assert by_session_id[present_session.id]["absent"] == 0

    assert by_session_id[late_session.id]["present"] == 1
    assert by_session_id[late_session.id]["late"] == 1
    assert by_session_id[late_session.id]["absent"] == 0

    assert by_session_id[absent_session.id]["present"] == 0
    assert by_session_id[absent_session.id]["late"] == 0
    assert by_session_id[absent_session.id]["absent"] == 1


def test_assign_teacher_blocks_faculty_mismatch(client, db_session):
    faculty_a, _, class_batch, course = _seed_course_graph(db_session)

    faculty_b = Faculty(name="Faculty B", code="FB")
    db_session.add(faculty_b)
    db_session.flush()

    department_b = _seed_department(db_session, faculty_b, name="Department B", code="DB")

    user = User(username="teacher01", hashed_password="x", is_active=True, faculty_id=faculty_b.id)
    db_session.add(user)
    db_session.flush()

    teacher = Teacher(
        teacher_number="T01",
        full_name="Teacher One",
        faculty_id=faculty_b.id,
        department_id=department_b.id,
    )
    db_session.add(teacher)
    db_session.commit()

    api, current = client
    current["roles"] = ["FACULTY"]

    response = api.post(
        "/courses/assign-teacher",
        json={"course_id": course.id, "teacher_id": teacher.id, "is_primary": True},
    )

    assert response.status_code == 400


def test_assign_teacher_blocks_faculty_mismatch(client, db_session):
    faculty = Faculty(name="Faculty of Engineering", code="ENG")
    other_faculty = Faculty(name="Faculty of Computing", code="CMP")
    db_session.add(faculty)
    db_session.add(other_faculty)
    db_session.flush()

    architecture = _seed_department(db_session, faculty, name="Department of Architecture", code="ARCH")
    other_department = _seed_department(db_session, other_faculty, name="Department of Systems", code="SYS")

    class_batch = ClassBatch(faculty_id=faculty.id, department_id=architecture.id, name="ARCH2201", year=2026)
    db_session.add(class_batch)
    db_session.flush()

    course = Course(faculty_id=faculty.id, department_id=department.id, code="ENG001", title="Design Studio")
    db_session.add(course)
    db_session.flush()

    teacher = Teacher(
        teacher_number="ENGT001",
        full_name="Civil Teacher",
        faculty_id=other_faculty.id,
        department_id=other_department.id,
    )
    db_session.add(teacher)
    db_session.commit()

    api, current = client
    current["roles"] = ["FACULTY"]

    response = api.post(
        "/courses/assign-teacher",
        json={"course_id": course.id, "teacher_id": teacher.id, "is_primary": True},
    )

    assert response.status_code == 400


def test_schedule_overlapping_courses_rejected_within_same_faculty(client, db_session):
    _, _, _, course = _seed_course_graph(db_session)
    db_session.add(
        CourseSchedule(
            course_id=course.id,
            weekday="sat",
            start_time=time(9, 0),
            end_time=time(10, 0),
            grace_period_minutes=10,
        )
    )
    db_session.commit()

    api, current = client
    current["roles"] = ["FACULTY"]

    response = api.post(
        "/schedules",
        json={
            "course_id": course.id,
            "weekday": ["sat"],
            "start_time": "09:30:00",
            "end_time": "10:30:00",
            "grace_period_minutes": 10,
        },
    )

    assert response.status_code == 400
    assert (
        response.json()["detail"]
        == "This course is already scheduled for this faculty on all selected days."
    )


def test_schedule_same_course_same_day_rejected_even_with_different_time(client, db_session):
    _, _, _, course = _seed_course_graph(db_session)
    db_session.add(
        CourseSchedule(
            course_id=course.id,
            weekday="sat",
            start_time=time(9, 0),
            end_time=time(10, 0),
            grace_period_minutes=10,
        )
    )
    db_session.commit()

    api, current = client
    current["roles"] = ["FACULTY"]

    response = api.post(
        "/schedules",
        json={
            "course_id": course.id,
            "weekday": ["sat"],
            "start_time": "11:00:00",
            "end_time": "12:00:00",
            "grace_period_minutes": 10,
        },
    )

    assert response.status_code == 400
    assert (
        response.json()["detail"]
        == "This course is already scheduled for this faculty on all selected days."
    )


def test_schedule_update_rejects_duplicate_course_day_for_department(client, db_session):
    _, _, _, course = _seed_course_graph(db_session)
    db_session.add_all(
        [
            CourseSchedule(
                course_id=course.id,
                weekday="sat",
                start_time=time(8, 0),
                end_time=time(9, 0),
                grace_period_minutes=10,
            ),
            CourseSchedule(
                course_id=course.id,
                weekday=2,
                start_time=time(11, 0),
                end_time=time(12, 0),
                grace_period_minutes=10,
            ),
        ]
    )
    db_session.commit()

    api, current = client
    current["roles"] = ["FACULTY"]

    response = api.put(
        "/schedules/2",
        json={
            "weekday": ["sat"],
        },
    )

    assert response.status_code == 400
    assert (
        response.json()["detail"]
        == "This course is already scheduled for this faculty on all selected days."
    )


def test_schedule_create_reports_specific_conflicting_days(client, db_session):
    _, _, _, course = _seed_course_graph(db_session)
    db_session.add(
        CourseSchedule(
            course_id=course.id,
            weekday="mon,wed",
            start_time=time(9, 0),
            end_time=time(10, 0),
            grace_period_minutes=10,
        )
    )
    db_session.commit()

    api, current = client
    current["roles"] = ["FACULTY"]

    response = api.post(
        "/schedules",
        json={
            "course_id": course.id,
            "weekday": ["tue", "wed", "thu"],
            "start_time": "11:00:00",
            "end_time": "12:00:00",
            "grace_period_minutes": 10,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "This course is already scheduled for this faculty on: wed."


def test_schedule_create_reports_all_days_scheduled(client, db_session):
    _, _, _, course = _seed_course_graph(db_session)
    db_session.add(
        CourseSchedule(
            course_id=course.id,
            weekday="sat,sun,mon,tue,wed,thu,fri",
            start_time=time(9, 0),
            end_time=time(10, 0),
            grace_period_minutes=10,
        )
    )
    db_session.commit()

    api, current = client
    current["roles"] = ["FACULTY"]

    response = api.post(
        "/schedules",
        json={
            "course_id": course.id,
            "weekday": ["mon", "thu"],
            "start_time": "11:00:00",
            "end_time": "12:00:00",
            "grace_period_minutes": 10,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "This course is already scheduled for this faculty on all days."


def test_schedule_same_day_allowed_for_different_departments_in_same_faculty(client, db_session):
    faculty = Faculty(name="Faculty of Engineering", code="ENG")
    dep_a = Department(faculty_id=1, name="Architecture", code="ARCH")
    dep_b = Department(faculty_id=1, name="Civil", code="CIV")
    db_session.add(faculty)
    db_session.flush()
    dep_a.faculty_id = faculty.id
    dep_b.faculty_id = faculty.id
    db_session.add_all([dep_a, dep_b])
    db_session.flush()

    batch_a = ClassBatch(faculty_id=faculty.id, department_id=dep_a.id, name="ARCH2201", year=2026)
    batch_b = ClassBatch(faculty_id=faculty.id, department_id=dep_b.id, name="CIV2201", year=2026)
    db_session.add_all([batch_a, batch_b])
    db_session.flush()

    course_a = Course(faculty_id=faculty.id, department_id=department.id, code="ENG001", title="Statics")
    course_b = Course(faculty_id=faculty.id, department_id=department.id, code="ENG002", title="Dynamics")
    db_session.add_all([course_a, course_b])
    db_session.flush()

    db_session.add(
        CourseSchedule(
            course_id=course_a.id,
            weekday="mon",
            start_time=time(8, 0),
            end_time=time(9, 0),
            grace_period_minutes=10,
        )
    )
    db_session.commit()

    api, current = client
    current["roles"] = ["FACULTY"]

    response = api.post(
        "/schedules",
        json={
            "course_id": course_b.id,
            "weekday": ["mon"],
            "start_time": "11:00:00",
            "end_time": "12:00:00",
            "grace_period_minutes": 10,
        },
    )

    assert response.status_code == 200


def test_scheduler_day_rollover_uses_current_weekday_only(db_session, monkeypatch):
    faculty, department, class_batch, course = _seed_course_graph(db_session)

    previous_day_course = Course(faculty_id=faculty.id, department_id=department.id, code="CSC402", title="Prev Day")
    db_session.add(previous_day_course)
    db_session.flush()

    fake_now = datetime(2026, 3, 9, 0, 5, 0)  # Monday
    current_weekday = schedule_weekday_from_datetime(fake_now)
    previous_weekday = "sun"

    db_session.add(
        CourseSchedule(
            course_id=course.id,
            weekday=weekday_code(current_weekday),
            start_time=time(0, 0),
            end_time=time(1, 0),
            grace_period_minutes=5,
        )
    )
    db_session.add(
        CourseSchedule(
            course_id=previous_day_course.id,
            weekday=previous_weekday,
            start_time=time(0, 0),
            end_time=time(1, 0),
            grace_period_minutes=5,
        )
    )
    db_session.commit()

    svc = ScheduleService()
    svc._tick(db_session)

    sessions_created = db_session.query(AttendanceSession).filter(AttendanceSession.status == SessionStatus.ACTIVE).all()
    assert sessions_created == []


def test_list_enrolled_students_by_course(client, db_session):
    faculty, department, class_batch, course = _seed_course_graph(db_session)

    student_a = Student(
        student_number="2201991",
        full_name="Student A",
        faculty_id=faculty.id,
        department_id=department.id,
        embedding_ref="2201991",
    )
    student_b = Student(
        student_number="2201992",
        full_name="Student B",
        faculty_id=faculty.id,
        department_id=department.id,
        embedding_ref="2201992",
    )
    db_session.add_all([student_a, student_b])
    db_session.flush()

    db_session.add(Enrollment(student_id=student_a.id, course_id=course.id))
    db_session.add(Enrollment(student_id=student_b.id, course_id=course.id))
    db_session.commit()

    api, current = client
    current["roles"] = ["FACULTY"]

    response = api.get(f"/courses/{course.id}/students")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert [item["student_number"] for item in body] == ["2201991", "2201992"]


def test_enroll_student_rejects_overlapping_course_sessions(client, db_session):
    faculty, department, class_batch, course_a = _seed_course_graph(db_session)
    course_b = Course(faculty_id=faculty.id, department_id=department.id, code="CSC402", title="Networks")
    db_session.add(course_b)
    db_session.flush()

    student = Student(
        student_number="2201888",
        full_name="Conflict Student",
        faculty_id=faculty.id,
        department_id=department.id,
        embedding_ref="2201888",
    )
    db_session.add(student)
    db_session.flush()

    db_session.add(
        CourseSchedule(
            course_id=course_a.id,
            weekday="sat",
            start_time=time(9, 0),
            end_time=time(10, 0),
            grace_period_minutes=10,
        )
    )
    db_session.add(
        CourseSchedule(
            course_id=course_b.id,
            weekday="sat",
            start_time=time(9, 30),
            end_time=time(10, 30),
            grace_period_minutes=10,
        )
    )
    db_session.add(Enrollment(student_id=student.id, course_id=course_a.id))
    db_session.commit()

    api, current = client
    current["roles"] = ["FACULTY"]

    response = api.post(f"/courses/{course_b.id}/enroll/{student.id}")

    assert response.status_code == 400
    assert "same time" in response.json()["detail"].lower()


def test_scheduler_backfills_missed_session_and_marks_absent(db_session, monkeypatch):
    faculty, department, class_batch, course = _seed_course_graph(db_session)

    student = Student(
        student_number="2201999",
        full_name="Missed Student",
        faculty_id=faculty.id,
        department_id=department.id,
        embedding_ref="2201999",
    )
    db_session.add(student)
    db_session.flush()
    db_session.add(Enrollment(student_id=student.id, course_id=course.id))

    # Session window is 08:00-10:00, but current time is already 12:30.
    db_session.add(
        CourseSchedule(
            course_id=course.id,
            weekday="sat",
            start_time=time(8, 0),
            end_time=time(10, 0),
            grace_period_minutes=10,
        )
    )
    db_session.commit()

    svc = ScheduleService()
    svc._tick(db_session)

    session = db_session.query(AttendanceSession).filter(AttendanceSession.course_id == course.id).first()
    assert session is None


def test_teacher_can_start_and_end_session(client, db_session):
    faculty, department, _, course = _seed_course_graph(db_session)
    schedule = CourseSchedule(
        course_id=course.id,
        weekday="mon",
        start_time=time(9, 0),
        end_time=time(10, 0),
        grace_period_minutes=10,
    )
    db_session.add(schedule)

    user = User(username="teacher01", hashed_password="x", is_active=True, faculty_id=faculty.id)
    db_session.add(user)
    db_session.flush()

    teacher = Teacher(
        teacher_number="T001",
        full_name="Teacher One",
        faculty_id=faculty.id,
        department_id=department.id,
        user_id=user.id,
    )
    db_session.add(teacher)
    db_session.commit()

    api, current = client
    current["roles"] = ["TEACHER"]
    current["user_id"] = user.id

    start_response = api.post("/sessions/start", json={"course_id": course.id, "schedule_id": schedule.id})
    assert start_response.status_code == 200
    started = start_response.json()
    assert started["course_id"] == course.id
    assert started["instructor_id"] == user.id
    assert started["status"] == "ACTIVE"
    assert started["end_time"] is None

    duplicate_response = api.post("/sessions/start", json={"course_id": course.id, "schedule_id": schedule.id})
    assert duplicate_response.status_code == 200
    assert duplicate_response.json()["id"] == started["id"]

    end_response = api.post("/sessions/end", json={"session_id": started["id"]})
    assert end_response.status_code == 200
    ended = end_response.json()
    assert ended["status"] == "CLOSED"
    assert ended["end_time"] is not None


def test_create_student_auto_generates_number(client, db_session, tmp_path, monkeypatch):
    monkeypatch.setattr(student_numbering, "_DATASET_ROOT", tmp_path)
    faculty = Faculty(name="Faculty of Computer Science", code="CIS")
    db_session.add(faculty)
    db_session.flush()

    department = _seed_department(db_session, faculty)
    db_session.commit()

    api, current = client
    current["roles"] = ["ADMISSIONS"]

    response = api.post(
        "/students",
        json={
            "full_name": "Student Auto",
            "faculty_id": faculty.id,
            "department_id": department.id,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["student_number"] == "26CIS001"
    assert body["embedding_ref"] == "26CIS001"
    assert body["department_id"] == department.id


def test_create_student_auto_enrolls_courses_in_same_faculty(client, db_session, tmp_path, monkeypatch):
    monkeypatch.setattr(student_numbering, "_DATASET_ROOT", tmp_path)

    faculty = Faculty(name="Faculty of Computer Science", code="CIS")
    db_session.add(faculty)
    db_session.flush()

    department_it = _seed_department(db_session, faculty, name="Department of IT", code="IT")
    department_cs = _seed_department(db_session, faculty, name="Department of CS", code="CS")

    batch_it = ClassBatch(faculty_id=faculty.id, department_id=department_it.id, name="CIS2201", year=2026)
    batch_cs = ClassBatch(faculty_id=faculty.id, department_id=department_cs.id, name="CIS2202", year=2026)
    db_session.add_all([batch_it, batch_cs])
    db_session.flush()

    course_a = Course(faculty_id=faculty.id, department_id=department.id, code="CIS001", title="Algorithms")
    course_b = Course(faculty_id=faculty.id, department_id=department.id, code="CIS002", title="Databases")
    course_other_department = Course(faculty_id=faculty.id, department_id=other_department.id, code="CIS003", title="Compilers")
    db_session.add_all([course_a, course_b, course_other_department])

    other_faculty = Faculty(name="Faculty of Engineering", code="ENG")
    db_session.add(other_faculty)
    db_session.flush()
    other_department = _seed_department(db_session, other_faculty, name="Department of Civil", code="CIV")
    other_batch = ClassBatch(faculty_id=other_faculty.id, department_id=other_department.id, name="ENG2201", year=2026)
    db_session.add(other_batch)
    db_session.flush()
    db_session.add(Course(faculty_id=other_faculty.id, department_id=other_department.id, code="ENG001", title="Statics"))
    db_session.commit()

    api, current = client
    current["roles"] = ["ADMISSIONS"]

    response = api.post(
        "/students",
        json={
            "full_name": "Auto Enroll Student",
            "faculty_id": faculty.id,
            "department_id": department_it.id,
        },
    )

    assert response.status_code == 200
    student_id = response.json()["id"]

    enrollment_rows = db_session.query(Enrollment.course_id).filter(Enrollment.student_id == student_id).all()
    enrolled_course_ids = {course_id for (course_id,) in enrollment_rows}
    assert enrolled_course_ids == {course_a.id, course_b.id, course_other_department.id}


def test_create_student_auto_generation_increments_sequence(client, db_session, tmp_path, monkeypatch):
    monkeypatch.setattr(student_numbering, "_DATASET_ROOT", tmp_path)
    faculty = Faculty(name="Engineering", code="ENG")
    db_session.add(faculty)
    db_session.flush()

    department = _seed_department(db_session, faculty, name="Department of Architecture", code="ARCH")

    class_batch = ClassBatch(faculty_id=faculty.id, department_id=department.id, name="ENG2201", year=2026)
    db_session.add(class_batch)
    db_session.flush()

    db_session.add(
        Student(
            student_number="26ENG001",
            full_name="Existing Student",
            faculty_id=faculty.id,
            department_id=department.id,
            embedding_ref="26ENG001",
        )
    )
    db_session.commit()

    api, current = client
    current["roles"] = ["ADMISSIONS"]

    response = api.post(
        "/students",
        json={
            "full_name": "Next Student",
            "faculty_id": faculty.id,
            "department_id": department.id,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["student_number"] == "26ENG002"


def test_next_student_number_skips_dataset_taken_ids(db_session, tmp_path, monkeypatch):
    monkeypatch.setattr(student_numbering, "_DATASET_ROOT", tmp_path)
    dataset_dir = tmp_path / "CIS"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "26CIS005").mkdir()
    (dataset_dir / "26CIS006").mkdir()

    value = student_numbering.next_available_student_number(db_session, "FCS", 2026, "CIS2201")

    assert value == "26CIS007"


def test_normalize_legacy_student_numbers_renames_dataset_and_db(db_session, tmp_path, monkeypatch):
    monkeypatch.setattr(student_numbering, "_DATASET_ROOT", tmp_path)

    faculty = Faculty(name="Faculty of Computer Science", code="CIS")
    db_session.add(faculty)
    db_session.flush()

    department = _seed_department(db_session, faculty)

    class_batch = ClassBatch(faculty_id=faculty.id, department_id=department.id, name="CIS2201", year=2026)
    db_session.add(class_batch)
    db_session.flush()

    student = Student(
        student_number="CIS001",
        full_name="Legacy Student",
        faculty_id=faculty.id,
        department_id=department.id,
        embedding_ref="CIS001",
    )
    db_session.add(student)
    db_session.commit()

    old_dir = tmp_path / "CIS" / "CIS001"
    old_dir.mkdir(parents=True, exist_ok=True)
    (old_dir / "img_001.jpg").write_bytes(b"x")

    renamed = student_numbering.normalize_legacy_student_numbers(db_session)

    assert renamed == [("CIS001", "26CIS001")]
    refreshed = db_session.query(Student).filter(Student.id == student.id).first()
    assert refreshed is not None
    assert refreshed.student_number == "26CIS001"
    assert refreshed.embedding_ref == "26CIS001"
    assert not old_dir.exists()
    assert (tmp_path / "CIS" / "26CIS001").exists()


def test_normalize_legacy_student_numbers_keeps_existing_year_prefixed_values(db_session, tmp_path, monkeypatch):
    monkeypatch.setattr(student_numbering, "_DATASET_ROOT", tmp_path)

    faculty = Faculty(name="Faculty of Computer Science", code="FCS")
    db_session.add(faculty)
    db_session.flush()

    department = _seed_department(db_session, faculty)

    class_batch = ClassBatch(faculty_id=faculty.id, department_id=department.id, name="CIS2201", year=2026)
    db_session.add(class_batch)
    db_session.flush()

    student = Student(
        student_number="26CIS001",
        full_name="Already Normalized",
        faculty_id=faculty.id,
        department_id=department.id,
        embedding_ref="26CIS001",
    )
    db_session.add(student)
    db_session.commit()

    renamed = student_numbering.normalize_legacy_student_numbers(db_session)

    assert renamed == []


def test_normalize_legacy_student_numbers_repairs_malformed_year_prefixed_value(db_session, tmp_path, monkeypatch):
    monkeypatch.setattr(student_numbering, "_DATASET_ROOT", tmp_path)

    faculty = Faculty(name="Faculty of Computer Science", code="FCS")
    db_session.add(faculty)
    db_session.flush()

    department = _seed_department(db_session, faculty)

    class_batch = ClassBatch(faculty_id=faculty.id, department_id=department.id, name="CIS2201", year=2026)
    db_session.add(class_batch)
    db_session.flush()

    student = Student(
        student_number="26CIS2201001",
        full_name="Malformed Student",
        faculty_id=faculty.id,
        department_id=department.id,
        embedding_ref="26CIS2201001",
    )
    db_session.add(student)
    db_session.commit()

    renamed = student_numbering.normalize_legacy_student_numbers(db_session)

    assert renamed == [("26CIS2201001", "26CIS001")]


def test_create_teacher_auto_generates_number(client, db_session):
    faculty = Faculty(name="Faculty of Engineering", code="ENG")
    db_session.add(faculty)
    db_session.flush()

    department = _seed_department(db_session, faculty, name="Department of Architecture", code="ARCH")
    db_session.commit()

    api, current = client
    current["roles"] = ["HR"]

    response = api.post(
        "/teachers",
        json={
            "full_name": "Engineer Teacher",
            "role": "Professor",
            "status": "Onleave",
            "faculty_id": faculty.id,
            "department_id": department.id,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["teacher_number"] == "ENGT001"
    assert body["role"] == "Professor"
    assert body["status"] == "Onleave"
    assert body["department_id"] == department.id


def test_create_teacher_auto_generation_increments_sequence(client, db_session):
    faculty = Faculty(name="Faculty of Engineering", code="ENG")
    db_session.add(faculty)
    db_session.flush()
    department = _seed_department(db_session, faculty, name="Department of Architecture", code="ARCH")
    db_session.add(
        Teacher(
            teacher_number="ENGT001",
            full_name="Existing Teacher",
            faculty_id=faculty.id,
            department_id=department.id,
        )
    )
    db_session.commit()

    api, current = client
    current["roles"] = ["HR"]

    response = api.post(
        "/teachers",
        json={
            "full_name": "Next Teacher",
            "faculty_id": faculty.id,
            "department_id": department.id,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["teacher_number"] == "ENGT002"
    assert body["role"] == "Lecturer"
    assert body["status"] == "Active"


def test_create_class_batch_rejects_department_faculty_mismatch(client, db_session):
    faculty_a = Faculty(name="Faculty A", code="FA")
    faculty_b = Faculty(name="Faculty B", code="FB")
    db_session.add_all([faculty_a, faculty_b])
    db_session.flush()

    department_b = _seed_department(db_session, faculty_b, name="Department B", code="DB")
    db_session.commit()

    api, current = client
    current["roles"] = ["ACADEMIA"]

    response = api.post(
        "/classes",
        json={
            "faculty_id": faculty_a.id,
            "department_id": department_b.id,
            "name": "MIX2201",
            "year": 2026,
        },
    )

    assert response.status_code == 400
    assert "department does not belong to faculty" in response.json()["detail"].lower()


def test_create_class_batch_allows_same_name_across_departments(client, db_session):
    faculty = Faculty(name="Faculty of Engineering", code="ENG")
    db_session.add(faculty)
    db_session.flush()

    department_a = _seed_department(db_session, faculty, name="Department A", code="DA")
    department_b = _seed_department(db_session, faculty, name="Department B", code="DB")
    db_session.add(ClassBatch(faculty_id=faculty.id, department_id=department_a.id, name="ENG2201", year=2026))
    db_session.commit()

    api, current = client
    current["roles"] = ["ACADEMIA"]

    response = api.post(
        "/classes",
        json={
            "faculty_id": faculty.id,
            "department_id": department_b.id,
            "name": "ENG2201",
            "year": 2026,
        },
    )

    assert response.status_code == 200
    assert response.json()["department_id"] == department_b.id


def test_create_class_batch_ignores_manual_name_and_generates_next_code(client, db_session):
    faculty = Faculty(name="Faculty of Engineering", code="ENG")
    db_session.add(faculty)
    db_session.flush()

    department = _seed_department(db_session, faculty, name="Department of Architecture", code="ARCH")
    db_session.add(ClassBatch(faculty_id=faculty.id, department_id=department.id, name="ENG2201", year=2026))
    db_session.commit()

    api, current = client
    current["roles"] = ["ACADEMIA"]

    response = api.post(
        "/classes",
        json={
            "faculty_id": faculty.id,
            "department_id": department.id,
            "name": "  eng2201  ",
            "year": 2026,
        },
    )

    assert response.status_code == 200
    assert response.json()["name"] == "ENG2202"


def test_create_class_batch_requires_faculty_id_without_scope(client, db_session):
    faculty = Faculty(name="Faculty A", code="FA")
    db_session.add(faculty)
    db_session.flush()
    department = _seed_department(db_session, faculty, name="Department A", code="DA")
    db_session.commit()

    api, current = client
    current["roles"] = ["ACADEMIA"]

    response = api.post(
        "/classes",
        json={"department_id": department.id, "year": 2026},
    )

    assert response.status_code == 422


def test_create_class_batch_auto_generates_name_and_increments_sequence(client, db_session):
    faculty = Faculty(name="Faculty of Computer Science", code="FCS")
    db_session.add(faculty)
    db_session.flush()

    department = _seed_department(db_session, faculty, name="Department of Information Technology", code="CIS")
    db_session.add(ClassBatch(faculty_id=faculty.id, department_id=department.id, name="CIS001", year=2026))
    db_session.commit()

    api, current = client
    current["roles"] = ["ACADEMIA"]

    response = api.post(
        "/classes",
        json={
            "faculty_id": faculty.id,
            "department_id": department.id,
            "year": 2027,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["name"] == "CIS002"
