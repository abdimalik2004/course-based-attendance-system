from __future__ import annotations

from datetime import date, datetime, time

from sqlalchemy import func, or_
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.security import get_current_user, require_roles
from app.db.faculty_scope import enforce_faculty_scope, get_optional_faculty_scope_context
from app.db.models import (
    AttendanceRecord,
    AttendanceSession,
    ClassBatch,
    AttendanceStatus,
    Course,
    Student,
    StudentAdmissionStatus,
    Teacher,
    Faculty,
    Department,
    User,
)
from app.db.role_scoped import get_role_scoped_db
from app.utils.activity_logger import log_activity

router = APIRouter(prefix="/reports", tags=["reports"])

report_access_dependency = Depends(require_roles("SUPER_ADMIN", "ADMIN", "ACADEMIA", "FACULTY", "TEACHER", "HR", "ADMISSIONS"))


@router.post(
    "/teacher-report",
    status_code=204,
    dependencies=[Depends(require_roles("HR"))],
    summary="Log HR teacher report generation",
)
def log_teacher_report(
    db: Session = Depends(get_role_scoped_db),
    current_user: User = Depends(get_current_user),
):
    """Called by the HR frontend when a teacher report is generated. Records the action in the activity log."""
    log_activity(action="HR Teacher Report Generated", user=current_user, db=db)


def _is_present_status(status: AttendanceStatus) -> bool:
    return status in (AttendanceStatus.PRESENT, AttendanceStatus.LATE)


def _course_faculty_id(db: Session, course_id: int) -> int:
    course = db.query(Course).filter(Course.id == course_id).first()
    if not course:
        raise HTTPException(status_code=404, detail="Course not found")
    return course.faculty_id


@router.get("/summary", dependencies=[report_access_dependency])
def report_summary(
    db: Session = Depends(get_role_scoped_db),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    student_query = db.query(Student)
    teacher_query = db.query(Teacher)
    department_query = db.query(Department)
    class_query = db.query(ClassBatch)
    course_query = db.query(Course)
    attendance_query = db.query(AttendanceRecord)

    if faculty_scope is not None:
        student_query = student_query.filter(Student.faculty_id == faculty_scope.faculty_id)
        teacher_query = teacher_query.filter(Teacher.faculty_id == faculty_scope.faculty_id)
        department_query = department_query.filter(Department.faculty_id == faculty_scope.faculty_id)
        class_query = class_query.filter(ClassBatch.faculty_id == faculty_scope.faculty_id)
        course_query = course_query.filter(Course.faculty_id == faculty_scope.faculty_id)
        attendance_query = attendance_query.join(Course, Course.id == AttendanceRecord.course_id).filter(
            Course.faculty_id == faculty_scope.faculty_id
        )

    total_students = student_query.count()
    total_teachers = teacher_query.count()
    total_departments = department_query.count()
    total_classes = class_query.count()
    total_courses = course_query.count()
    total_faculties = db.query(func.count(Faculty.id)).scalar() or 0
    attendance_total = attendance_query.count()
    attendance_present = attendance_query.filter(
        AttendanceRecord.status.in_([AttendanceStatus.PRESENT, AttendanceStatus.LATE])
    ).count()
    attendance_rate = round((attendance_present / attendance_total) * 100, 1) if attendance_total else 0.0

    return {
        "totalStudents": total_students,
        "totalTeachers": total_teachers,
        "totalDepartments": total_departments,
        "totalClasses": total_classes,
        "totalCourses": total_courses,
        "totalFaculties": total_faculties,
        "totalAttendanceRecords": attendance_total,
        "attendanceRate": attendance_rate,
    }


@router.get("/absence-ranking", dependencies=[report_access_dependency])
def absence_ranking(
    page: int = 1,
    limit: int = 10,
    search: str | None = None,
    type: str | None = None,
    faculty: str | None = None,
    department: str | None = None,
    course: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    db: Session = Depends(get_role_scoped_db),
):
    query = (
        db.query(AttendanceRecord, Student, Course, AttendanceSession)
        .join(Student, Student.id == AttendanceRecord.student_id)
        .join(Course, Course.id == AttendanceRecord.course_id)
        .join(AttendanceSession, AttendanceSession.id == AttendanceRecord.session_id)
    )

    if start_date:
        query = query.filter(AttendanceSession.session_date >= start_date)
    if end_date:
        query = query.filter(AttendanceSession.session_date <= end_date)

    if search:
        pattern = f"%{search.strip()}%"
        query = query.filter(
            or_(
                Student.full_name.ilike(pattern),
                Student.student_number.ilike(pattern),
                Course.title.ilike(pattern),
                Course.code.ilike(pattern),
            )
        )

    if faculty and faculty.lower() != "all":
        pattern = f"%{faculty.strip()}%"
        query = query.filter(
            or_(
                Course.faculty.has(Faculty.name.ilike(pattern)),
                Student.faculty.has(Faculty.name.ilike(pattern)),
                Course.title.ilike(pattern),
                Course.code.ilike(pattern),
            )
        )

    if department and department.lower() != "all":
        pattern = f"%{department.strip()}%"
        query = query.filter(
            or_(
                Course.department.has(Department.name.ilike(pattern)),
                Student.department.has(Department.name.ilike(pattern)),
                Course.title.ilike(pattern),
                Course.code.ilike(pattern),
            )
        )

    if course and course.lower() != "all":
        pattern = f"%{course.strip()}%"
        query = query.filter(
            or_(
                Course.title.ilike(pattern),
                Course.code.ilike(pattern),
            )
        )

    if type and type.lower() != "" and type.lower() != "student_attendance":
        return {"data": [], "total": 0}

    grouped: dict[str, dict] = {}
    for record, student, course_obj, _session in query.all():
        key = f"{student.id}:{course_obj.id}"
        item = grouped.setdefault(
            key,
            {
                "id": key,
                "studentName": student.full_name,
                "type": "Student",
                "facultyOrDepartment": f"{course_obj.title} ({course_obj.code})",
                "totalAbsences": 0,
                "presentCount": 0,
                "lateCount": 0,
                "total": 0,
            },
        )
        if record.status == AttendanceStatus.ABSENT:
            item["totalAbsences"] += 1
        if record.status in (AttendanceStatus.PRESENT, AttendanceStatus.LATE):
            item["presentCount"] += 1
        if record.status == AttendanceStatus.LATE:
            item["lateCount"] += 1
        item["total"] += 1

    records = []
    for item in grouped.values():
        total = item["total"]
        present_or_late = item["presentCount"]
        attendance_percentage = round((present_or_late / total) * 100) if total else 0
        records.append(
            {
                "id": item["id"],
                "studentName": item["studentName"],
                "type": item["type"],
                "facultyOrDepartment": item["facultyOrDepartment"],
                "totalAbsences": item["totalAbsences"],
                "attendancePercentage": attendance_percentage,
                "status": (
                    "Low"
                    if attendance_percentage < 50
                    else "Normal"
                    if attendance_percentage < 75
                    else "Good"
                ),
            }
        )

    records.sort(key=lambda x: (-x["totalAbsences"], -x["attendancePercentage"], x["studentName"]))
    total = len(records)
    start = (page - 1) * limit
    end = start + limit

    return {"data": records[start:end], "total": total}


@router.get("/course/{course_id}", dependencies=[Depends(require_roles("TEACHER", "FACULTY", "ACADEMIA"))])
def course_report(
    course_id: int,
    db: Session = Depends(get_role_scoped_db),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    if faculty_scope is not None:
        enforce_faculty_scope(_course_faculty_id(db, course_id), faculty_scope)

    total = (
        db.query(func.count(AttendanceRecord.id))
        .filter(AttendanceRecord.course_id == course_id)
        .scalar()
    )
    present = (
        db.query(func.count(AttendanceRecord.id))
        .filter(
            AttendanceRecord.course_id == course_id,
            AttendanceRecord.status.in_([AttendanceStatus.PRESENT, AttendanceStatus.LATE]),
        )
        .scalar()
    )
    late = (
        db.query(func.count(AttendanceRecord.id))
        .filter(
            AttendanceRecord.course_id == course_id,
            AttendanceRecord.status == AttendanceStatus.LATE,
        )
        .scalar()
    )
    absent = (
        db.query(func.count(AttendanceRecord.id))
        .filter(
            AttendanceRecord.course_id == course_id,
            AttendanceRecord.status == AttendanceStatus.ABSENT,
        )
        .scalar()
    )

    course = db.query(Course).filter(Course.id == course_id).first()

    return {
        "course_id": course_id,
        "course_title": course.title if course else None,
        "total_records": total or 0,
        "present": present or 0,
        "late": late or 0,
        "absent": absent or 0,
    }


@router.get("/course/{course_id}/range", dependencies=[Depends(require_roles("TEACHER", "FACULTY", "ACADEMIA"))])
def course_report_range(
    course_id: int,
    start_date: date,
    end_date: date,
    db: Session = Depends(get_role_scoped_db),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    if faculty_scope is not None:
        enforce_faculty_scope(_course_faculty_id(db, course_id), faculty_scope)

    if end_date < start_date:
        raise HTTPException(status_code=400, detail="end_date must be greater than or equal to start_date")

    start_dt = datetime.combine(start_date, time.min)
    end_dt = datetime.combine(end_date, time.max)

    query = (
        db.query(AttendanceRecord)
        .join(AttendanceSession, AttendanceSession.id == AttendanceRecord.session_id)
        .filter(
            AttendanceRecord.course_id == course_id,
            AttendanceSession.start_time >= start_dt,
            AttendanceSession.start_time <= end_dt,
        )
    )

    records = query.all()
    present = sum(1 for r in records if _is_present_status(r.status))
    late = sum(1 for r in records if r.status == AttendanceStatus.LATE)
    absent = sum(1 for r in records if r.status == AttendanceStatus.ABSENT)

    return {
        "course_id": course_id,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "total_records": len(records),
        "present": present,
        "late": late,
        "absent": absent,
    }


@router.get("/course/{course_id}/students", dependencies=[Depends(require_roles("TEACHER", "FACULTY", "ACADEMIA"))])
def course_report_by_student(
    course_id: int,
    start_date: date | None = None,
    end_date: date | None = None,
    db: Session = Depends(get_role_scoped_db),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    if faculty_scope is not None:
        enforce_faculty_scope(_course_faculty_id(db, course_id), faculty_scope)

    query = (
        db.query(AttendanceRecord, Student)
        .join(Student, Student.id == AttendanceRecord.student_id)
        .join(AttendanceSession, AttendanceSession.id == AttendanceRecord.session_id)
        .filter(AttendanceRecord.course_id == course_id)
    )

    if start_date:
        query = query.filter(AttendanceSession.start_time >= datetime.combine(start_date, time.min))
    if end_date:
        query = query.filter(AttendanceSession.start_time <= datetime.combine(end_date, time.max))

    grouped: dict[int, dict] = {}
    for record, student in query.all():
        item = grouped.setdefault(
            student.id,
            {
                "student_id": student.id,
                "student_number": student.student_number,
                "student_name": student.full_name,
                "present": 0,
                "late": 0,
                "absent": 0,
                "total": 0,
            },
        )
        if _is_present_status(record.status):
            item["present"] += 1
        if record.status == AttendanceStatus.LATE:
            item["late"] += 1
        if record.status == AttendanceStatus.ABSENT:
            item["absent"] += 1
        item["total"] += 1

    return {
        "course_id": course_id,
        "start_date": start_date.isoformat() if start_date else None,
        "end_date": end_date.isoformat() if end_date else None,
        "students": list(grouped.values()),
    }


@router.get("/course/{course_id}/sessions", dependencies=[Depends(require_roles("TEACHER", "FACULTY", "ACADEMIA"))])
def course_report_by_session(
    course_id: int,
    start_date: date | None = None,
    end_date: date | None = None,
    db: Session = Depends(get_role_scoped_db),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    if faculty_scope is not None:
        enforce_faculty_scope(_course_faculty_id(db, course_id), faculty_scope)

    session_query = db.query(AttendanceSession).filter(AttendanceSession.course_id == course_id)
    if start_date:
        session_query = session_query.filter(AttendanceSession.start_time >= datetime.combine(start_date, time.min))
    if end_date:
        session_query = session_query.filter(AttendanceSession.start_time <= datetime.combine(end_date, time.max))

    sessions = session_query.order_by(AttendanceSession.start_time.desc()).all()

    breakdown = []
    for session in sessions:
        records = (
            db.query(AttendanceRecord)
            .filter(AttendanceRecord.course_id == course_id, AttendanceRecord.session_id == session.id)
            .all()
        )
        breakdown.append(
            {
                "session_id": session.id,
                "session_date": session.session_date.isoformat() if session.session_date else None,
                "start_time": session.start_time.isoformat() if session.start_time else None,
                "end_time": session.end_time.isoformat() if session.end_time else None,
                "status": session.status.value,
                "present": sum(1 for r in records if _is_present_status(r.status)),
                "late": sum(1 for r in records if r.status == AttendanceStatus.LATE),
                "absent": sum(1 for r in records if r.status == AttendanceStatus.ABSENT),
                "total": len(records),
            }
        )

    return {
        "course_id": course_id,
        "start_date": start_date.isoformat() if start_date else None,
        "end_date": end_date.isoformat() if end_date else None,
        "sessions": breakdown,
    }


# ---------------------------------------------------------------------------
# Academia-scope endpoints — cross-faculty comparison
# ---------------------------------------------------------------------------

academia_access = Depends(require_roles("SUPER_ADMIN", "ADMIN", "ACADEMIA"))


@router.get("/faculty-comparison", dependencies=[academia_access])
def faculty_comparison(
    start_date: date | None = None,
    end_date: date | None = None,
    db: Session = Depends(get_role_scoped_db),
):
    """
    Return attendance statistics for every faculty side-by-side.
    Used by the Academia Reports page for cross-faculty comparison.
    """
    faculties = db.query(Faculty).order_by(Faculty.name).all()

    results = []
    for faculty in faculties:
        # Count enrolled (approved) students
        student_count = (
            db.query(func.count(Student.id))
            .filter(
                Student.faculty_id == faculty.id,
                Student.status == StudentAdmissionStatus.APPROVED,
            )
            .scalar() or 0
        )

        # Count sessions that belong to this faculty's courses
        session_q = (
            db.query(func.count(AttendanceSession.id))
            .join(Course, Course.id == AttendanceSession.course_id)
            .filter(Course.faculty_id == faculty.id)
        )
        if start_date:
            session_q = session_q.filter(AttendanceSession.session_date >= start_date)
        if end_date:
            session_q = session_q.filter(AttendanceSession.session_date <= end_date)
        session_count = session_q.scalar() or 0

        # Count attendance records for this faculty
        record_q = (
            db.query(AttendanceRecord)
            .join(Course, Course.id == AttendanceRecord.course_id)
            .join(AttendanceSession, AttendanceSession.id == AttendanceRecord.session_id)
            .filter(Course.faculty_id == faculty.id)
        )
        if start_date:
            record_q = record_q.filter(AttendanceSession.session_date >= start_date)
        if end_date:
            record_q = record_q.filter(AttendanceSession.session_date <= end_date)

        records = record_q.all()
        total = len(records)
        present = sum(1 for r in records if _is_present_status(r.status))
        late = sum(1 for r in records if r.status == AttendanceStatus.LATE)
        absent = sum(1 for r in records if r.status == AttendanceStatus.ABSENT)
        pct = round((present / total) * 100, 1) if total else 0.0

        # At-risk: students with < 75% across any course in this faculty
        # We compute per-student totals and count those below threshold
        student_totals: dict[int, dict] = {}
        for r in records:
            entry = student_totals.setdefault(r.student_id, {"present": 0, "total": 0})
            entry["total"] += 1
            if _is_present_status(r.status):
                entry["present"] += 1
        at_risk = sum(
            1 for v in student_totals.values()
            if v["total"] > 0 and (v["present"] / v["total"]) < 0.75
        )

        results.append({
            "faculty_id": faculty.id,
            "faculty_name": faculty.name,
            "faculty_code": faculty.code,
            "total_students": student_count,
            "total_sessions": session_count,
            "total_records": total,
            "present": present,
            "late": late,
            "absent": absent,
            "attendance_pct": pct,
            "at_risk_students": at_risk,
        })

    # Sort highest attendance first
    results.sort(key=lambda x: -x["attendance_pct"])

    # Institution-wide summary
    all_total = sum(r["total_records"] for r in results)
    all_present = sum(r["present"] for r in results)
    institution_avg = round((all_present / all_total) * 100, 1) if all_total else 0.0

    return {
        "faculties": results,
        "institution_avg": institution_avg,
        "start_date": start_date.isoformat() if start_date else None,
        "end_date": end_date.isoformat() if end_date else None,
    }


@router.get("/department-comparison", dependencies=[academia_access])
def department_comparison(
    faculty_id: int,
    start_date: date | None = None,
    end_date: date | None = None,
    db: Session = Depends(get_role_scoped_db),
):
    """
    Return attendance stats for all departments within one faculty.
    Used for the drill-down view when clicking a faculty card.
    """
    faculty = db.query(Faculty).filter(Faculty.id == faculty_id).first()
    if not faculty:
        raise HTTPException(status_code=404, detail="Faculty not found")

    departments = (
        db.query(Department)
        .filter(Department.faculty_id == faculty_id)
        .order_by(Department.name)
        .all()
    )

    results = []
    for dept in departments:
        student_count = (
            db.query(func.count(Student.id))
            .filter(
                Student.department_id == dept.id,
                Student.status == StudentAdmissionStatus.APPROVED,
            )
            .scalar() or 0
        )

        record_q = (
            db.query(AttendanceRecord)
            .join(Course, Course.id == AttendanceRecord.course_id)
            .join(AttendanceSession, AttendanceSession.id == AttendanceRecord.session_id)
            .filter(Course.department_id == dept.id)
        )
        if start_date:
            record_q = record_q.filter(AttendanceSession.session_date >= start_date)
        if end_date:
            record_q = record_q.filter(AttendanceSession.session_date <= end_date)

        records = record_q.all()
        total = len(records)
        present = sum(1 for r in records if _is_present_status(r.status))
        late = sum(1 for r in records if r.status == AttendanceStatus.LATE)
        absent = sum(1 for r in records if r.status == AttendanceStatus.ABSENT)
        pct = round((present / total) * 100, 1) if total else 0.0

        results.append({
            "department_id": dept.id,
            "department_name": dept.name,
            "department_code": dept.code,
            "total_students": student_count,
            "total_records": total,
            "present": present,
            "late": late,
            "absent": absent,
            "attendance_pct": pct,
        })

    results.sort(key=lambda x: -x["attendance_pct"])

    return {
        "faculty_id": faculty_id,
        "faculty_name": faculty.name,
        "departments": results,
        "start_date": start_date.isoformat() if start_date else None,
        "end_date": end_date.isoformat() if end_date else None,
    }


@router.get("/attendance-trends", dependencies=[academia_access])
def attendance_trends(
    period: str = "monthly",   # "weekly" | "monthly"
    start_date: date | None = None,
    end_date: date | None = None,
    db: Session = Depends(get_role_scoped_db),
):
    """
    Return per-faculty attendance % bucketed by week or month.
    Used for the trend line chart.
    """
    if period not in ("weekly", "monthly"):
        raise HTTPException(status_code=400, detail="period must be 'weekly' or 'monthly'")

    faculties = db.query(Faculty).order_by(Faculty.name).all()
    faculty_map = {f.id: f.name for f in faculties}

    record_q = (
        db.query(AttendanceRecord, AttendanceSession, Course)
        .join(AttendanceSession, AttendanceSession.id == AttendanceRecord.session_id)
        .join(Course, Course.id == AttendanceRecord.course_id)
    )
    if start_date:
        record_q = record_q.filter(AttendanceSession.session_date >= start_date)
    if end_date:
        record_q = record_q.filter(AttendanceSession.session_date <= end_date)

    # Group into buckets: {bucket_label: {faculty_id: {present, total}}}
    buckets: dict[str, dict[int, dict]] = {}
    for record, session, course in record_q.all():
        d = session.session_date
        if period == "weekly":
            # ISO week label e.g. "2024-W03"
            label = f"{d.isocalendar()[0]}-W{d.isocalendar()[1]:02d}"
        else:
            label = f"{d.year}-{d.month:02d}"

        fid = course.faculty_id
        bucket = buckets.setdefault(label, {})
        entry = bucket.setdefault(fid, {"present": 0, "total": 0})
        entry["total"] += 1
        if _is_present_status(record.status):
            entry["present"] += 1

    # Build sorted list of series per faculty
    # Return format: [{period, series: [{faculty_id, faculty_name, pct}]}]
    sorted_labels = sorted(buckets.keys())
    series = []
    for label in sorted_labels:
        bucket = buckets[label]
        faculty_points = []
        for fid, fname in faculty_map.items():
            entry = bucket.get(fid, {"present": 0, "total": 0})
            pct = round((entry["present"] / entry["total"]) * 100, 1) if entry["total"] else None
            faculty_points.append({
                "faculty_id": fid,
                "faculty_name": fname,
                "pct": pct,
            })
        series.append({"period": label, "faculties": faculty_points})

    return {
        "period": period,
        "series": series,
        "faculty_names": list(faculty_map.values()),
        "start_date": start_date.isoformat() if start_date else None,
        "end_date": end_date.isoformat() if end_date else None,
    }


@router.get("/course-ranking", dependencies=[academia_access])
def course_ranking(
    start_date: date | None = None,
    end_date: date | None = None,
    faculty_id: int | None = None,
    limit: int = 50,
    db: Session = Depends(get_role_scoped_db),
):
    """
    Return all courses ranked by attendance percentage.
    Optionally scoped to a single faculty.
    """
    record_q = (
        db.query(AttendanceRecord, Course, AttendanceSession)
        .join(Course, Course.id == AttendanceRecord.course_id)
        .join(AttendanceSession, AttendanceSession.id == AttendanceRecord.session_id)
    )
    if faculty_id:
        record_q = record_q.filter(Course.faculty_id == faculty_id)
    if start_date:
        record_q = record_q.filter(AttendanceSession.session_date >= start_date)
    if end_date:
        record_q = record_q.filter(AttendanceSession.session_date <= end_date)

    course_stats: dict[int, dict] = {}
    for record, course, _session in record_q.all():
        entry = course_stats.setdefault(course.id, {
            "course_id": course.id,
            "course_title": course.title,
            "course_code": course.code,
            "faculty_id": course.faculty_id,
            "present": 0,
            "total": 0,
        })
        entry["total"] += 1
        if _is_present_status(record.status):
            entry["present"] += 1

    # Resolve faculty names in bulk
    fids = {v["faculty_id"] for v in course_stats.values()}
    fac_names = {f.id: f.name for f in db.query(Faculty).filter(Faculty.id.in_(fids)).all()} if fids else {}

    results = []
    for entry in course_stats.values():
        total = entry["total"]
        pct = round((entry["present"] / total) * 100, 1) if total else 0.0
        results.append({
            "course_id": entry["course_id"],
            "course_title": entry["course_title"],
            "course_code": entry["course_code"],
            "faculty_name": fac_names.get(entry["faculty_id"], "—"),
            "total_records": total,
            "present": entry["present"],
            "attendance_pct": pct,
            "status": "good" if pct >= 75 else "warning" if pct >= 50 else "low",
        })

    results.sort(key=lambda x: -x["attendance_pct"])
    return {"courses": results[:limit], "total": len(results)}
