"""Faculty-side excuse request service.

Handles listing pending requests and approving / denying them.
On approval, every matching ABSENT AttendanceRecord for that student+date
(optionally scoped to a specific course) is flipped to EXCUSED.
"""
from __future__ import annotations

from datetime import datetime

from fastapi import HTTPException
from sqlalchemy.orm import Session, joinedload

from app.db.models import (
    AttendanceRecord,
    AttendanceSession,
    AttendanceStatus,
    ExcuseRequest,
    ExcuseRequestStatus,
    Student,
    User,
)
from app.services.notification_service import create_notification, NotificationType


class ExcuseRequestService:

    def list_for_faculty(self, db: Session, faculty_id: int) -> list[dict]:
        """Return all requests (any status) for students in this faculty, newest-first."""
        rows = (
            db.query(ExcuseRequest)
            .join(Student, Student.id == ExcuseRequest.student_id)
            .options(
                joinedload(ExcuseRequest.course),
                joinedload(ExcuseRequest.student),
            )
            .filter(Student.faculty_id == faculty_id)
            .order_by(ExcuseRequest.created_at.desc())
            .all()
        )
        return [self._to_dict(r) for r in rows]

    def review(
        self,
        db: Session,
        request_id: int,
        faculty_id: int,
        reviewer: User,
        action: str,   # "approve" | "deny"
    ) -> dict:
        """Approve or deny a request.

        On approval:
          - Flip matching ABSENT AttendanceRecord rows → EXCUSED
          - Notify the student
        On denial:
          - Notify the student
        """
        req = (
            db.query(ExcuseRequest)
            .join(Student, Student.id == ExcuseRequest.student_id)
            .options(
                joinedload(ExcuseRequest.course),
                joinedload(ExcuseRequest.student),
            )
            .filter(
                ExcuseRequest.id == request_id,
                Student.faculty_id == faculty_id,
            )
            .first()
        )
        if req is None:
            raise HTTPException(status_code=404, detail="Excuse request not found")
        if req.status != ExcuseRequestStatus.PENDING:
            raise HTTPException(status_code=409, detail="Request already reviewed")

        if action == "approve":
            req.status = ExcuseRequestStatus.APPROVED
            self._flip_absent_records(db, req)
            notif_title = "Excuse Request Approved"
            notif_msg = (
                f"Your excuse request for {req.request_date}"
                + (f" ({req.course.title})" if req.course else " (all courses)")
                + " has been approved."
            )
            notif_type = NotificationType.SUCCESS
        elif action == "deny":
            req.status = ExcuseRequestStatus.DENIED
            notif_title = "Excuse Request Denied"
            notif_msg = (
                f"Your excuse request for {req.request_date}"
                + (f" ({req.course.title})" if req.course else " (all courses)")
                + " has been denied."
            )
            notif_type = NotificationType.WARNING
        else:
            raise HTTPException(status_code=400, detail="action must be 'approve' or 'deny'")

        req.reviewed_at = datetime.utcnow()
        req.reviewed_by = reviewer.id
        db.commit()
        db.refresh(req)

        # Notify the student's User account (match by student_number / student_id)
        student = req.student
        student_user = (
            db.query(User)
            .filter(
                (User.student_id == student.id) | (User.username == student.student_number)
            )
            .first()
        )
        if student_user:
            create_notification(
                db, student_user.id,
                notif_title, notif_msg, notif_type,
                link="/student/attendance",
            )

        return self._to_dict(req)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _flip_absent_records(self, db: Session, req: ExcuseRequest) -> None:
        """Set all matching ABSENT records to EXCUSED."""
        q = (
            db.query(AttendanceRecord)
            .join(AttendanceSession, AttendanceSession.id == AttendanceRecord.session_id)
            .filter(
                AttendanceRecord.student_id == req.student_id,
                AttendanceRecord.status == AttendanceStatus.ABSENT,
                AttendanceSession.session_date == req.request_date,
            )
        )
        if req.course_id is not None:
            q = q.filter(AttendanceRecord.course_id == req.course_id)
        for record in q.all():
            record.status = AttendanceStatus.EXCUSED
        # commit happens in review() after status update

    @staticmethod
    def _to_dict(r: ExcuseRequest) -> dict:
        student = r.student
        return {
            "id": r.id,
            "student_id": r.student_id,
            "student_name": student.full_name if student else None,
            "student_number": student.student_number if student else None,
            "course_id": r.course_id,
            "course_name": r.course.title if r.course else None,
            "course_code": r.course.code if r.course else None,
            "request_date": r.request_date.isoformat(),
            "reason": r.reason,
            "status": r.status.value,
            "created_at": r.created_at.isoformat(),
            "reviewed_at": r.reviewed_at.isoformat() if r.reviewed_at else None,
        }


excuse_request_service = ExcuseRequestService()
