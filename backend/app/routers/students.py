from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import or_, func
from sqlalchemy.exc import IntegrityError

from app.core.security import get_current_user, require_roles, get_password_hash
from app.db.faculty_scope import enforce_faculty_scope, get_optional_faculty_scope_context
from app.db.models import Role, Student, StudentAdmissionStatus, User
from app.db.role_scoped import get_role_scoped_db
from app.utils.activity_logger import log_activity
from app.services.enrollment_service import auto_enroll_student_in_matching_courses
from app.schemas.student import (
    StudentCreate,
    StudentDashboardStatsRead,
    StudentCapturedImagesRead,
    StudentRead,
    StudentStatus,
    StudentUpdate,
    PaginatedStudentRead,
)
from app.utils.organization import (
    ensure_department_belongs_to_faculty,
    get_department_or_404,
    get_faculty_or_404,
)
from app.utils.student_numbering import next_available_student_number
from app.utils.student_numbering import student_dataset_dir as resolve_student_dataset_dir
from pydantic import BaseModel, Field
from fastapi import Body
from utils.config import load_config
from app.utils.image_decode import decode_base64_image
from face_recognition.train import train_embeddings_from_dataset
from app.services.face_service import face_service
import cv2
from pathlib import Path
import re
from fastapi import Response
from app.services.notification_service import create_notification, NotificationType


router = APIRouter(prefix="/students", tags=["students"])


NEW_ADMISSIONS_WINDOW_DAYS = 30
_DATASET_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def _count_students_by_status(db: Session, status: StudentAdmissionStatus) -> int:
    return int(db.query(func.count(Student.id)).filter(Student.status == status).scalar() or 0)


def _student_or_404(db: Session, student_id: int) -> Student:
    student = db.query(Student).filter(Student.id == student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    return student


def _student_dataset_dir(student: Student) -> Path | None:
    candidate_numbers: list[str] = []
    if student.embedding_ref:
        candidate_numbers.append(student.embedding_ref)
    candidate_numbers.append(student.student_number)

    for value in candidate_numbers:
        candidate = resolve_student_dataset_dir(value)
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _student_dataset_images(student: Student) -> list[Path]:
    dataset_dir = _student_dataset_dir(student)
    if dataset_dir is None:
        return []

    return sorted(
        [
            file_path
            for file_path in dataset_dir.iterdir()
            if file_path.is_file() and file_path.suffix.lower() in _DATASET_IMAGE_SUFFIXES
        ],
        key=lambda path: path.name.lower(),
    )


@router.get(
    "/stats",
    response_model=StudentDashboardStatsRead,
    dependencies=[Depends(require_roles("ADMISSIONS", "FACULTY", "TEACHER", "ACADEMIA", "HR"))],
)
def student_dashboard_stats(db: Session = Depends(get_role_scoped_db)):
    now_utc = datetime.now(timezone.utc)
    recent_cutoff = now_utc - timedelta(days=NEW_ADMISSIONS_WINDOW_DAYS)

    total_students = int(db.query(func.count(Student.id)).scalar() or 0)
    new_admissions = int(
        db.query(func.count(Student.id)).filter(Student.created_at >= recent_cutoff).scalar() or 0
    )
    pending_admissions = _count_students_by_status(db, StudentAdmissionStatus.PENDING)
    rejected_applications = _count_students_by_status(db, StudentAdmissionStatus.REJECTED)

    return {
        "total_students": total_students,
        "new_admissions": new_admissions,
        "pending_admissions": pending_admissions,
        "rejected_applications": rejected_applications,
    }


@router.get(
    "/{student_id}/captured-images",
    response_model=StudentCapturedImagesRead,
    dependencies=[Depends(require_roles("ADMISSIONS", "FACULTY", "TEACHER", "ACADEMIA", "HR"))],
)
def list_student_captured_images(
    student_id: int,
    db: Session = Depends(get_role_scoped_db),
):
    student = _student_or_404(db, student_id)
    image_paths = _student_dataset_images(student)
    images = [
        {
            "file_name": image_path.name,
            "url": f"/students/{student.id}/captured-images/{image_path.name}",
        }
        for image_path in image_paths
    ]

    return {
        "student_id": student.id,
        "student_number": student.student_number,
        "image_count": len(images),
        "images": images,
    }


@router.get(
    "/{student_id}/captured-images/{file_name}",
    dependencies=[Depends(require_roles("ADMISSIONS", "FACULTY", "TEACHER", "ACADEMIA", "HR"))],
)
def read_student_captured_image(
    student_id: int,
    file_name: str,
    db: Session = Depends(get_role_scoped_db),
):
    student = _student_or_404(db, student_id)
    safe_file_name = Path(file_name).name
    if safe_file_name != file_name:
        raise HTTPException(status_code=400, detail="Invalid file name")

    dataset_dir = _student_dataset_dir(student)
    if dataset_dir is None:
        raise HTTPException(status_code=404, detail="No dataset images found for this student")

    image_path = dataset_dir / safe_file_name
    if not image_path.exists() or not image_path.is_file() or image_path.suffix.lower() not in _DATASET_IMAGE_SUFFIXES:
        raise HTTPException(status_code=404, detail="Captured image not found")

    return FileResponse(image_path)


@router.post("", response_model=StudentRead, dependencies=[Depends(require_roles("ADMISSIONS"))])
def create_student(
    payload: StudentCreate,
    db: Session = Depends(get_role_scoped_db),
    current_user: User = Depends(get_current_user),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    if faculty_scope is None:
        faculty = get_faculty_or_404(db, payload.faculty_id)
        faculty_code = faculty.code
    else:
        enforce_faculty_scope(payload.faculty_id, faculty_scope)
        faculty_code = faculty_scope.faculty_code

    department = get_department_or_404(db, payload.department_id)
    ensure_department_belongs_to_faculty(department, payload.faculty_id)
    student_number = next_available_student_number(db, faculty_code, date.today().year)
    embedding_ref = student_number

    obj = Student(
        student_number=student_number,
        full_name=payload.full_name,
        faculty_id=payload.faculty_id,
        department_id=payload.department_id,
        embedding_ref=embedding_ref,
    )
    db.add(obj)
    try:
        db.flush()
        auto_enroll_student_in_matching_courses(db, obj)
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Student number already exists") from exc
    db.refresh(obj)

    # No user account is created at registration time.
    # Credentials are only issued when admissions approves the student (see update_student).

    log_activity(
        action=f"Student Registered - {obj.full_name} ({obj.student_number})",
        user=current_user,
        db=db,
    )

    return StudentRead.model_validate(obj)


class FaceDetectRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image frame (data URI or raw base64)")


@router.post(
    "/detect",
    dependencies=[Depends(require_roles("ADMISSIONS"))],
    summary="Detect faces in a frame (for capture-time validation)",
)
def detect_faces_in_frame(payload: FaceDetectRequest):
    """
    Detect how many faces are present in the given frame and return an embedding
    for the largest detected face.  Used by the face-capture UI to enforce the
    single-person and same-person rules during image collection.

    Works even before any student model has been trained.
    """
    try:
        frame = decode_base64_image(payload.image)
    except Exception:
        return {"face_count": 0, "embedding": None}

    result = face_service.detect_faces(frame)
    return result


@router.get("", response_model=PaginatedStudentRead, dependencies=[Depends(require_roles("ADMISSIONS", "FACULTY", "TEACHER", "ACADEMIA"))])
def list_students(
    db: Session = Depends(get_role_scoped_db),
    skip: int = Query(default=0, ge=0, description="Number of rows to skip", examples=[0]),
    limit: int = Query(default=50, ge=1, le=200, description="Page size", examples=[20]),
    faculty_id: int | None = Query(default=None, description="Filter by faculty id", examples=[1]),
    department_id: int | None = Query(default=None, description="Filter by department id", examples=[1]),
    status: StudentStatus | None = Query(default=None, description="Filter by admission status", examples=["pending"]),
    search: str | None = Query(default=None, description="Search by student number or full name", examples=["2201"]),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    query = db.query(Student)
    if faculty_scope is not None:
        if faculty_id is not None:
            enforce_faculty_scope(faculty_id, faculty_scope)
        else:
            faculty_id = faculty_scope.faculty_id
    if faculty_id is not None:
        query = query.filter(Student.faculty_id == faculty_id)
    if department_id is not None:
        query = query.filter(Student.department_id == department_id)
    if status is not None:
        query = query.filter(Student.status == StudentAdmissionStatus(status.value))
    if search:
        pattern = f"%{search.strip()}%"
        query = query.filter(or_(Student.full_name.ilike(pattern), Student.student_number.ilike(pattern)))
    total = query.count()
    items = query.order_by(Student.created_at.desc(), Student.id.desc()).offset(skip).limit(limit).all()
    return {"items": items, "total": total, "skip": skip, "limit": limit}


@router.put("/{student_id}", response_model=StudentRead, dependencies=[Depends(require_roles("ADMISSIONS"))])
def update_student(
    student_id: int,
    payload: StudentUpdate,
    db: Session = Depends(get_role_scoped_db),
    faculty_scope = Depends(get_optional_faculty_scope_context),
):
    obj = db.query(Student).filter(Student.id == student_id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Student not found")

    target_faculty_id = payload.faculty_id if payload.faculty_id is not None else obj.faculty_id
    target_department_id = payload.department_id if payload.department_id is not None else obj.department_id

    if faculty_scope is None:
        get_faculty_or_404(db, target_faculty_id)
    else:
        enforce_faculty_scope(target_faculty_id, faculty_scope)

    department = get_department_or_404(db, target_department_id)
    ensure_department_belongs_to_faculty(department, target_faculty_id)

    previous_status = obj.status

    payload_data = payload.model_dump(exclude_unset=True)
    for field in ("faculty_id", "department_id"):
        payload_data.pop(field, None)

    for field, value in payload_data.items():
        setattr(obj, field, value)

    obj.faculty_id = target_faculty_id
    obj.department_id = target_department_id

    db.add(obj)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Update conflicts with existing student data") from exc
    db.refresh(obj)

    # ── Create user account on approval / handle status change notifications ──
    generated_password: str | None = None
    new_status = obj.status

    if previous_status != new_status:
        if new_status == StudentAdmissionStatus.APPROVED:
            # Create login credentials now that the student is approved.
            existing_user = db.query(User).filter(User.username == obj.student_number).first()
            if not existing_user:
                plain_password = obj.student_number
                student_role = db.query(Role).filter(Role.name == "STUDENT").first()
                new_user = User(
                    username=obj.student_number,
                    hashed_password=get_password_hash(plain_password),
                    student_id=obj.id,
                    is_active=True,
                )
                if student_role:
                    new_user.roles = [student_role]
                db.add(new_user)
                try:
                    db.commit()
                    generated_password = plain_password
                except IntegrityError:
                    db.rollback()
            else:
                # Account already exists (e.g. re-approved after rejection) — reactivate it
                existing_user.is_active = True
                existing_user.student_id = obj.id
                db.commit()

            # Notify the student
            student_user = db.query(User).filter(User.student_id == obj.id).first()
            if student_user:
                create_notification(
                    db, student_user.id,
                    title="Admission Approved",
                    message=(
                        "Congratulations! Your admission has been approved. "
                        "You can now log in to your student portal."
                    ),
                    notif_type=NotificationType.SUCCESS,
                    link="/student/dashboard",
                )

        elif new_status == StudentAdmissionStatus.REJECTED:
            # Deactivate the student's login account if it exists
            student_user = db.query(User).filter(User.student_id == obj.id).first()
            if student_user:
                student_user.is_active = False
                db.commit()
                create_notification(
                    db, student_user.id,
                    title="Admission Update",
                    message=(
                        "Your admission application has been rejected. "
                        "Please contact the admissions office for more information."
                    ),
                    notif_type=NotificationType.ERROR,
                    link="/student/dashboard",
                )

    result = StudentRead.model_validate(obj)
    result.generated_password = generated_password
    return result


@router.delete("/{student_id}", dependencies=[Depends(require_roles("ADMISSIONS"))])
def delete_student(student_id: int, db: Session = Depends(get_role_scoped_db)):
    obj = db.query(Student).filter(Student.id == student_id).first()
    if not obj:
        raise HTTPException(status_code=404, detail="Student not found")
    # Detach any user accounts that reference this student.
    # The users.student_id FK has no ON DELETE rule so MySQL would block the
    # delete otherwise.  Setting it to NULL preserves the user account but
    # removes the student link.
    db.query(User).filter(User.student_id == student_id).update(
        {User.student_id: None}, synchronize_session="fetch"
    )
    db.delete(obj)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Cannot delete student due to related records") from exc
    return {"deleted": True, "student_id": student_id}



class CaptureRequest(BaseModel):
    faculty_code: str = Field(..., min_length=2)
    student_number: str = Field(..., min_length=3)
    images: list[str] = Field(..., min_items=1)
    overwrite: bool = Field(
        False,
        description="If True, clear existing images for this student before saving new ones.",
    )


@router.post(
    "/capture",
    dependencies=[Depends(require_roles("ADMISSIONS"))],
)
def capture_and_train(
    payload: CaptureRequest = Body(...),
    db: Session = Depends(get_role_scoped_db),
):
    # Basic validation and sanitization
    faculty_code = payload.faculty_code.strip().upper()
    student_number = payload.student_number.strip().upper()

    if not re.match(r"^[A-Z0-9_-]+$", faculty_code):
        raise HTTPException(status_code=400, detail="Invalid faculty code")
    if not re.match(r"^[A-Z0-9_-]+$", student_number):
        raise HTTPException(status_code=400, detail="Invalid student number")

    # Resolve dataset root from config
    cfg = load_config()
    dataset_root = Path(cfg.dataset_dir)
    faculty_dir = dataset_root / faculty_code
    student_dir = faculty_dir / student_number

    try:
        faculty_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to create faculty dir: {exc}") from exc

    # Guard against accidentally overwriting existing images unless the caller explicitly requested it
    if student_dir.exists() and any(student_dir.iterdir()):
        if not payload.overwrite:
            raise HTTPException(
                status_code=409,
                detail="Student dataset folder already exists. Send overwrite=true to replace existing images.",
            )
        # overwrite=True: remove existing image files before saving new ones
        for old_img in list(student_dir.iterdir()):
            if old_img.is_file() and old_img.suffix.lower() in _DATASET_IMAGE_SUFFIXES:
                try:
                    old_img.unlink()
                except Exception:
                    pass  # best effort

    try:
        student_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to create student dir: {exc}") from exc

    saved = 0
    # Determine next image index
    existing = sorted([p for p in student_dir.iterdir() if p.is_file()])
    def _next_index():
        idx = 1
        for p in existing:
            m = re.search(r"(\d+)", p.stem)
            if m:
                try:
                    idx = max(idx, int(m.group(1)) + 1)
                except Exception:
                    pass
        return idx

    index = _next_index()
    for img_b64 in payload.images:
        try:
            frame = decode_base64_image(img_b64)
        except Exception as exc:
            # Rollback saved files if any
            for p in student_dir.iterdir():
                try:
                    p.unlink()
                except Exception:
                    pass
            raise HTTPException(status_code=400, detail=f"Invalid image payload: {exc}") from exc

        # write jpeg
        file_name = f"img_{index:03d}.jpg"
        dest = student_dir / file_name
        success, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not success:
            for p in student_dir.iterdir():
                try:
                    p.unlink()
                except Exception:
                    pass
            raise HTTPException(status_code=500, detail="Failed to encode image")
        try:
            with open(dest, 'wb') as f:
                f.write(encoded.tobytes())
        except Exception as exc:
            for p in student_dir.iterdir():
                try:
                    p.unlink()
                except Exception:
                    pass
            raise HTTPException(status_code=500, detail=f"Failed to write image file: {exc}") from exc

        saved += 1
        index += 1

    # Enqueue embedding training as a background job.
    # _train_wrapper runs in its own background thread (via enqueue_embeddings_training),
    # so it is safe to do CPU-heavy work here — it will not block any HTTP requests.
    try:
        from app.services.training_manager import enqueue_embeddings_training

        def _train_wrapper():
            cfg = load_config()

            # ── Borrow already-loaded models from the recognition service ──────
            # Creating brand-new FaceNet + SCRFD instances inside the training
            # thread causes a long hang on Windows: ONNX Runtime serialises
            # concurrent model loads. Reusing the live recognizer's objects
            # (or the detect-only recognizer pre-warmed at startup) avoids this.
            borrowed_model = None
            borrowed_detector = None
            try:
                with face_service._lock:
                    rec = face_service._recognizer
                    if rec is not None:
                        borrowed_model = rec.model
                        borrowed_detector = rec.detector
            except Exception:
                pass

            if borrowed_model is None:
                # Fall back to the detect-only recognizer (pre-warmed at startup).
                try:
                    det_rec = face_service._ensure_detect_recognizer()
                    if det_rec is not None:
                        borrowed_model = det_rec.model
                        borrowed_detector = det_rec.detector
                except Exception:
                    pass  # last resort: training loads its own fresh copies

            # Run training directly — this thread is already a background job.
            train_embeddings_from_dataset(
                cfg,
                model=borrowed_model,
                detector=borrowed_detector,
            )

            # Reload recognition models so next scan uses the new embeddings.
            try:
                face_service.reload_models()
            except Exception:
                # Non-fatal: mtime-based auto-reload will pick up the new file
                # on the next recognition request.
                pass

        job_id = enqueue_embeddings_training(_train_wrapper, meta={
            "faculty_code": faculty_code,
            "student_number": student_number,
            "saved_images": saved,
        })
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to enqueue training job: {exc}") from exc

    return {"saved": saved, "job_id": job_id, "student_number": student_number}





@router.options("/capture")
def capture_preflight():
    # Explicitly allow preflight requests for clients that send CORS preflight
    return Response(status_code=200)
