from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_
from sqlalchemy.exc import IntegrityError
from datetime import datetime, timezone

from app.core.security import require_roles
from app.db.models import Faculty
from app.db.session import get_db
from app.schemas.faculty import FacultyCreate, FacultyRead, FacultyUpdate, PaginatedFacultyRead
from app.services.tenant_provisioning import build_tenant_db_name, provision_faculty_tenant_database


router = APIRouter(prefix="/faculties", tags=["faculties"])


@router.post("", response_model=FacultyRead, dependencies=[Depends(require_roles("ACADEMIA"))])
def create_faculty(payload: FacultyCreate, db: Session = Depends(get_db)):
    try:
        tenant_db_name = build_tenant_db_name(payload.code)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    faculty = Faculty(name=payload.name, code=payload.code, tenant_db_name=tenant_db_name)
    db.add(faculty)
    try:
        db.flush()

        provision = provision_faculty_tenant_database(tenant_db_name)
        if not (provision.provisioned or provision.skipped):
            raise HTTPException(
                status_code=500,
                detail=f"Faculty tenant database provisioning failed: {provision.reason}",
            )

        if provision.provisioned:
            faculty.tenant_db_provisioned_at = datetime.now(timezone.utc)

        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Faculty with same name/code/tenant DB already exists") from exc
    except HTTPException:
        db.rollback()
        raise
    db.refresh(faculty)
    return faculty


@router.get("", response_model=PaginatedFacultyRead, dependencies=[Depends(require_roles("ACADEMIA", "FACULTY_ADMIN"))])
def list_faculties(
    db: Session = Depends(get_db),
    skip: int = Query(default=0, ge=0, description="Number of rows to skip", examples=[0]),
    limit: int = Query(default=50, ge=1, le=200, description="Page size", examples=[20]),
    search: str | None = Query(default=None, description="Search by faculty name or code", examples=["computer"]),
):
    query = db.query(Faculty)
    if search:
        pattern = f"%{search.strip()}%"
        query = query.filter(or_(Faculty.name.ilike(pattern), Faculty.code.ilike(pattern)))
    total = query.count()
    items = query.order_by(Faculty.name).offset(skip).limit(limit).all()
    return {"items": items, "total": total, "skip": skip, "limit": limit}


@router.put("/{faculty_id}", response_model=FacultyRead, dependencies=[Depends(require_roles("ACADEMIA"))])
def update_faculty(faculty_id: int, payload: FacultyUpdate, db: Session = Depends(get_db)):
    faculty = db.query(Faculty).filter(Faculty.id == faculty_id).first()
    if not faculty:
        raise HTTPException(status_code=404, detail="Faculty not found")

    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(faculty, field, value)

    db.add(faculty)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Update conflicts with existing faculty data") from exc
    db.refresh(faculty)
    return faculty


@router.delete("/{faculty_id}", dependencies=[Depends(require_roles("ACADEMIA"))])
def delete_faculty(faculty_id: int, db: Session = Depends(get_db)):
    faculty = db.query(Faculty).filter(Faculty.id == faculty_id).first()
    if not faculty:
        raise HTTPException(status_code=404, detail="Faculty not found")
    db.delete(faculty)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="Cannot delete faculty due to related records") from exc
    return {"deleted": True, "faculty_id": faculty_id}
