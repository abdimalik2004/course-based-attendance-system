from __future__ import annotations

import os
import uuid
from pathlib import Path

from datetime import datetime

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.security import get_current_user, require_roles
from app.db.session import get_db, SessionLocal
from app.db.models import User, Role

router = APIRouter(prefix="/users", tags=["users"])


class UserListItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    email: str | None
    is_active: bool
    faculty_id: int | None
    role_names: list[str]
    created_at: datetime


class UsersListResponse(BaseModel):
    total: int
    items: list[UserListItem]


class UserUpdateRequest(BaseModel):
    username: str | None = None
    email: str | None = None
    faculty_id: int | None = None
    is_active: bool | None = None
    role_names: list[str] | None = None


@router.get("", response_model=UsersListResponse, dependencies=[Depends(require_roles("SUPER_ADMIN"))])
def list_users(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=500),
    search: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    query = db.query(User)
    if search:
        term = f"%{search.strip()}%"
        query = query.filter((User.username.ilike(term)) | (User.email.ilike(term)))

    total = query.count()
    items = query.order_by(User.id.desc()).offset(skip).limit(limit).all()
    return UsersListResponse(total=total, items=items)


@router.put("/{user_id}", response_model=UserListItem, dependencies=[Depends(require_roles("SUPER_ADMIN"))])
def update_user(user_id: int, payload: UserUpdateRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    data = payload.model_dump(exclude_unset=True)

    if "username" in data:
        existing_username = db.query(User).filter(User.username == data["username"], User.id != user_id).first()
        if existing_username:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists")
        user.username = data["username"]

    if "email" in data:
        email = data["email"]
        if email:
            existing_email = db.query(User).filter(User.email == email, User.id != user_id).first()
            if existing_email:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already exists")
        user.email = email

    if "faculty_id" in data:
        user.faculty_id = data["faculty_id"]

    if "is_active" in data:
        user.is_active = data["is_active"]

    if "role_names" in data and data["role_names"] is not None:
        normalized = [name.strip().upper() for name in data["role_names"] if name.strip()]
        if not normalized:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="At least one valid role is required")
        roles = db.query(Role).filter(Role.name.in_(normalized)).all()
        found = {r.name for r in roles}
        missing = [name for name in normalized if name not in found]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"message": "Invalid role names", "missing": missing},
            )
        user.roles = roles

    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(require_roles("SUPER_ADMIN"))])
def delete_user(user_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    if user.id == current_user.id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="You cannot delete your own account")

    db.delete(user)
    db.commit()
    return JSONResponse(status_code=status.HTTP_204_NO_CONTENT, content=None)


@router.post("/me/profile-image")
def upload_profile_image(file: UploadFile = File(...), user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Accepts a multipart upload, saves file to server storage, and updates the user's profile_image_url.

    This is a simple server-side upload handler. In a production setup prefer presigned uploads
    to object storage (S3/GCS) and serving via a CDN.
    """
    # basic validation
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file must be an image")

    filename = f"{user.id}_{uuid.uuid4().hex}{Path(file.filename).suffix}"
    upload_dir = Path(settings.static_upload_dir or "static/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    target_path = upload_dir / filename

    try:
        with target_path.open("wb") as out_file:
            content = file.file.read()
            out_file.write(content)
    except Exception as exc:  # pragma: no cover - IO
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save file") from exc

    # Compose a URL that can be used by the frontend to fetch the image
    # The application mounts the `static` directory at `/static`
    relative_url = f"/static/{target_path.relative_to(Path(settings.static_dir or 'static')).as_posix()}"

    user.profile_image_url = relative_url
    db.add(user)
    db.commit()
    db.refresh(user)

    return JSONResponse(status_code=status.HTTP_200_OK, content={"profile_image_url": relative_url})
