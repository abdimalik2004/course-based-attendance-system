from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.rate_limit import rate_limit_dependency
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_refresh_token,
    get_current_user,
    get_password_hash,
    pwd_context,
    require_roles,
    TokenPayloadError,
    verify_password,
)
from app.db.models import Role, User
from app.db.session import get_db
from app.db.reset_database import reset_database_to_clean_state
from app.schemas.auth import (
    RefreshTokenRequest,
    ResetDatabaseRequest,
    RoleCreate,
    RoleRead,
    TokenPair,
    UserCreate,
    UserRead,
)
from app.services.user_service import create_user


router = APIRouter(prefix="/auth", tags=["auth"])


@router.get(
    "/roles",
    response_model=list[RoleRead],
    dependencies=[Depends(require_roles("SUPER_ADMIN"))],
)
def list_roles(db: Session = Depends(get_db)):
    return db.query(Role).order_by(Role.name).all()


@router.post(
    "/roles",
    response_model=RoleRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_roles("SUPER_ADMIN"))],
    responses={
        400: {"description": "Invalid role name"},
        409: {"description": "Role already exists"},
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
    },
)
def create_role(payload: RoleCreate, db: Session = Depends(get_db)):
    name = payload.name.strip().upper()
    if not name:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Role name is required")

    role = Role(name=name)
    db.add(role)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Role already exists") from exc

    db.refresh(role)
    return role


@router.post(
    "/register",
    response_model=UserRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_roles("SUPER_ADMIN"))],
    responses={
        400: {"description": "Invalid input (duplicate username or invalid role)"},
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
    },
)
def register_user(payload: UserCreate, db: Session = Depends(get_db)):
    user = create_user(
        db,
        email=payload.email,
        username=payload.username,
        password=payload.password,
        role_names=payload.role_names,
    )
    db.commit()
    db.refresh(user)
    return user


@router.post(
    "/token",
    response_model=TokenPair,
    dependencies=[Depends(rate_limit_dependency(settings.auth_rate_limit_requests, settings.auth_rate_limit_window_seconds))],
    responses={401: {"description": "Incorrect username or password"}},
)
def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if pwd_context.needs_update(user.hashed_password):
        user.hashed_password = get_password_hash(form_data.password)
        db.add(user)
        db.commit()

    access_token = create_access_token(subject=user.username)
    refresh_token = create_refresh_token(subject=user.username)
    return TokenPair(access_token=access_token, refresh_token=refresh_token)


@router.post(
    "/refresh",
    response_model=TokenPair,
    dependencies=[Depends(rate_limit_dependency(settings.auth_rate_limit_requests, settings.auth_rate_limit_window_seconds))],
)
def refresh_tokens(payload: RefreshTokenRequest, db: Session = Depends(get_db)):
    try:
        username = decode_refresh_token(payload.refresh_token)
    except TokenPayloadError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token") from exc

    user = db.query(User).filter(User.username == username).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

    access_token = create_access_token(subject=user.username)
    refresh_token = create_refresh_token(subject=user.username)
    return TokenPair(access_token=access_token, refresh_token=refresh_token)


@router.get("/me", response_model=UserRead)
def read_current_user(user: User = Depends(get_current_user)):
    return user


@router.post(
    "/reset-database",
    dependencies=[Depends(require_roles("SUPER_ADMIN"))],
    responses={
        400: {"description": "Invalid confirmation value"},
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
    },
)
def reset_database(payload: ResetDatabaseRequest):
    if payload.confirmation.strip().upper() != "RESET":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Confirmation must be RESET")

    summary = reset_database_to_clean_state()
    return {"ok": True, "summary": summary}
