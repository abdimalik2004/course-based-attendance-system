from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.rate_limit import rate_limit_dependency
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_refresh_token,
    get_current_user,
    get_password_hash,
    require_roles,
    TokenPayloadError,
    verify_password,
)
from app.db.models import Role, User
from app.db.session import get_db
from app.db.reset_database import reset_database_to_clean_state
from app.schemas.auth import RefreshTokenRequest, ResetDatabaseRequest, TokenPair, UserCreate, UserRead


router = APIRouter(prefix="/auth", tags=["auth"])


@router.post(
    "/register",
    response_model=UserRead,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_roles("ACADEMIA"))],
    responses={
        400: {"description": "Invalid input (duplicate username or invalid role)"},
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
    },
)
def register_user(payload: UserCreate, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.username == payload.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")

    user = User(
        username=payload.username,
        email=payload.email,
        hashed_password=get_password_hash(payload.password),
        faculty_id=payload.faculty_id,
    )
    db.add(user)
    db.flush()

    if payload.role_names:
        roles = db.query(Role).filter(Role.name.in_(payload.role_names)).all()
        if len(roles) != len(set(payload.role_names)):
            raise HTTPException(status_code=400, detail="One or more role names are invalid")
        user.roles = roles

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

    # Migrate legacy hashes (e.g., bcrypt) to the preferred current scheme.
    if not user.hashed_password.startswith("$pbkdf2-sha256$"):
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
    dependencies=[Depends(require_roles("ACADEMIA"))],
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
