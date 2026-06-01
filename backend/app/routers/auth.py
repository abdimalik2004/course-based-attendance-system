from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.rate_limit import login_attempt_tracker, rate_limit_dependency
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
    ChangePasswordRequest,
    ResetDatabaseRequest,
    RoleCreate,
    RoleRead,
    Token,
    UserCreate,
    UserRead,
)
from app.services.user_service import create_user
from app.utils.activity_logger import log_activity


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
def register_user(payload: UserCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    normalized_roles = {role_name.strip().upper() for role_name in payload.role_names if role_name.strip()}
    faculty_id = payload.faculty_id
    if "FACULTY" in normalized_roles:
        if faculty_id is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="faculty_id is required when role_names includes FACULTY",
            )
    else:
        faculty_id = None

    user = create_user(
        db,
        email=payload.email,
        username=payload.username,
        password=payload.password,
        role_names=payload.role_names,
        faculty_id=faculty_id,
        teacher_id=payload.teacher_id,
        student_id=payload.student_id,
    )
    db.commit()
    db.refresh(user)

    # Log user creation
    roles_str = ", ".join(payload.role_names) if payload.role_names else "No Role"
    log_activity(
        action=f"User Registered - {user.username} ({roles_str})",
        user=current_user,
        status="Success",
        db=db,
    )
    db.commit()

    return user


@router.post(
    "/token",
    response_model=Token,
    dependencies=[Depends(rate_limit_dependency(settings.auth_rate_limit_requests, settings.auth_rate_limit_window_seconds))],
    responses={401: {"description": "Incorrect username or password"}},
)
def login_for_access_token(
    request: Request,
    response: Response,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    # Reject immediately if this username is currently locked out
    login_attempt_tracker.check_locked(form_data.username)

    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        log_activity(
            action=f"Login Failed - {form_data.username}",
            username=form_data.username,
            status="Failed",
            db=db,
        )
        db.commit()
        # Count the failure — raises 429 automatically on the 3rd consecutive miss
        login_attempt_tracker.record_failure(form_data.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Successful login — reset failure counter
    login_attempt_tracker.record_success(form_data.username)

    if pwd_context.needs_update(user.hashed_password):
        user.hashed_password = get_password_hash(form_data.password)
        db.add(user)

    role_names = ", ".join(r.name for r in user.roles) if user.roles else "No Role"
    log_activity(
        action=f"User Login - {user.username} ({role_names})",
        user=user,
        status="Success",
        db=db,
    )
    db.commit()

    access_token = create_access_token(subject=user.username)
    refresh_token = create_refresh_token(subject=user.username)

    # Set refresh token as httpOnly cookie. Cookie security depends on environment.
    secure = settings.app_env == "production" and request.url.scheme == "https"
    max_age = settings.refresh_token_expire_minutes * 60
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=secure,
        samesite="lax",
        max_age=max_age,
        path="/",
    )
    return Token(access_token=access_token)


@router.post(
    "/refresh",
    response_model=Token,
    dependencies=[Depends(rate_limit_dependency(settings.auth_rate_limit_requests, settings.auth_rate_limit_window_seconds))],
)
def refresh_tokens(request: Request, response: Response, db: Session = Depends(get_db)):
    # Read refresh token from httpOnly cookie
    cookie_token = request.cookies.get("refresh_token")
    if not cookie_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing refresh token")

    try:
        username = decode_refresh_token(cookie_token)
    except TokenPayloadError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token") from exc

    user = db.query(User).filter(User.username == username).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

    access_token = create_access_token(subject=user.username)
    refresh_token = create_refresh_token(subject=user.username)

    secure = settings.app_env == "production" and request.url.scheme == "https"
    max_age = settings.refresh_token_expire_minutes * 60
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=secure,
        samesite="lax",
        max_age=max_age,
        path="/",
    )

    return Token(access_token=access_token)


@router.post("/logout")
def logout(response: Response, current_user: User = Depends(get_current_user)):
    # Log the logout action
    log_activity(
        action=f"User Logout - {current_user.username}",
        user=current_user,
        status="Success",
    )
    # Clear the refresh token cookie
    response.delete_cookie("refresh_token", path="/")
    return {"ok": True}


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


@router.post(
    "/change-password",
    responses={
        400: {"description": "New password same as current or invalid"},
        401: {"description": "Current password is incorrect"},
    },
)
def change_password(
    payload: ChangePasswordRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not verify_password(payload.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Current password is incorrect",
        )

    if payload.current_password == payload.new_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be different from current password",
        )

    current_user.hashed_password = get_password_hash(payload.new_password)
    db.commit()
    return {"ok": True, "message": "Password updated successfully"}
