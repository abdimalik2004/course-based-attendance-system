from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from passlib.exc import UnknownHashError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.models import User
from app.db.session import get_db


pwd_context = CryptContext(schemes=["bcrypt", "pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")


ROLE_EQUIVALENTS: dict[str, set[str]] = {
    "FACULTY": {"FACULTY", "FACULTY_ADMIN"},
    "FACULTY_ADMIN": {"FACULTY", "FACULTY_ADMIN"},
}

SUPER_ADMIN_ROLE = "SUPER_ADMIN"


class TokenPayloadError(Exception):
    pass


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except (UnknownHashError, ValueError, TypeError):
        return False


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(subject: str) -> str:
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=settings.access_token_expire_minutes)
    payload = {
        "sub": subject,
        "type": "access",
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


def create_refresh_token(subject: str) -> str:
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=settings.refresh_token_expire_minutes)
    payload = {
        "sub": subject,
        "type": "refresh",
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> str:
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        raise TokenPayloadError("Invalid token") from exc
    username = payload.get("sub")
    if not username:
        raise TokenPayloadError("Token missing subject")
    return username


def decode_refresh_token(token: str) -> str:
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        raise TokenPayloadError("Invalid refresh token") from exc

    if payload.get("type") != "refresh":
        raise TokenPayloadError("Invalid token type")

    username = payload.get("sub")
    if not username:
        raise TokenPayloadError("Refresh token missing subject")
    return username


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        username = decode_token(token)
    except TokenPayloadError as exc:
        raise credentials_exception from exc

    user = db.query(User).filter(User.username == username).first()
    if not user or not user.is_active:
        raise credentials_exception
    return user


def require_roles(*required_roles: str):
    def role_dependency(user: User = Depends(get_current_user)) -> User:
        assigned = {role.name for role in user.roles}
        if SUPER_ADMIN_ROLE in assigned:
            return user

        allowed: set[str] = set()
        for role_name in required_roles:
            allowed.update(ROLE_EQUIVALENTS.get(role_name, {role_name}))

        if not assigned.intersection(allowed):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role required: {', '.join(required_roles)}",
            )
        return user

    return role_dependency
