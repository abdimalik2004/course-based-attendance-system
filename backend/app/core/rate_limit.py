from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from threading import Lock

from fastapi import HTTPException, Request


class LoginAttemptTracker:
    """Per-username consecutive-failure tracker with a timed lockout."""

    MAX_ATTEMPTS: int = 3
    LOCKOUT_SECONDS: int = 30

    def __init__(self) -> None:
        self._attempts: dict[str, int] = defaultdict(int)
        self._locked_until: dict[str, datetime] = {}
        self._lock = Lock()

    def _remaining(self, username: str, now: datetime) -> int:
        """Return remaining lockout seconds (0 means not locked)."""
        until = self._locked_until.get(username)
        if until is None:
            return 0
        secs = (until - now).total_seconds()
        if secs <= 0:
            # Lockout expired — clean up
            del self._locked_until[username]
            self._attempts[username] = 0
            return 0
        return int(secs) + 1  # round up so UI never shows "0s remaining"

    def check_locked(self, username: str) -> None:
        """Raise 429 if the username is currently locked out."""
        now = datetime.now(timezone.utc)
        with self._lock:
            remaining = self._remaining(username, now)
        if remaining > 0:
            raise HTTPException(
                status_code=429,
                detail={
                    "message": f"Too many failed attempts. Please wait {remaining} seconds.",
                    "retry_after": remaining,
                },
            )

    def record_failure(self, username: str) -> None:
        """Increment failure counter; start lockout if threshold is reached."""
        now = datetime.now(timezone.utc)
        with self._lock:
            # Re-check — another thread may have just set the lock
            remaining = self._remaining(username, now)
            if remaining > 0:
                raise HTTPException(
                    status_code=429,
                    detail={
                        "message": f"Too many failed attempts. Please wait {remaining} seconds.",
                        "retry_after": remaining,
                    },
                )

            self._attempts[username] += 1

            if self._attempts[username] >= self.MAX_ATTEMPTS:
                self._locked_until[username] = now + timedelta(seconds=self.LOCKOUT_SECONDS)
                self._attempts[username] = 0
                raise HTTPException(
                    status_code=429,
                    detail={
                        "message": f"Too many failed attempts. Please wait {self.LOCKOUT_SECONDS} seconds.",
                        "retry_after": self.LOCKOUT_SECONDS,
                    },
                )

    def record_success(self, username: str) -> None:
        """Clear failure state after a successful login."""
        with self._lock:
            self._attempts.pop(username, None)
            self._locked_until.pop(username, None)


login_attempt_tracker = LoginAttemptTracker()


class InMemoryRateLimiter:
    def __init__(self) -> None:
        self._events: dict[str, deque[datetime]] = defaultdict(deque)
        self._lock = Lock()

    def enforce(self, *, key: str, max_requests: int, window_seconds: int) -> None:
        now = datetime.now(timezone.utc)
        window_start = now - timedelta(seconds=window_seconds)
        with self._lock:
            q = self._events[key]
            while q and q[0] < window_start:
                q.popleft()
            if len(q) >= max_requests:
                raise HTTPException(status_code=429, detail="Too many requests")
            q.append(now)


rate_limiter = InMemoryRateLimiter()


def rate_limit_dependency(max_requests: int, window_seconds: int):
    def _dep(request: Request):
        client_ip = request.client.host if request.client else "unknown"
        key = f"{request.url.path}:{client_ip}"
        rate_limiter.enforce(key=key, max_requests=max_requests, window_seconds=window_seconds)

    return _dep
