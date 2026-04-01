from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from threading import Lock

from fastapi import HTTPException, Request


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
