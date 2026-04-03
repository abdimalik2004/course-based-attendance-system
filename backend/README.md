# Course Attendance Backend

## Setup

```bash
# from backend/
/mnt/d/zust/.venv/Scripts/python.exe -m pip install -r requirements.txt
/mnt/d/zust/.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
/mnt/d/zust/.venv/Scripts/python.exe -m alembic upgrade head
/mnt/d/zust/.venv/Scripts/python.exe -m app.db.seed
/mnt/d/zust/.venv/Scripts/python.exe -m uvicorn app.main:app --reload
```

## Environment Profiles

Profile files:

- `.env.development`
- `.env.production`

Selection order at startup:

1. `backend/.env`
2. `backend/.env.<APP_ENV>`

Example:

```bash
export APP_ENV=development
/mnt/d/zust/.venv/Scripts/python.exe -m uvicorn app.main:app --reload
```

## Data Ownership

The central database is the source of truth for all application data:

- `faculties`
- `roles`
- `users`
- `user_role_links`

The same database also owns faculty-scoped operational tables:

- `departments`
- `class_batches`
- `teachers`
- `students`
- `courses`
- `course_assignments`
- `course_schedules`
- `enrollments`
- `attendance_sessions`
- `attendance_records`

Faculty delete and cleanup behavior:

- `DELETE /faculties/{faculty_id}` is strict and fails when related rows still exist.
- `DELETE /faculties/{faculty_id}?force=true` removes the faculty and its related academic rows in the central database.
- `GET /faculties/{faculty_id}/delete-preview` shows the rows that would be removed before forcing deletion.

## Operational Readiness

Included in backend startup:

- DB connectivity check (`SELECT 1`)
- Required AI model file checks:
  - `models/face_embeddings.npz`
  - `models/anti_spoof_minifasnet.onnx`
- Structured JSON logging
- CORS policy from environment variables
- Strong `SECRET_KEY` validation

Health endpoints:

- `GET /health` (basic process check)
- `GET /health/live` (liveness probe)
- `GET /health/ready` (readiness probe: DB, model files, scheduler)

OpenAPI docs:

- `http://localhost:8000/docs`

## Auth Flow

1. Obtain token pair:

- `POST /auth/token` with form fields `username`, `password`

Response:

```json
{
  "access_token": "...",
  "refresh_token": "...",
  "token_type": "bearer"
}
```

2. Use token:

- Header: `Authorization: Bearer <access_token>`

3. Refresh tokens when access token expires:

- `POST /auth/refresh`

Request:

```json
{
  "refresh_token": "..."
}
```

Response returns a new access token and refresh token pair.

## Frontend Auth Integration

Recommended client behavior:

1. On login success:

- Store `access_token` in memory.
- Store `refresh_token` in secure storage (for web, prefer httpOnly cookie if backend/frontend are same-site; otherwise use the safest available storage for your architecture).

2. On every API call:

- Send `Authorization: Bearer <access_token>`.

3. On `401` responses:

- Call `POST /auth/refresh` once using the current refresh token.
- If refresh succeeds, update stored token pair and retry the original request once.
- If refresh fails (`401`), clear auth state and redirect user to login.

4. Rate limiting awareness:

- `/auth/token`, `/auth/refresh`, and `/attendance/frame` are rate-limited.
- Handle `429 Too many requests` with exponential backoff and user-friendly messages.

Seeded users:

- `academia / academia123`
- `facultyadmin / faculty123`
- `teacher1 / teacher123`

Password hash cleanup for seeded users:

```bash
/mnt/d/zust/.venv/Scripts/python.exe -m app.db.migrate_hashes
```

Note: running seed again also refreshes seeded account hashes to the current scheme.

## Core Endpoints

- `POST /auth/register` (`ACADEMIA` only)
- `GET /faculties`, `POST /faculties`
- `GET /classes`, `POST /classes`
- `GET /students`, `POST /students`
- `GET /teachers`, `POST /teachers`
- `GET /courses`, `POST /courses`
- `POST /courses/assign-teacher`
- `POST /courses/{course_id}/enroll/{student_id}`
- `GET /schedules`, `POST /schedules`
- `GET /sessions`
- `POST /attendance/frame`
- `GET /reports/course/{course_id}`

## Attendance Frame Contract

`POST /attendance/frame`

Request body:

```json
{
  "session_id": 12,
  "image": "data:image/jpeg;base64,/9j/4AAQSk..."
}
```

Possible responses:

- Attendance recorded
- Attendance already marked
- Recognition timeout exceeded 2.0 seconds
- No valid face matched
- Session not active / not found

## Test Commands

```bash
/mnt/d/zust/.venv/Scripts/python.exe -m pytest -q tests/test_attendance_logic.py tests/test_attendance_performance.py tests/test_api_permissions_and_scheduler.py
```

## Container Run

From project root (`attendance_system/`):

```bash
docker compose up --build
```
