# Course Attendance Backend

## Setup

```bash
# from backend/
/mnt/d/zust/.venv/Scripts/python.exe -m pip install -r requirements.txt
/mnt/d/zust/.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
/mnt/d/zust/.venv/Scripts/python.exe -m alembic upgrade head
/mnt/d/zust/.venv/Scripts/python.exe -m app.db.provision_tenants
/mnt/d/zust/.venv/Scripts/python.exe -m app.db.sync_tenants
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

Tenant provisioning settings:

- `TENANT_DB_AUTO_PROVISION_ENABLED=true|false`
- `TENANT_DB_RUNTIME_ROUTING_ENABLED=true|false` (default `false` for safe rollout)
- `TENANT_DB_SCHEDULER_ENABLED=true|false` (default `false`; only used when runtime routing is enabled)
- `TENANT_DB_ALLOW_LEGACY_OPERATIONAL_SYNC=true|false` (default `false`; emergency-only override for full tenant backfill)
- `TENANT_DB_PREFIX=tenant_`
- `TENANT_DB_CHARSET=utf8mb4`
- `TENANT_DB_COLLATION=utf8mb4_unicode_ci`
- `MYSQL_ADMIN_DB=mysql`
- `SCHEDULER_TENANT_FAILURE_THRESHOLD=3` (tenant marked unhealthy after this many consecutive failures)
- `SCHEDULER_TENANT_STALE_SECONDS=180` (tenant marked unhealthy if no completed tick within this window while scheduler is running)

Runtime routing behavior:

- `TENANT_DB_RUNTIME_ROUTING_ENABLED=false`: all operational routes still use central DB.
- `TENANT_DB_RUNTIME_ROUTING_ENABLED=true`: faculty-admin/teacher operational routes resolve DB sessions through faculty tenant metadata.
- `TENANT_DB_SCHEDULER_ENABLED=true`: scheduler tick runs per provisioned tenant DB instead of central DB.

## Data Ownership

Central DB remains the source of truth for global/auth/platform tables:

- `faculties`
- `roles`
- `users`
- `user_role_links`

Tenant DBs own faculty-scoped operational tables:

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

Enforcement behavior:

- `ACADEMIA` continues to use central DB for platform-wide operations.
- Faculty-scoped operational routes use tenant DBs when `TENANT_DB_RUNTIME_ROUTING_ENABLED=true`.
- Faculty-scoped users do not fall back to central DB if tenant metadata is missing or unprovisioned; requests fail with `503` until tenant state is fixed.

Backfill tenants for existing faculties:

```bash
/mnt/d/zust/.venv/Scripts/python.exe -m app.db.provision_tenants
```

Re-check all faculties including already provisioned rows:

```bash
/mnt/d/zust/.venv/Scripts/python.exe -m app.db.provision_tenants --include-provisioned
```

Sync central faculty data into tenant DBs:

```bash
/mnt/d/zust/.venv/Scripts/python.exe -m app.db.sync_tenants
```

Sync behavior note:

- Default mode is now `metadata-only` (faculties/users/roles links).
- Operational table replacement is hard-disabled by default in tenant-first mode.

Legacy full replacement mode (temporary/emergency only):

```bash
export TENANT_DB_ALLOW_LEGACY_OPERATIONAL_SYNC=true
/mnt/d/zust/.venv/Scripts/python.exe -m app.db.sync_tenants --include-operational-tables
```

Sync only one faculty:

```bash
/mnt/d/zust/.venv/Scripts/python.exe -m app.db.sync_tenants --faculty-code ENG
```

Recommended rollout order for tenant mode:

1. Run migrations on central DB.
2. Provision tenant DBs.
3. Sync central faculty data into tenant DBs.
4. Enable `TENANT_DB_RUNTIME_ROUTING_ENABLED=true`.

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
- `GET /health/ready` (readiness probe: DB, model files, scheduler; returns `503` if tenant scheduler alert thresholds are breached)
- `GET /health/scheduler-tenants` (scheduler mode, per-tenant tick status, recent failures, unhealthy tenant summary)

## Tenant Scheduler Staging Soak

Enable tenant runtime routing and tenant scheduler in staging:

```bash
export APP_ENV=staging
export TENANT_DB_RUNTIME_ROUTING_ENABLED=true
export TENANT_DB_SCHEDULER_ENABLED=true
/mnt/d/zust/.venv/Scripts/python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Monitor scheduler status during soak:

```bash
curl -s http://localhost:8000/health/scheduler-tenants
```

Readiness now fails fast if the tenant scheduler is unhealthy. This is suitable for uptime monitoring and deployment gates:

```bash
curl -i http://localhost:8000/health/ready
```

Recommended soak validation suite:

```bash
/mnt/d/zust/.venv/Scripts/python.exe -m pytest tests/test_tenant_scheduler_execution.py tests/test_tenant_first_attendance_report_scope.py tests/test_tenant_first_schedule_writes.py tests/test_tenant_first_course_writes.py tests/test_tenant_first_student_writes.py tests/test_tenant_first_teacher_writes.py tests/test_tenant_first_structure_writes.py tests/test_tenant_provisioning.py tests/test_tenant_sync.py
```

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
