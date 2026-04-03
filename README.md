# Course-Based Attendance System

A web-based, role-based attendance platform with AI face recognition. The repository combines a FastAPI backend for academic operations and attendance APIs with desktop/CLI tools for dataset management, training, and live recognition.

## Project Overview

This project manages attendance by course, schedule, and session.

- Backend (`backend/`) provides authentication, RBAC, academic CRUD, scheduling, session lifecycle, face-frame attendance processing, and reporting.
- Desktop/CLI (`main.py`, `gui.py`, `capture_gui.py`) supports training, validation, recognition, and dataset tooling.
- AI modules (`face_recognition/`) handle detection, anti-spoofing, occlusion checks, and embedding-based recognition.

The current architecture uses a central scheduler service in the backend, faculty-aware data scoping, and automatic absent marking when sessions close.

## Features

### Core Attendance Features

- Course-based attendance using face recognition
- Session-bound attendance validation (only active sessions can accept frames)
- Status handling for `PRESENT`, `LATE`, and `ABSENT`
- Duplicate-attendance prevention per student/course/session
- Automatic absence generation when sessions end

### Scheduling and Session Logic

- Auto-creation of daily attendance sessions from course schedules
- One active session per course per day protection
- Backfill behavior for missed windows (creates and closes session, then marks absences)
- Overlap protection for class-level schedules
- Guard against scheduling the same course twice on the same day in the same department

### Role-Based Academic Management

- JWT authentication with access/refresh token flow
- Faculty-aware scoping for faculty users
- Role-protected APIs for faculties, departments, classes, courses, schedules, teachers, students, sessions, attendance, and reports
- Auto enrollment helpers for new students and new courses (matching class context)

### Face Recognition and AI Pipeline

- SCRFD-based face detection and alignment
- MiniFASNet anti-spoof support
- Occlusion checks before identity matching
- Face embedding recognizer as primary path
- Optional LBPH training/recognition path for desktop flow
- Processing timeout and confidence threshold controls in backend attendance processing

### Ops and Reliability

- Startup checks for DB readiness, schema compatibility, model files, and secret strength
- Health endpoints (`/health`, `/health/live`, `/health/ready`, `/health/scheduler`)
- Rate limiting for auth and frame submission endpoints
- Structured logging and CORS configuration
- Alembic migrations and seed tooling
- Dockerized backend runtime with readiness healthcheck

## System Roles

The platform currently models these roles in the system and workflows:

- **Academia**
  - Global academic governance.
  - Can register users and manage faculties.
  - Can manage course definitions and view reports/sessions.

- **HR**
  - Manages teacher records and teacher-user linking.
  - Read access where permitted by router guards.

- **Admission**
  - Manages student records.
  - Student creation supports automatic enrollment in matching courses.

- **Teacher**
  - Views active sessions and reports.
  - Submits attendance frames for active sessions.

- **Student**
  - Domain participant of attendance records and reports.
  - Student-facing self-service APIs/UI are not yet included in this repository.

Notes:

- Backend RBAC currently enforces `ACADEMIA`, `FACULTY`/`FACULTY_ADMIN` equivalence, `HR`, `ADMISSIONS`, and `TEACHER` roles.
- `FACULTY_ADMIN` is maintained as a compatibility alias through role-equivalence logic.

## Attendance Workflow (Auto-Scheduled Sessions)

The backend scheduler runs in a central polling loop and applies this workflow:

1. Load all course schedules for the current weekday.
2. For each schedule, calculate today start/end times.
3. If current time is inside the window and no session exists for that course/day, create an `ACTIVE` session.
4. If backend was offline and the window already passed, backfill a missed session, close it, and mark absences.
5. For active sessions that pass end time, close the session and mark absent records for enrolled students with no attendance record.

Frame submission flow (`POST /attendance/frame`):

1. Validate session existence, active status, and time window.
2. Decode base64 image and run recognition pipeline.
3. Enforce confidence and timeout thresholds.
4. Validate recognized student class and course enrollment.
5. Store attendance as `PRESENT` or `LATE` (based on grace period), or return duplicate/no-match responses.

## Tech Stack

- **Backend**: FastAPI, Uvicorn, SQLAlchemy 2.x, Alembic, Pydantic v2
- **Auth/Security**: JWT (`python-jose`), password hashing (`passlib` + `bcrypt`), router-level RBAC
- **Database**: SQLite (dev) and MySQL (configurable production target)
- **AI/ML & CV**: OpenCV, InsightFace/SCRFD, FaceNet embeddings (`facenet-pytorch`, `torch`), ONNX Runtime (MiniFASNet)
- **Desktop/Tooling**: Python CLI + GUI scripts, dataset validators and organizers
- **Testing**: Pytest-based backend tests
- **Containers**: Docker + Docker Compose

Frontend note:

- No dedicated web frontend application is currently committed in this repository; backend APIs are ready to be consumed by a separate web client.

## Project Structure

```text
attendance_system/
├── api/                            # Placeholder integration area (not active backend runtime)
├── attendance/                     # Attendance rule helpers for desktop flow
├── backend/
│   ├── app/
│   │   ├── core/                   # Config, security, logging, rate limit, startup checks
│   │   ├── db/                     # Models, session management, seed and reset utilities
│   │   ├── routers/                # Auth, faculties, departments, classes, courses, schedules,
│   │   │                           # teachers, students, sessions, attendance, reports
│   │   ├── schemas/                # Pydantic API schemas
│   │   ├── services/               # Face service, attendance service, scheduler
│   │   └── utils/
│   ├── alembic/                    # DB migrations
│   ├── tests/                      # Backend tests
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   └── alembic.ini
├── database/                       # SQL schema files and desktop DB helper
├── dataset/                        # Face image dataset organized by faculty/student
├── face_recognition/               # Detector, anti-spoof, recognizer, trainer, validators
├── logs/
├── models/                         # Trained models and mapping assets
├── utils/                          # Shared config/logging/TTS helpers
├── capture_gui.py
├── gui.py
├── main.py                         # Desktop CLI entry point
├── Dockerfile
├── docker-compose.yml
├── requirements.txt                # Desktop/AI dependencies
└── README.md
```

## Setup Instructions

### 1) Clone and create environment

```bash
git clone https://github.com/abdimalik2004/course-based-attendance-system.git
cd course-based-attendance-system/attendance_system
python -m venv .venv
```

### 2) Install dependencies

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r backend\requirements.txt
pip install -r backend\requirements-dev.txt
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
pip install -r backend/requirements.txt
pip install -r backend/requirements-dev.txt
```

### 3) Configure backend environment

Backend loads environment in this order:

1. `backend/.env`
2. `backend/.env.<APP_ENV>`

Common values to configure:

- `APP_ENV`, `SECRET_KEY`, `JWT_ALGORITHM`
- `DB_TYPE`, `SQLITE_PATH` or MySQL connection variables
- `SCHEDULER_POLL_SECONDS`, `DEFAULT_GRACE_PERIOD_MINUTES`
- `FACE_CONFIDENCE_THRESHOLD`, `FACE_TIMEOUT_SECONDS`
- `CORS_*`, `LOG_LEVEL`, and rate-limit settings

### 4) Run migrations and seed data

From `attendance_system/backend`:

```bash
python -m alembic upgrade head
python -m app.db.seed
```

### 5) Start backend API

From `attendance_system/backend`:

```bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Verify:

- Swagger: `http://127.0.0.1:8000/docs`
- Ready: `http://127.0.0.1:8000/health/ready`
- Scheduler health: `http://127.0.0.1:8000/health/scheduler`

### 6) Optional desktop workflows

From `attendance_system` root:

```bash
python main.py validate-dataset
python main.py train
python main.py recognize --auto-schedule --camera-index 0
```

Additional CLI utilities:

- `python main.py organize-dataset`
- `python main.py normalize-student-ids`

## API / Backend Notes

### Authentication

- `POST /auth/token` obtains access/refresh tokens
- `POST /auth/refresh` rotates token pair
- `GET /auth/me` returns current user
- `POST /auth/register` is restricted to Academia role

### Main Functional Endpoints

- Academic entities: `/faculties`, `/departments`, `/classes`, `/courses`, `/students`, `/teachers`
- Scheduling: `/schedules`
- Sessions: `/sessions`, `/sessions/active`
- Attendance ingestion: `POST /attendance/frame`
- Reports: `/reports/course/{course_id}` and breakdown endpoints

### Seeded Accounts

Current seed script creates:

- `academia / academia123`
- `facultyadmin / faculty123`
- `teacher1 / teacher123`
- `hr / hr123`
- `admission / admission123`

### Testing

From `attendance_system/backend`:

```bash
pytest -q
```

Or run focused suites:

```bash
pytest -q tests/test_attendance_logic.py tests/test_attendance_performance.py tests/test_api_permissions_and_scheduler.py tests/test_rbac_responsibilities.py
```

### Docker

From `attendance_system` root:

```bash
docker compose up --build
```

## Removed / Deprecated References Cleaned Up

The README now reflects current code and intentionally removes outdated references:

- Removed references to non-existent screenshot files under `screenshots/`
- Removed old role summary that omitted active `HR` and `ADMISSIONS` role usage
- Removed future-state statements that are already implemented (FastAPI backend, scheduler, reporting)
- Clarified that `api/` is currently a placeholder and active API runtime is `backend/`

## Known Issues / Future Improvements

- No in-repo web frontend currently; backend is API-first.
- Student-facing self-service portal and auth flow are not implemented yet.
- Scheduler runs by polling; event-driven scheduling could reduce latency and DB polling load.
- Expanded observability (metrics/tracing) and CI automation would improve production readiness.
