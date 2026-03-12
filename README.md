# Course-Based Attendance System

An AI-powered attendance platform that now includes both the original desktop recognition workflow and a FastAPI backend for production-style API access, scheduling, reporting, and role-based administration.

## Overview

This repository currently contains two working surfaces:

1. A desktop and CLI recognition pipeline for training, validating, and running face-based attendance.
2. A backend API that manages authentication, faculties, classes, teachers, students, courses, schedules, attendance sessions, frame submission, and attendance reports.

The recognition flow uses SCRFD for face detection and alignment, MiniFASNet anti-spoof checks, occlusion validation, and FaceNet embeddings for the primary recognition path. LBPH remains available as a lightweight fallback mode.

## Latest Additions

- FastAPI backend under `backend/` with modular routers and SQLAlchemy models
- JWT-based authentication with access and refresh tokens
- Role-based authorization for `ACADEMIA`, `FACULTY_ADMIN`, and `TEACHER`
- Automatic session scheduling from course schedules
- Attendance session lifecycle with `present`, `late`, and `absent` handling
- Duplicate-attendance prevention in API attendance processing
- Course reports by totals, date range, student breakdown, and session breakdown
- Health, liveness, and readiness endpoints for service monitoring
- Startup checks for database access, model files, and secret-key strength
- Structured logging, CORS configuration, and rate limiting
- Alembic migrations and seed data for backend setup
- Dockerfile and `docker-compose.yml` for backend container runs
- Backend tests covering permissions, scheduling, attendance logic, and timeout behavior

## System Architecture

```text
Camera Frame
  ↓
SCRFD Face Detection
  ↓
Face Alignment (5 keypoints)
  ↓
MiniFASNet Anti-Spoof
  ↓
Occlusion Check
  ↓
FaceNet / LBPH Recognition
  ↓
Attendance Validation
  ↓
Database Storage
```

The same recognition stack supports both the desktop flow and the backend attendance frame endpoint. In the backend path, recognized students are validated against active sessions, enrollments, and grace-period rules before records are stored.

## Key Features

- Course-aware face recognition attendance
- Multi-face detection and aligned face processing
- FaceNet embedding recognition with LBPH fallback
- MiniFASNet anti-spoof protection
- Occlusion and quality gating before recognition
- Auto-schedule support from timetable data
- Desktop CLI and GUI workflows
- FastAPI backend with OpenAPI docs
- JWT auth with refresh-token flow
- Role-based CRUD APIs for core academic entities
- Active-session discovery for teachers and admins
- Attendance reports for courses, students, sessions, and date ranges
- Dockerized backend deployment path

## Project Structure

```text
attendance_system/
├── api/                          # Legacy/future integration area
├── attendance/                   # Desktop attendance rules and marking logic
├── backend/                      # FastAPI backend, Alembic, tests, schemas
│   ├── app/
│   │   ├── core/                 # Config, security, rate limiting, startup checks
│   │   ├── db/                   # SQLAlchemy models, session, seed helpers
│   │   ├── routers/              # Auth, attendance, reports, academic CRUD APIs
│   │   ├── schemas/              # Pydantic request/response models
│   │   ├── services/             # Face service, attendance service, scheduler
│   │   └── utils/
│   ├── alembic/
│   └── tests/
├── database/                     # Desktop DB access layer + SQL schemas
├── dataset/                      # Student face image dataset
├── face_recognition/             # Detection, recognition, anti-spoof, occlusion
├── models/                       # Trained and inference models
├── utils/                        # Desktop config, logging, TTS helpers
├── capture_gui.py                # Student image enrollment GUI
├── gui.py                        # Desktop admin GUI
├── main.py                       # Desktop CLI entry point
├── Dockerfile                    # Backend container image
├── docker-compose.yml            # Backend container orchestration
├── requirements.txt              # Desktop/CLI dependencies
└── README.md
```

## Installation

```bash
git clone https://github.com/abdimalik2004/course-based-attendance-system.git
cd course-based-attendance-system/attendance_system
python -m venv .venv
```

Windows:

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

## Desktop Configuration

The desktop recognition flow uses environment variables prefixed with `ATTENDANCE_`.

Common groups:

- Database: `ATTENDANCE_DB_TYPE`, `ATTENDANCE_DB_HOST`, `ATTENDANCE_DB_PORT`, `ATTENDANCE_DB_NAME`, `ATTENDANCE_DB_USER`, `ATTENDANCE_DB_PASSWORD`
- Recognition: `ATTENDANCE_RECOGNIZER`, `ATTENDANCE_EMBEDDING_MIN_SIMILARITY`, `ATTENDANCE_CONFIDENCE_THRESHOLD`, `ATTENDANCE_MIN_FACE_SIZE`
- Anti-spoof: `ATTENDANCE_ANTI_SPOOF_ENABLED`, `ATTENDANCE_ANTI_SPOOF_MODEL_PATH`, `ATTENDANCE_ANTI_SPOOF_THRESHOLD`
- Occlusion: `ATTENDANCE_OCCLUSION_CHECK_ENABLED`, `ATTENDANCE_OCCLUSION_BACKEND`, `ATTENDANCE_OCCLUSION_MIN_EYES_VISIBLE`
- Quality: `ATTENDANCE_QUALITY_CHECK_ENABLED`, `ATTENDANCE_QUALITY_MIN_BLUR_VARIANCE`, `ATTENDANCE_QUALITY_MIN_BRIGHTNESS`, `ATTENDANCE_QUALITY_MAX_BRIGHTNESS`
- Detector: `ATTENDANCE_SCRFD_DET_SIZE`, `ATTENDANCE_SCRFD_THRESHOLD`, `ATTENDANCE_SCRFD_MAX_FACES`
- Scheduling and camera: `ATTENDANCE_AUTO_SCHEDULE`, `ATTENDANCE_CAMERA_WIDTH`, `ATTENDANCE_CAMERA_HEIGHT`, `ATTENDANCE_PREVIEW_WIDTH`, `ATTENDANCE_PREVIEW_HEIGHT`

Recommended tuning order:

1. Tune detector quality first.
2. Tune recognition strictness second.
3. Tune anti-spoof and occlusion thresholds last.

## Backend Configuration

The backend loads configuration in this order:

1. `backend/.env`
2. `backend/.env.<APP_ENV>`

Example profiles:

- `backend/.env.development`
- `backend/.env.production`

Important backend settings:

- App and auth: `APP_ENV`, `APP_NAME`, `SECRET_KEY`, `JWT_ALGORITHM`, `ACCESS_TOKEN_EXPIRE_MINUTES`, `REFRESH_TOKEN_EXPIRE_MINUTES`
- Database: `DB_TYPE`, `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DB`, `SQLITE_PATH`
- Scheduler: `SCHEDULER_POLL_SECONDS`, `DEFAULT_GRACE_PERIOD_MINUTES`
- Attendance service: `FACE_CONFIDENCE_THRESHOLD`, `FACE_TIMEOUT_SECONDS`
- API runtime: `CORS_ALLOW_ORIGINS`, `CORS_ALLOW_METHODS`, `CORS_ALLOW_HEADERS`, `CORS_ALLOW_CREDENTIALS`, `LOG_LEVEL`
- Rate limits: `AUTH_RATE_LIMIT_REQUESTS`, `AUTH_RATE_LIMIT_WINDOW_SECONDS`, `FRAME_RATE_LIMIT_REQUESTS`, `FRAME_RATE_LIMIT_WINDOW_SECONDS`

## Desktop Usage

Train model data:

```bash
python main.py train
```

Training output by mode:

- `facenet` builds `models/face_embeddings.npz`
- `lbph` builds `models/lbph_trainer.yml` and `models/label_map.json`

Validate dataset images:

```bash
python main.py validate-dataset
```

Run attendance recognition with auto-schedule:

```bash
python main.py recognize --auto-schedule --camera-index 0
```

Run attendance recognition with manual course and session:

```bash
python main.py recognize --course-id CSC101 --session-label lecture-1 --camera-index 0
```

Launch GUIs:

```bash
python gui.py
python capture_gui.py
```

## Backend Usage

From `attendance_system/backend/`:

Apply migrations:

```bash
python -m alembic upgrade head
```

Seed initial data and accounts:

```bash
python -m app.db.seed
```

Optional password-hash refresh for older seeded users:

```bash
python -m app.db.migrate_hashes
```

Run the API locally:

```bash
python -m uvicorn app.main:app --reload
```

OpenAPI docs:

- `http://localhost:8000/docs`

Health endpoints:

- `GET /health`
- `GET /health/live`
- `GET /health/ready`

## API Highlights

Authentication:

- `POST /auth/token`
- `POST /auth/refresh`
- `POST /auth/register`
- `GET /auth/me`

Academic management:

- `GET/POST/PUT/DELETE /faculties`
- `GET/POST/PUT/DELETE /classes`
- `GET/POST/PUT/DELETE /students`
- `GET/POST/PUT/DELETE /teachers`
- `GET/POST/PUT/DELETE /courses`
- `POST /courses/assign-teacher`
- `POST /courses/{course_id}/enroll/{student_id}`
- `GET/POST/PUT/DELETE /schedules`

Attendance and sessions:

- `GET /sessions`
- `GET /sessions/active`
- `POST /attendance/frame`

Reports:

- `GET /reports/course/{course_id}`
- `GET /reports/course/{course_id}/range`
- `GET /reports/course/{course_id}/students`
- `GET /reports/course/{course_id}/sessions`

## Seeded Accounts

The backend seed currently provides these users:

- `academia / academia123`
- `facultyadmin / faculty123`
- `teacher1 / teacher123`

## Testing

Run backend tests from `attendance_system/backend/`:

```bash
pytest -q tests/test_attendance_logic.py tests/test_attendance_performance.py tests/test_api_permissions_and_scheduler.py
```

These tests cover:

- Permission enforcement for role-protected endpoints
- Schedule overlap and teacher/faculty consistency rules
- Attendance duplicate prevention
- Present, late, and absent status behavior
- Scheduler idempotency and date rollover handling
- Frame processing timeout behavior

## Docker

From the project root:

```bash
docker compose up --build
```

This starts the backend container and exposes the API on port `8000` with a readiness health check against `GET /health/ready`.

## Models Used

- `models/anti_spoof_minifasnet.onnx` for anti-spoof inference
- `models/anti_spoof.json` for heuristic fallback coefficients
- `models/face_embeddings.npz` for FaceNet identity embeddings
- `models/lbph_trainer.yml` for LBPH recognizer fallback path
- `models/label_map.json` for label-to-student mapping

## Notes

- The backend expects required model files to exist at startup.
- SQLite is supported for development, and MySQL is supported through backend configuration.
- The desktop and backend layers currently coexist in the same repository and share the model assets under `models/`.

Then render them directly:

![Attendance System](screenshots/attendance_preview.png)
![Recognition View](screenshots/recognition_view.png)

## Future Development

- FastAPI backend for centralized attendance APIs
- Web dashboard (React) for reports/analytics
- Cloud deployment and centralized monitoring
- Mobile portal for students and instructors
- Notification integration (SMS/email)

## License

Academic Research Project - Zamzam University
