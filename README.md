# Course-Based Attendance System

An intelligent, course-aware attendance platform that combines modern face detection, anti-spoof security, occlusion checks, and deep recognition to mark attendance in real time.

## Overview

This project evolved from a prototype into a layered biometric pipeline. It now uses SCRFD for robust face detection and keypoints, applies occlusion and anti-spoof validation, then performs FaceNet embedding recognition before attendance is written to the database.

The system supports both CLI and GUI workflows, multi-student environments, and configurable thresholds via `.env` for different classrooms and camera setups.

## System Architecture

```text
Camera Frame
  ↓
SCRFD Face Detection
  ↓
Face Alignment (5 keypoints)
  ↓
MiniFASNet Anti-Spoof Model
  ↓
Occlusion Detection
  ↓
FaceNet Embedding Recognition
  ↓
Attendance Logic
  ↓
Database Storage
```

The pipeline first detects and aligns faces using SCRFD with 5-point landmarks. Each candidate face is then validated with anti-spoof and occlusion checks to reduce spoof or covered-face errors before final recognition. FaceNet embeddings are matched against enrolled student vectors, then attendance rules (`on_time`, `late`, `absent`) are applied and stored.

Architecture step summary:

- `Camera Frame`: Captures live frames from the selected webcam stream.
- `SCRFD Face Detection`: Detects one or more faces in real time.
- `Face Alignment (5 keypoints)`: Normalizes face orientation/position before downstream checks.
- `MiniFASNet Anti-Spoof`: Estimates whether the face is live vs spoofed (photo/video/screen).
- `Occlusion Detection`: Verifies face visibility (especially eye-region quality and coverage).
- `FaceNet Embedding Recognition`: Computes embeddings and matches identity using cosine similarity.
- `Attendance Logic`: Applies enrollment and course-time rules before marking attendance.
- `Database Storage`: Writes verified attendance records to MySQL/SQLite.

## Key Features

- Course-based automated attendance system
- SCRFD face detection with 5-point alignment
- FaceNet deep learning recognition with embedding index
- LBPH recognition path available as lightweight fallback
- MiniFASNet ONNX anti-spoof protection
- Occlusion detection (sunglasses/partial eye coverage handling)
- Multi-frame verification for stable anti-spoof decisions
- Configurable runtime thresholds using `.env`
- Auto session scheduling from course timetable
- Multi-face detection support in real time
- Desktop enrollment and attendance interfaces
- Modular architecture ready for API/web expansion

## Project Structure

```text
attendance_system/
├── api/                     # Future backend integration
├── attendance/              # Attendance rules and marking logic
├── database/                # DB access layer + SQL schemas
├── dataset/                 # Student face images (training input)
├── face_recognition/        # Detection, recognition, anti-spoof, occlusion
│   ├── anti_spoof.py
│   ├── detector.py
│   ├── embedding_recognizer.py
│   ├── occlusion.py
│   ├── recognize.py
│   ├── train.py
│   └── validate_dataset.py
├── models/                  # Trained and inference models
├── utils/                   # Config/logging/TTS helpers
├── capture_gui.py           # Student image enrollment GUI
├── gui.py                   # Desktop admin GUI
├── main.py                  # Main CLI entry point
├── requirements.txt
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
```

Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration (.env)

Important runtime behavior is controlled in `.env`. These are the core variables used most often during tuning:

| Variable                                | Description                                   |
| --------------------------------------- | --------------------------------------------- |
| `ATTENDANCE_RECOGNIZER`                 | Recognition mode: `facenet` or `lbph`         |
| `ATTENDANCE_EMBEDDING_MIN_SIMILARITY`   | Minimum FaceNet similarity to accept identity |
| `ATTENDANCE_ANTI_SPOOF_ENABLED`         | Enable/disable anti-spoof validation          |
| `ATTENDANCE_ANTI_SPOOF_THRESHOLD`       | Minimum liveness confidence threshold         |
| `ATTENDANCE_ANTI_SPOOF_REQUIRED_FRAMES` | Frame window size for stable spoof decision   |
| `ATTENDANCE_OCCLUSION_CHECK_ENABLED`    | Enable/disable occlusion validation           |
| `ATTENDANCE_OCCLUSION_BACKEND`          | Occlusion backend mode                        |
| `ATTENDANCE_MIN_FACE_SIZE`              | Minimum face size to process                  |
| `ATTENDANCE_SCRFD_DET_SIZE`             | SCRFD detector input size                     |
| `ATTENDANCE_SCRFD_THRESHOLD`            | SCRFD detection confidence threshold          |
| `ATTENDANCE_SCRFD_MAX_FACES`            | Max faces processed per frame                 |
| `ATTENDANCE_QUALITY_CHECK_ENABLED`      | Optional blur/brightness quality gate         |
| `ATTENDANCE_AUTO_SCHEDULE`              | Automatically pick active course/session      |

Recommended tuning approach:

- Start with detector quality (`ATTENDANCE_SCRFD_THRESHOLD`, `ATTENDANCE_MIN_FACE_SIZE`).
- Tune recognition strictness (`ATTENDANCE_EMBEDDING_MIN_SIMILARITY`).
- Then tune security gates (`ATTENDANCE_ANTI_SPOOF_*`, `ATTENDANCE_OCCLUSION_*`).
- Keep values camera-specific when moving between low-light and bright environments.

## Usage

Train model data:

```bash
python main.py train
```

Validate dataset images:

```bash
python main.py validate-dataset
```

Run attendance recognition (auto schedule):

```bash
python main.py recognize --auto-schedule --camera-index 1
```

Run attendance recognition (manual course/session):

```bash
python main.py recognize --course-id CSC101 --session-label lecture-1 --camera-index 0
```

Launch GUIs:

```bash
python gui.py
python capture_gui.py
```

## Models Used

- `models/anti_spoof_minifasnet.onnx` for anti-spoof inference
- `models/anti_spoof.json` for heuristic fallback coefficients
- `models/face_embeddings.npz` for FaceNet identity embeddings
- `models/lbph_trainer.yml` for LBPH recognizer fallback path
- `models/label_map.json` for label-to-student mapping

## Screenshots

Add your latest captures under `screenshots/`:

- `screenshots/attendance_preview.png`
- `screenshots/recognition_view.png`

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
