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
Occlusion Detection
  ↓
MiniFASNet Anti-Spoof Model
  ↓
FaceNet Embedding Recognition
  ↓
Attendance Logic
  ↓
Database Storage
```

The pipeline first detects and localizes faces using SCRFD. Each detected face is checked for visibility/occlusion and liveness to block spoof attacks (photo/video/screen). Only then does FaceNet matching run, and successful identities are processed by attendance rules (`on_time`, `late`, `absent`) and saved.

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

Important runtime behavior is controlled in `.env`.

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

Create a `screenshots/` folder and place your latest UI captures there, then render them in the README:

```markdown
![Attendance Preview](screenshots/attendance_gui.png)
![Recognition Preview](screenshots/recognition_preview.png)
```

## Future Development

- FastAPI backend for centralized attendance APIs
- Web dashboard (React) for reports/analytics
- Cloud deployment and centralized monitoring
- Mobile portal for students and instructors
- Notification integration (SMS/email)

## License

Academic Research Project - Zamzam University
