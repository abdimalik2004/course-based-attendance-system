# Course-Based Attendance System (Face Recognition)

A modular Python system that uses Haar Cascade face detection and LBPH face recognition to mark attendance per course session. The architecture is designed for future expansion to APIs, web or mobile clients, and cloud storage.

## Features

- Face detection with Haar Cascade
- Face recognition with LBPH (OpenCV contrib)
- Session-based attendance with duplicate prevention
- SQLite database by default, MySQL-ready abstraction
- Configurable via environment variables
- File logging for audit trails

## Dataset Structure

```
dataset/
  student_id/
    img1.jpg
    img2.jpg
```

Folder name is the student ID. Images are read dynamically during training.

## Setup

1. Create a virtual environment and install dependencies:

```
pip install -r requirements.txt
```

2. Create the database schema (auto-run on first launch):

- SQLite uses [attendance_system/database/schema_sqlite.sql](attendance_system/database/schema_sqlite.sql).
- MySQL uses [attendance_system/database/schema_mysql.sql](attendance_system/database/schema_mysql.sql).

3. Prepare the dataset in [attendance_system/dataset](attendance_system/dataset).

## Train

```
python main.py train
```

## Validate Dataset Images

```
python main.py validate-dataset
```

This reports which images did not yield a detectable face.

This writes the LBPH model to [attendance_system/models/lbph_trainer.yml](attendance_system/models/lbph_trainer.yml) and label mapping to `models/label_map.json`.

## Recognize and Mark Attendance

```
python main.py recognize --course-id CSC101 --session-label lecture-1
```

## GUI (Admin + Recognition)

```
python gui.py
```

## Configuration

Set environment variables to override defaults:

- `ATTENDANCE_DB_TYPE` (sqlite or mysql)
- `ATTENDANCE_DB_PATH` (for sqlite)
- `ATTENDANCE_DB_HOST`, `ATTENDANCE_DB_PORT`, `ATTENDANCE_DB_NAME`, `ATTENDANCE_DB_USER`, `ATTENDANCE_DB_PASSWORD`
- `ATTENDANCE_CONFIDENCE_THRESHOLD` (LBPH threshold; lower means stricter)
- `ATTENDANCE_RECOGNIZER` (`lbph` or `facenet`)
- `ATTENDANCE_EMBEDDING_MIN_SIMILARITY` (FaceNet similarity; higher means stricter)
- `ATTENDANCE_ANTI_SPOOF_ENABLED` (true/false)
- `ATTENDANCE_ANTI_SPOOF_THRESHOLD` (live score threshold)
- `ATTENDANCE_ANTI_SPOOF_REQUIRED_FRAMES` (stabilization window size)
- `ATTENDANCE_ANTI_SPOOF_MARGIN` (extra average score margin over threshold, default `0.0`)
- `ATTENDANCE_ANTI_SPOOF_MIN_PASS_RATIO` (fraction of window frames that must pass threshold, default `0.67`)
- `ATTENDANCE_CAMERA_INDEX`
- `ATTENDANCE_HAAR_CASCADE_PATH` (optional)
- `ATTENDANCE_AUTO_SCHEDULE` (true/false to auto-pick course by time)

## Course Scheduling

Courses can store schedule fields so the app can auto-pick by time:

- `start_time` and `end_time` use `HH:MM` (24h)
- `days` is a CSV list like `sat,sun,mon,tue,wed,thu`

## Notes

- The LBPH confidence is a distance measure; lower is a better match.
- Avoid storing raw face images outside the dataset; treat biometric data carefully.
