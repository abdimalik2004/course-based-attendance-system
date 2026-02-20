import argparse
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("GLOG_minloglevel", "2")

from attendance.mark_attendance import mark_attendance
from attendance.rules import (
    assess_course_time,
    build_auto_session_label,
    build_session_key,
    select_course_for_time,
)
from database.db import Database
from face_recognition.embedding_recognizer import FaceEmbeddingRecognizer
from face_recognition.recognize import FaceRecognizer
from face_recognition.train import train_embeddings_from_dataset, train_from_dataset
from face_recognition.validate_dataset import validate_dataset
from utils.config import load_config
from utils.logging import get_logger, setup_logging


def _parse_args():
    parser = argparse.ArgumentParser(description="Course-based attendance system")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_cmd = subparsers.add_parser("train", help="Train LBPH model from dataset")
    train_cmd.add_argument("--dataset", default=None, help="Dataset path override")

    subparsers.add_parser("validate-dataset", help="Validate dataset images for detectable faces")

    rec_cmd = subparsers.add_parser("recognize", help="Recognize faces and mark attendance")
    rec_cmd.add_argument("--course-id", required=False, help="Course ID for attendance")
    rec_cmd.add_argument("--session-label", default="default", help="Session label")
    rec_cmd.add_argument("--camera-index", type=int, default=None, help="Camera index override")
    rec_cmd.add_argument("--auto-schedule", action="store_true", help="Pick course based on current time")

    return parser.parse_args()


def run_training():
    config = load_config()
    setup_logging(config.log_file)
    if config.recognizer_type == "facenet":
        train_embeddings_from_dataset(config)
    else:
        train_from_dataset(config)


def run_validation():
    config = load_config()
    setup_logging(config.log_file)
    validate_dataset(config)


def run_recognition(course_id: str, session_label: str, camera_index=None, auto_schedule=False):
    config = load_config()
    setup_logging(config.log_file)
    logger = get_logger(__name__)

    db = Database(config)
    now = datetime.now()
    use_auto_schedule = auto_schedule or (config.auto_schedule and not course_id)
    course_name = None
    if use_auto_schedule:
        db.connect()
        if config.db_init_schema:
            db.init_schema()
        courses = db.get_courses()
        selected, shift_index = select_course_for_time(courses, now)
        if not selected:
            logger.info("No attendance time for now.")
            db.close()
            return
        course_id = selected["course_id"]
        course_name = selected.get("course_name")
        session_label = build_auto_session_label(shift_index, now)

    if not course_id:
        raise RuntimeError("Course ID is required when auto-schedule is disabled")

    if not use_auto_schedule:
        db.connect()
        if config.db_init_schema:
            db.init_schema()
        course = db.get_course(course_id)
        status = assess_course_time(course, now)
        course_name = course.get("course_name") if course else None
        course_display = course_name or course_id
        if status == "missing":
            logger.info("Course %s not found.", course_display)
            db.close()
            return
        if status == "no_schedule":
            logger.info("Course %s has no schedule configured.", course_display)
            db.close()
            return
        if status == "wrong_day":
            logger.info("Course %s is not scheduled for today.", course_display)
            db.close()
            return
        if status == "too_early":
            logger.info("Course %s has not started yet.", course_display)
            db.close()
            return
        if status == "too_late":
            session_key = build_session_key(course_id, session_label, now)
            absent_count = db.mark_absent_for_course_session(course_id, session_key)
            logger.info(
                "Marked %d absent records for course %s session=%s",
                absent_count,
                course_display,
                session_key,
            )
            logger.info("Course %s time has already passed.", course_display)
            db.close()
            return

    if config.recognizer_type == "facenet":
        recognizer = FaceEmbeddingRecognizer(config)
    else:
        recognizer = FaceRecognizer(config)
    recognizer.load_model()

    session_key = build_session_key(course_id, session_label, datetime.utcnow())

    def on_recognized(result):
        nonlocal course_name
        if not result["is_known"]:
            return
        if db.conn is None:
            db.connect()
            if config.db_init_schema:
                db.init_schema()
        student_id = result["student_id"]
        student = db.get_student(student_id) or {"name": "Unknown"}
        if course_name is None:
            course = db.get_course(course_id)
            course_name = course.get("course_name") if course else None
        course_display = course_name or course_id
        if not db.is_enrolled(student_id, course_id):
            logger.info(
                "Student ID: %s Name: %s Course: %s Confidence: %.2f Attendance: SKIPPED (not enrolled)",
                student_id,
                student["name"],
                course_display,
                result["confidence"],
            )
            return
        marked = mark_attendance(db, student_id, course_id, result["confidence"], session_key)
        if marked:
            logger.info(
                "Student ID: %s Name: %s Course: %s Confidence: %.2f Attendance: MARKED",
                student_id,
                student["name"],
                course_display,
                result["confidence"],
            )
        else:
            logger.info(
                "Student ID: %s Name: %s Course: %s Attendance: ALREADY MARKED",
                student_id,
                student["name"],
                course_display,
            )

    recognizer.run_webcam(course_id, on_recognized, camera_index=camera_index)

    db.close()


def main():
    args = _parse_args()
    if args.command == "train":
        run_training()
    elif args.command == "validate-dataset":
        run_validation()
    elif args.command == "recognize":
        run_recognition(
            args.course_id,
            args.session_label,
            camera_index=args.camera_index,
            auto_schedule=args.auto_schedule,
        )


if __name__ == "__main__":
    main()
