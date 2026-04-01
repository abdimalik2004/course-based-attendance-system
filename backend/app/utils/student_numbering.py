from __future__ import annotations

import re
from pathlib import Path

from sqlalchemy.orm import Session

from app.db.models import ClassBatch, Faculty, Student


_NUMBER_RE = re.compile(r"^(?P<prefix>[A-Z0-9]+?)(?P<seq>\d+)$")
_LEADING_ALPHA_RE = re.compile(r"^(?P<alpha>[A-Z]+)")
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DATASET_ROOT = _PROJECT_ROOT / "dataset"


def year_prefix(year: int | None) -> str:
    if year is None:
        return ""
    return f"{int(year) % 100:02d}"


def _program_prefix(faculty_code: str, class_batch_name: str | None) -> str:
    if class_batch_name:
        match = _LEADING_ALPHA_RE.match(class_batch_name.strip().upper())
        if match:
            return match.group("alpha")

    faculty_prefix = faculty_code.strip().upper()
    if not faculty_prefix:
        raise ValueError("faculty code is required")
    return faculty_prefix


def build_student_number_prefix(faculty_code: str, class_year: int | None, class_batch_name: str | None = None) -> str:
    return f"{year_prefix(class_year)}{_program_prefix(faculty_code, class_batch_name)}"


def student_dataset_dir(student_number: str) -> Path:
    normalized = student_number.strip().upper()
    letters = "".join(ch for ch in normalized if ch.isalpha())
    bucket = letters[-3:] if len(letters) >= 3 else letters
    if bucket:
        return _DATASET_ROOT / bucket / normalized
    return _DATASET_ROOT / normalized


def _candidate_numbers_from_dataset(prefix: str) -> set[str]:
    values: set[str] = set()
    if not _DATASET_ROOT.exists():
        return values

    for path in _DATASET_ROOT.rglob("*"):
        if not path.is_dir():
            continue
        name = path.name.strip().upper()
        if name.startswith(prefix):
            values.add(name)
    return values


def _candidate_numbers_from_db(db: Session, prefix: str) -> set[str]:
    rows = db.query(Student.student_number).filter(Student.student_number.ilike(f"{prefix}%")).all()
    return {value.strip().upper() for (value,) in rows if value}


def next_available_student_number(db: Session, faculty_code: str, class_year: int | None, class_batch_name: str | None = None) -> str:
    prefix = build_student_number_prefix(faculty_code, class_year, class_batch_name)
    taken = _candidate_numbers_from_db(db, prefix) | _candidate_numbers_from_dataset(prefix)

    max_seq = 0
    for value in taken:
        match = _NUMBER_RE.match(value)
        if not match:
            continue
        if match.group("prefix") != prefix:
            continue
        max_seq = max(max_seq, int(match.group("seq")))

    return f"{prefix}{max_seq + 1:03d}"


def normalize_legacy_student_numbers(db: Session) -> list[tuple[str, str]]:
    students = (
        db.query(Student, Faculty, ClassBatch)
        .join(Faculty, Faculty.id == Student.faculty_id)
        .join(ClassBatch, ClassBatch.id == Student.class_batch_id)
        .order_by(Student.id)
        .all()
    )

    renamed: list[tuple[str, str]] = []
    reserved: set[str] = {student.student_number.strip().upper() for student, _, _ in students if student.student_number}
    reserved |= {
        path.name.strip().upper()
        for path in _DATASET_ROOT.rglob("*")
        if path.is_dir()
    }

    for student, faculty, class_batch in students:
        current = (student.student_number or "").strip().upper()
        year_pref = year_prefix(class_batch.year)
        prefix = build_student_number_prefix(faculty.code, class_batch.year, class_batch.name)
        if current.startswith(prefix) and current[len(prefix):].isdigit() and len(current[len(prefix):]) == 3:
            continue

        seq_match = re.search(r"(\d+)$", current)
        seq = int(seq_match.group(1)[-3:]) if seq_match else None
        proposed = f"{prefix}{seq:03d}" if seq is not None else None

        if not proposed or proposed in reserved:
            index = 1
            while True:
                candidate = f"{prefix}{index:03d}"
                if candidate not in reserved:
                    proposed = candidate
                    break
                index += 1

        old_dataset_dir = student_dataset_dir(current)
        new_dataset_dir = student_dataset_dir(proposed)
        if old_dataset_dir.exists() and old_dataset_dir != new_dataset_dir:
            new_dataset_dir.parent.mkdir(parents=True, exist_ok=True)
            old_dataset_dir.replace(new_dataset_dir)

        student.student_number = proposed
        if student.embedding_ref is None or student.embedding_ref.strip().upper() == current:
            student.embedding_ref = proposed

        reserved.discard(current)
        reserved.add(proposed)
        renamed.append((current, proposed))

    if renamed:
        db.commit()
    return renamed