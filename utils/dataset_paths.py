from __future__ import annotations

import re
from pathlib import Path


_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
_FULL_STUDENT_ID_RE = re.compile(r"^(?P<year>\d{2})(?P<bucket>[A-Z]{2,})(?P<seq>\d{3})$")


def normalize_student_id(student_id: str) -> str:
    return student_id.strip().upper()


def faculty_bucket_from_student_id(student_id: str) -> str | None:
    normalized = normalize_student_id(student_id)
    matches = re.findall(r"[A-Z]+", normalized)
    if not matches:
        return None

    # Prefer the longest alpha token, so values like 26CIS006 map to CIS.
    bucket = max(matches, key=len)
    return bucket if len(bucket) >= 2 else None


def student_dataset_dir(dataset_root: str | Path, student_id: str) -> Path:
    root = Path(dataset_root)
    normalized = normalize_student_id(student_id)
    bucket = faculty_bucket_from_student_id(normalized)
    if bucket:
        return root / bucket / normalized
    return root / normalized


def is_full_student_id(student_id: str) -> bool:
    return bool(_FULL_STUDENT_ID_RE.fullmatch(normalize_student_id(student_id)))


def student_id_matches_bucket(student_id: str, bucket: str) -> bool:
    normalized = normalize_student_id(student_id)
    match = _FULL_STUDENT_ID_RE.fullmatch(normalized)
    if not match:
        return False
    return match.group("bucket") == normalize_student_id(bucket)


def student_id_example(bucket: str) -> str:
    return f"26{normalize_student_id(bucket)}001"


def iter_student_dataset_dirs(dataset_root: str | Path) -> list[Path]:
    root = Path(dataset_root)
    if not root.exists():
        return []

    student_dirs: list[Path] = []
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_dir():
            continue
        if candidate == root:
            continue
        if any(path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES for path in candidate.iterdir()):
            student_dirs.append(candidate)
    return student_dirs


def reorganize_dataset_layout(dataset_root: str | Path) -> list[tuple[Path, Path]]:
    root = Path(dataset_root)
    if not root.exists():
        return []

    moved: list[tuple[Path, Path]] = []
    for candidate in sorted(root.iterdir()):
        if not candidate.is_dir():
            continue

        target_dir = student_dataset_dir(root, candidate.name)
        if candidate == target_dir:
            continue
        if target_dir.is_relative_to(candidate):
            continue

        if target_dir.exists():
            target_dir.mkdir(parents=True, exist_ok=True)
            for child in candidate.iterdir():
                destination = target_dir / child.name
                if destination.exists():
                    continue
                child.replace(destination)
            try:
                candidate.rmdir()
            except OSError:
                pass
        else:
            target_dir.parent.mkdir(parents=True, exist_ok=True)
            candidate.replace(target_dir)
        moved.append((candidate, target_dir))

    return moved