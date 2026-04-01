import threading
import time
import tkinter as tk
import sys
from pathlib import Path
from tkinter import messagebox, ttk

import cv2

from face_recognition.detector import FaceDetector
from utils.config import load_config
from utils.dataset_paths import (
    is_full_student_id,
    normalize_student_id,
    student_dataset_dir,
    student_id_example,
    student_id_matches_bucket,
)
from utils.logging import get_logger, setup_logging


logger = get_logger(__name__)


class CaptureGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Photo Capture")
        self.root.geometry("560x310")

        self.config = load_config()
        setup_logging(self.config.log_file)

        self.detector = FaceDetector(self.config)
        self.capture_thread = None
        self.stop_event = threading.Event()

        self.faculty_var = tk.StringVar()
        self.student_id_var = tk.StringVar()
        self.count_var = tk.StringVar(value="30")
        self.camera_var = tk.StringVar(value=str(self.config.camera_index))
        self.status_var = tk.StringVar(value="Ready")
        self.progress_var = tk.StringVar(value="0")
        self.hint_var = tk.StringVar(value="Select Faculty/Program, then enter full student id like 26CIS001")
        self.faculty_options = self._load_faculty_options()

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._last_hint_text = ""
        self._last_hint_time = 0.0

    def _build_ui(self):
        form = ttk.Frame(self.root)
        form.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        self._combo_row(form, 0, "Faculty/Program", self.faculty_var, self.faculty_options)
        self._row(form, 1, "Student ID", self.student_id_var)
        self._row(form, 2, "Photo Count", self.count_var)
        self._row(form, 3, "Camera Index", self.camera_var)

        btns = ttk.Frame(form)
        btns.grid(row=4, column=1, sticky=tk.W, pady=6)
        ttk.Button(btns, text="Start Capture", command=self._start_capture).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Stop", command=self._stop_capture).pack(side=tk.LEFT, padx=4)

        ttk.Label(form, text="Captured").grid(row=5, column=0, sticky=tk.W, padx=6, pady=6)
        ttk.Label(form, textvariable=self.progress_var).grid(row=5, column=1, sticky=tk.W, padx=6, pady=6)

        ttk.Label(form, textvariable=self.hint_var, wraplength=420, foreground="#555").grid(
            row=6,
            column=0,
            columnspan=2,
            sticky=tk.W,
            padx=6,
            pady=(2, 6),
        )
        ttk.Label(form, textvariable=self.status_var).grid(row=7, column=0, columnspan=2, sticky=tk.W, padx=6)

    def _row(self, parent, row_idx, label, var):
        ttk.Label(parent, text=label).grid(row=row_idx, column=0, sticky=tk.W, padx=6, pady=4)
        ttk.Entry(parent, textvariable=var, width=30).grid(row=row_idx, column=1, sticky=tk.W, padx=6, pady=4)

    def _combo_row(self, parent, row_idx, label, var, values):
        ttk.Label(parent, text=label).grid(row=row_idx, column=0, sticky=tk.W, padx=6, pady=4)
        combo = ttk.Combobox(parent, textvariable=var, values=values, width=27, state="normal")
        combo.grid(row=row_idx, column=1, sticky=tk.W, padx=6, pady=4)
        combo.bind("<<ComboboxSelected>>", lambda _event: self._on_faculty_selected())

    def _load_faculty_options(self):
        options: set[str] = set()

        # Prefer program prefixes from backend class batches, e.g. CIS from CIS2201.
        try:
            backend_dir = Path(__file__).resolve().parent / "backend"
            if str(backend_dir) not in sys.path:
                sys.path.insert(0, str(backend_dir))

            from app.db.models import ClassBatch, Faculty  # noqa: WPS433
            from app.db.session import SessionLocal  # noqa: WPS433
            from app.utils.student_numbering import _program_prefix  # noqa: WPS433

            db = SessionLocal()
            try:
                for faculty in db.query(Faculty).all():
                    code = normalize_student_id(faculty.code)
                    if code:
                        options.add(code)

                for class_batch in db.query(ClassBatch).all():
                    try:
                        prefix = _program_prefix("", class_batch.name)
                    except ValueError:
                        continue
                    if prefix:
                        options.add(prefix)
            finally:
                db.close()
        except Exception as exc:  # noqa: BLE001
            logger.info("Could not load faculty/program options from backend DB: %s", exc)

        dataset_root = Path(self.config.dataset_dir)
        if dataset_root.exists():
            for item in dataset_root.iterdir():
                if item.is_dir() and item.name.isalpha():
                    options.add(item.name.upper())

        ordered = sorted(options)
        if ordered:
            self.faculty_var.set(ordered[0])
            self._update_hint_for_faculty(ordered[0])
        return ordered

    def _on_faculty_selected(self):
        self._update_hint_for_faculty(self.faculty_var.get())

    def _update_hint_for_faculty(self, bucket):
        normalized = normalize_student_id(bucket)
        if normalized:
            self.hint_var.set(
                f"Enter full student id only, e.g. {student_id_example(normalized)}. Do not type {normalized}/{student_id_example(normalized)}."
            )

    def _set_status(self, text):
        self.root.after(0, lambda: self.status_var.set(text))

    def _set_progress(self, value):
        self.root.after(0, lambda: self.progress_var.set(str(value)))

    def _start_capture(self):
        if self.capture_thread and self.capture_thread.is_alive():
            self._set_status("Capture already running.")
            return

        selected_bucket = normalize_student_id(self.faculty_var.get())
        if not selected_bucket:
            messagebox.showwarning("Missing Data", "Faculty/Program is required.")
            return

        student_id = normalize_student_id(self.student_id_var.get())
        if not student_id:
            messagebox.showwarning("Missing Data", "Student ID is required.")
            return
        if "/" in student_id or "\\" in student_id:
            messagebox.showwarning("Invalid Data", "Enter only the full student id like 26CIS001, not a folder path.")
            return
        if not is_full_student_id(student_id):
            messagebox.showwarning("Invalid Data", f"Student ID must look like {student_id_example(selected_bucket)}.")
            return
        if not student_id_matches_bucket(student_id, selected_bucket):
            messagebox.showwarning(
                "Invalid Data",
                f"Selected Faculty/Program is {selected_bucket}, so student id must match it, e.g. {student_id_example(selected_bucket)}.",
            )
            return

        try:
            photo_count = int(self.count_var.get())
        except ValueError:
            messagebox.showwarning("Invalid Data", "Photo Count must be a number.")
            return

        if photo_count <= 0:
            messagebox.showwarning("Invalid Data", "Photo Count must be greater than 0.")
            return

        camera_index = self._safe_int(self.camera_var.get(), self.config.camera_index)
        dataset_dir = student_dataset_dir(self.config.dataset_dir, student_id)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        self.stop_event.clear()
        self._set_progress(0)
        self._set_status(f"Capturing {photo_count} photos for {student_id}...")
        self.root.update_idletasks()
        self.capture_thread = threading.Thread(
            target=self._run_capture,
            args=(camera_index, photo_count, dataset_dir),
            daemon=True,
        )
        self.capture_thread.start()

    def _stop_capture(self):
        self.stop_event.set()
        self._set_status("Stopping capture...")

    def _open_camera(self, camera_index):
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if cap.isOpened():
                return cap
        return cv2.VideoCapture(camera_index)

    def _apply_capture_resolution(self, cap):
        candidates = []
        if self.config.camera_width and self.config.camera_height:
            candidates.append((self.config.camera_width, self.config.camera_height))
        candidates.extend([(1280, 720), (640, 480)])

        for width, height in candidates:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            ok, _ = cap.read()
            if ok:
                break

        # Keep buffer shallow to reduce perceived lag when processing is slower than camera FPS.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def _show_error(self, title, message):
        self.root.after(0, lambda: messagebox.showerror(title, message))

    def _hint(self, text):
        now = time.monotonic()
        if text != self._last_hint_text or (now - self._last_hint_time) >= 0.7:
            self._last_hint_text = text
            self._last_hint_time = now
            self._set_status(text)

    def _run_capture(self, camera_index, photo_count, dataset_dir):
        cap = self._open_camera(camera_index)
        if not cap.isOpened():
            self._set_status("Camera not available.")
            self._show_error("Camera Error", "Camera not available. Check the index and permissions.")
            return

        self._apply_capture_resolution(cap)

        cv2.namedWindow("Capture", cv2.WINDOW_NORMAL)

        saved = 0
        try:
            while saved < photo_count and not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    self._set_status("Failed to read from camera.")
                    break
                gray, faces = self.detector.detect(frame)

                if len(faces) != 1:
                    label = "No face" if len(faces) == 0 else "Multiple faces"
                    color = (0, 165, 255) if len(faces) == 0 else (0, 0, 255)
                    if faces:
                        fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
                        cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), color, 2)
                    cv2.putText(
                        frame,
                        f"{label} - keep exactly one face in frame",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                    )
                    self._hint("Keep exactly one face visible.")
                    cv2.imshow("Capture", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue

                x, y, w, h = faces[0]
                x1, y1, x2, y2 = self._expand_box(x, y, w, h, frame.shape)
                face_img = gray[y1:y2, x1:x2]
                filename = dataset_dir / f"img_{saved + 1:03d}.jpg"
                cv2.imwrite(str(filename), face_img)
                saved += 1
                self._set_progress(saved)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
                cv2.putText(
                    frame,
                    f"Captured {saved}/{photo_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 200, 0),
                    2,
                )
                cv2.imshow("Capture", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                if saved >= photo_count:
                    break

                time.sleep(0.1)
        finally:
            cap.release()
            cv2.destroyAllWindows()

        if saved >= photo_count:
            self._set_status(f"Done. Saved {saved} photos to {dataset_dir}.")
        else:
            self._set_status(f"Stopped. Saved {saved} photos to {dataset_dir}.")

    def _safe_int(self, value, default):
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _expand_box(self, x, y, w, h, frame_shape):
        height, width = frame_shape[:2]
        pad_top = int(h * self.config.capture_pad_top)
        pad_bottom = int(h * self.config.capture_pad_bottom)
        pad_left = int(w * self.config.capture_pad_left)
        pad_right = int(w * self.config.capture_pad_right)

        x1 = max(0, x - pad_left)
        y1 = max(0, y - pad_top)
        x2 = min(width, x + w + pad_right)
        y2 = min(height, y + h + pad_bottom)
        return x1, y1, x2, y2

    def _on_close(self):
        self.stop_event.set()
        self.root.destroy()


def main():
    root = tk.Tk()
    CaptureGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
