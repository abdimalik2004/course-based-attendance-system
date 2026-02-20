import threading
import time
import tkinter as tk
import sys
from pathlib import Path
from tkinter import messagebox, ttk

import cv2

from face_recognition.detector import FaceDetector
from utils.config import load_config
from utils.logging import get_logger, setup_logging


logger = get_logger(__name__)


class CaptureGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Photo Capture")
        self.root.geometry("520x260")

        self.config = load_config()
        setup_logging(self.config.log_file)

        self.detector = FaceDetector(self.config.haar_cascade_path)
        self.capture_thread = None
        self.stop_event = threading.Event()

        self.student_id_var = tk.StringVar()
        self.count_var = tk.StringVar(value="30")
        self.camera_var = tk.StringVar(value=str(self.config.camera_index))
        self.status_var = tk.StringVar(value="Ready")
        self.progress_var = tk.StringVar(value="0")

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        form = ttk.Frame(self.root)
        form.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        self._row(form, 0, "Student ID", self.student_id_var)
        self._row(form, 1, "Photo Count", self.count_var)
        self._row(form, 2, "Camera Index", self.camera_var)

        btns = ttk.Frame(form)
        btns.grid(row=3, column=1, sticky=tk.W, pady=6)
        ttk.Button(btns, text="Start Capture", command=self._start_capture).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Stop", command=self._stop_capture).pack(side=tk.LEFT, padx=4)

        ttk.Label(form, text="Captured").grid(row=4, column=0, sticky=tk.W, padx=6, pady=6)
        ttk.Label(form, textvariable=self.progress_var).grid(row=4, column=1, sticky=tk.W, padx=6, pady=6)

        ttk.Label(form, textvariable=self.status_var).grid(row=5, column=0, columnspan=2, sticky=tk.W, padx=6)

    def _row(self, parent, row_idx, label, var):
        ttk.Label(parent, text=label).grid(row=row_idx, column=0, sticky=tk.W, padx=6, pady=4)
        ttk.Entry(parent, textvariable=var, width=30).grid(row=row_idx, column=1, sticky=tk.W, padx=6, pady=4)

    def _set_status(self, text):
        self.root.after(0, lambda: self.status_var.set(text))

    def _set_progress(self, value):
        self.root.after(0, lambda: self.progress_var.set(str(value)))

    def _start_capture(self):
        if self.capture_thread and self.capture_thread.is_alive():
            self._set_status("Capture already running.")
            return

        student_id = self.student_id_var.get().strip()
        if not student_id:
            messagebox.showwarning("Missing Data", "Student ID is required.")
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
        dataset_dir = Path(self.config.dataset_dir) / student_id
        dataset_dir.mkdir(parents=True, exist_ok=True)

        self.stop_event.clear()
        self._set_progress(0)
        self._set_status(f"Capturing {photo_count} photos for {student_id}...")
        self.root.update_idletasks()

        self._run_capture(camera_index, photo_count, dataset_dir)

    def _stop_capture(self):
        self.stop_event.set()
        self._set_status("Stopping capture...")

    def _open_camera(self, camera_index):
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            if cap.isOpened():
                return cap
        return cv2.VideoCapture(camera_index)

    def _run_capture(self, camera_index, photo_count, dataset_dir):
        cap = self._open_camera(camera_index)
        if not cap.isOpened():
            self._set_status("Camera not available.")
            messagebox.showerror("Camera Error", "Camera not available. Check the index and permissions.")
            return

        cv2.namedWindow("Capture", cv2.WINDOW_NORMAL)

        saved = 0
        try:
            while saved < photo_count and not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    self._set_status("Failed to read from camera.")
                    break

                cv2.imshow("Capture", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                gray, faces = self.detector.detect(frame)
                if len(faces) == 0:
                    self.root.update_idletasks()
                    self.root.update()
                    continue

                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                x1, y1, x2, y2 = self._expand_box(x, y, w, h, frame.shape)
                face_img = gray[y1:y2, x1:x2]
                filename = dataset_dir / f"img_{saved + 1:03d}.jpg"
                cv2.imwrite(str(filename), face_img)
                saved += 1
                self._set_progress(saved)

                if saved >= photo_count:
                    break

                time.sleep(0.1)
                self.root.update_idletasks()
                self.root.update()
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
