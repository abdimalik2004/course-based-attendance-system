import tkinter as tk
from tkinter import messagebox, ttk

from database.db import Database
from utils.config import load_config
from utils.logging import get_logger, setup_logging


logger = get_logger(__name__)


class AttendanceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Attendance System")
        self.root.geometry("820x520")

        self.config = load_config()
        setup_logging(self.config.log_file)

        self.db = Database(self.config)
        self.db.connect()
        if self.config.db_init_schema:
            self.db.init_schema()

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        admin_tab = ttk.Frame(notebook)
        notebook.add(admin_tab, text="Admin")

        self._build_admin(admin_tab)

    def _build_admin(self, parent):
        admin_tabs = ttk.Notebook(parent)
        admin_tabs.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        students_tab = ttk.Frame(admin_tabs)
        courses_tab = ttk.Frame(admin_tabs)
        enroll_tab = ttk.Frame(admin_tabs)
        admin_tabs.add(students_tab, text="Students")
        admin_tabs.add(courses_tab, text="Courses")
        admin_tabs.add(enroll_tab, text="Enrollments")

        self._build_students_tab(students_tab)
        self._build_courses_tab(courses_tab)
        self._build_enrollments_tab(enroll_tab)

        self._admin_status = tk.StringVar(value="Ready")
        ttk.Label(parent, textvariable=self._admin_status).pack(anchor=tk.W, padx=12, pady=6)

        self._refresh_students()
        self._refresh_courses()
        self._refresh_enrollments()

    def _build_students_tab(self, parent):
        form = ttk.LabelFrame(parent, text="Student Details")
        form.pack(fill=tk.X, padx=6, pady=6)

        self.student_id_var = tk.StringVar()
        self.student_name_var = tk.StringVar()
        self.student_dept_var = tk.StringVar()
        self.student_image_var = tk.StringVar()

        self._row(form, 0, "Name", self.student_name_var)
        self._row(form, 1, "Department", self.student_dept_var)
        self._row(form, 2, "Image Path", self.student_image_var)

        btns = ttk.Frame(form)
        btns.grid(row=3, column=1, sticky=tk.W, pady=6)
        ttk.Button(btns, text="Add", command=self._add_student).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Update", command=self._update_student).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Delete", command=self._delete_student).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Clear", command=self._clear_student_form).pack(side=tk.LEFT, padx=4)

        table_frame = ttk.LabelFrame(parent, text="Students")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        columns = ("id", "name", "department", "image_path")
        self.students_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=8)
        self.students_tree.heading("id", text="ID")
        self.students_tree.heading("name", text="Name")
        self.students_tree.heading("department", text="Department")
        self.students_tree.heading("image_path", text="Image Path")
        self.students_tree.column("id", width=120)
        self.students_tree.column("name", width=200)
        self.students_tree.column("department", width=160)
        self.students_tree.column("image_path", width=260)
        self.students_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.students_tree.yview)
        self.students_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.students_tree.bind("<<TreeviewSelect>>", self._on_student_select)
        ttk.Button(parent, text="Refresh", command=self._refresh_students).pack(anchor=tk.W, padx=8, pady=4)

    def _build_courses_tab(self, parent):
        form = ttk.LabelFrame(parent, text="Course Details")
        form.pack(fill=tk.X, padx=6, pady=6)

        self.course_id_var = tk.StringVar()
        self.course_name_var = tk.StringVar()
        self.course_start_var = tk.StringVar()
        self.course_end_var = tk.StringVar()
        self.course_days_var = tk.StringVar(value="sat,sun,mon,tue,wed,thu")

        self._row(form, 0, "Course Name", self.course_name_var)
        self._row(form, 1, "Start Time (HH:MM)", self.course_start_var)
        self._row(form, 2, "End Time (HH:MM)", self.course_end_var)
        self._row(form, 3, "Days (CSV)", self.course_days_var)

        btns = ttk.Frame(form)
        btns.grid(row=4, column=1, sticky=tk.W, pady=6)
        ttk.Button(btns, text="Add", command=self._add_course).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Update", command=self._update_course).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Delete", command=self._delete_course).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Clear", command=self._clear_course_form).pack(side=tk.LEFT, padx=4)

        table_frame = ttk.LabelFrame(parent, text="Courses")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        columns = ("course_id", "course_name", "start_time", "end_time", "days")
        self.courses_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=8)
        self.courses_tree.heading("course_id", text="ID")
        self.courses_tree.heading("course_name", text="Name")
        self.courses_tree.heading("start_time", text="Start")
        self.courses_tree.heading("end_time", text="End")
        self.courses_tree.heading("days", text="Days")
        self.courses_tree.column("course_id", width=120)
        self.courses_tree.column("course_name", width=200)
        self.courses_tree.column("start_time", width=80)
        self.courses_tree.column("end_time", width=80)
        self.courses_tree.column("days", width=200)
        self.courses_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.courses_tree.yview)
        self.courses_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.courses_tree.bind("<<TreeviewSelect>>", self._on_course_select)
        ttk.Button(parent, text="Refresh", command=self._refresh_courses).pack(anchor=tk.W, padx=8, pady=4)

    def _build_enrollments_tab(self, parent):
        form = ttk.LabelFrame(parent, text="Enrollment")
        form.pack(fill=tk.X, padx=6, pady=6)

        self.enroll_student_var = tk.StringVar()
        self.enroll_course_var = tk.StringVar()
        self._row(form, 0, "Student ID", self.enroll_student_var)
        self._row(form, 1, "Course ID", self.enroll_course_var)

        btns = ttk.Frame(form)
        btns.grid(row=2, column=1, sticky=tk.W, pady=6)
        ttk.Button(btns, text="Enroll", command=self._enroll_student).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Remove", command=self._remove_enrollment).pack(side=tk.LEFT, padx=4)
        ttk.Button(btns, text="Clear", command=self._clear_enroll_form).pack(side=tk.LEFT, padx=4)

        table_frame = ttk.LabelFrame(parent, text="Enrollments")
        table_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        columns = ("student_id", "course_id")
        self.enrollments_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=8)
        self.enrollments_tree.heading("student_id", text="Student ID")
        self.enrollments_tree.heading("course_id", text="Course ID")
        self.enrollments_tree.column("student_id", width=160)
        self.enrollments_tree.column("course_id", width=160)
        self.enrollments_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.enrollments_tree.yview)
        self.enrollments_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.enrollments_tree.bind("<<TreeviewSelect>>", self._on_enrollment_select)
        ttk.Button(parent, text="Refresh", command=self._refresh_enrollments).pack(anchor=tk.W, padx=8, pady=4)

    def _row(self, parent, row_idx, label, var):
        ttk.Label(parent, text=label).grid(row=row_idx, column=0, sticky=tk.W, padx=6, pady=4)
        ttk.Entry(parent, textvariable=var, width=40).grid(row=row_idx, column=1, sticky=tk.W, padx=6, pady=4)

    def _add_student(self):
        student_id = self.student_id_var.get().strip()
        name = self.student_name_var.get().strip()
        department = self.student_dept_var.get().strip() or None
        image_path = self.student_image_var.get().strip() or None

        if not name:
            messagebox.showwarning("Missing Data", "Student Name is required.")
            return

        if not student_id:
            student_id = self.db.get_next_student_id()
            self.student_id_var.set(student_id)

        if not image_path:
            image_path = f"dataset/{student_id}/"

        self.db.add_student(student_id, name, department, image_path)
        self._admin_status.set(f"Student {student_id} added")
        self._refresh_students()
        self._clear_student_form()

    def _update_student(self):
        student_id = self.student_id_var.get().strip()
        name = self.student_name_var.get().strip()
        department = self.student_dept_var.get().strip() or None
        image_path = self.student_image_var.get().strip() or None
        if not student_id or not name:
            messagebox.showwarning("Missing Data", "Student ID and Name are required.")
            return
        self.db.update_student(student_id, name, department, image_path)
        self._admin_status.set(f"Student {student_id} updated")
        self._refresh_students()

    def _delete_student(self):
        student_id = self.student_id_var.get().strip()
        if not student_id:
            messagebox.showwarning("Missing Data", "Student ID is required.")
            return
        if not messagebox.askyesno("Confirm", f"Delete student {student_id}?"):
            return
        self.db.delete_student(student_id)
        self._admin_status.set(f"Student {student_id} deleted")
        self._refresh_students()
        self._clear_student_form()

    def _add_course(self):
        course_id = self.course_id_var.get().strip()
        course_name = self.course_name_var.get().strip()
        start_time = self.course_start_var.get().strip()
        end_time = self.course_end_var.get().strip()
        days = self.course_days_var.get().strip().lower()

        if not course_name:
            messagebox.showwarning("Missing Data", "Course Name is required.")
            return

        if not course_id:
            course_id = self.db.get_next_course_id(prefix="CSC", start=101)
            self.course_id_var.set(course_id)
        if not self._valid_time(start_time) or not self._valid_time(end_time):
            messagebox.showwarning("Invalid Data", "Start/End time must be HH:MM (24h).")
            return
        if not days:
            messagebox.showwarning("Missing Data", "Days CSV is required (e.g., sat,sun,mon,tue,wed,thu).")
            return

        self.db.add_course(course_id, course_name, start_time, end_time, days)
        self._admin_status.set(f"Course {course_id} added")
        self._refresh_courses()
        self._clear_course_form()

    def _update_course(self):
        course_id = self.course_id_var.get().strip()
        course_name = self.course_name_var.get().strip()
        start_time = self.course_start_var.get().strip()
        end_time = self.course_end_var.get().strip()
        days = self.course_days_var.get().strip().lower()
        if not course_id or not course_name:
            messagebox.showwarning("Missing Data", "Course ID and Course Name are required.")
            return
        if not self._valid_time(start_time) or not self._valid_time(end_time):
            messagebox.showwarning("Invalid Data", "Start/End time must be HH:MM (24h).")
            return
        if not days:
            messagebox.showwarning("Missing Data", "Days CSV is required (e.g., sat,sun,mon,tue,wed,thu).")
            return
        self.db.update_course(course_id, course_name, start_time, end_time, days)
        self._admin_status.set(f"Course {course_id} updated")
        self._refresh_courses()

    def _delete_course(self):
        course_id = self.course_id_var.get().strip()
        if not course_id:
            messagebox.showwarning("Missing Data", "Course ID is required.")
            return
        if not messagebox.askyesno("Confirm", f"Delete course {course_id}?"):
            return
        self.db.delete_course(course_id)
        self._admin_status.set(f"Course {course_id} deleted")
        self._refresh_courses()
        self._clear_course_form()

    def _enroll_student(self):
        student_id = self.enroll_student_var.get().strip()
        course_id = self.enroll_course_var.get().strip()
        if not student_id or not course_id:
            messagebox.showwarning("Missing Data", "Student ID and Course ID are required.")
            return
        self.db.enroll_student(student_id, course_id)
        self._admin_status.set(f"Enrolled {student_id} -> {course_id}")
        self._refresh_enrollments()
        self._clear_enroll_form()

    def _remove_enrollment(self):
        student_id = self.enroll_student_var.get().strip()
        course_id = self.enroll_course_var.get().strip()
        if not student_id or not course_id:
            messagebox.showwarning("Missing Data", "Student ID and Course ID are required.")
            return
        if not messagebox.askyesno("Confirm", f"Remove {student_id} from {course_id}?"):
            return
        self.db.delete_enrollment(student_id, course_id)
        self._admin_status.set(f"Removed {student_id} -> {course_id}")
        self._refresh_enrollments()
        self._clear_enroll_form()

    def _refresh_students(self):
        self._clear_tree(self.students_tree)
        for student in self.db.get_students():
            self.students_tree.insert(
                "",
                tk.END,
                values=(
                    student["id"],
                    student["name"],
                    student.get("department") or "",
                    student.get("image_path") or "",
                ),
            )

    def _refresh_courses(self):
        self._clear_tree(self.courses_tree)
        for course in self.db.get_courses():
            self.courses_tree.insert(
                "",
                tk.END,
                values=(
                    course["course_id"],
                    course["course_name"],
                    course.get("start_time") or "",
                    course.get("end_time") or "",
                    course.get("days") or "",
                ),
            )

    def _refresh_enrollments(self):
        self._clear_tree(self.enrollments_tree)
        for enrollment in self.db.get_enrollments():
            self.enrollments_tree.insert(
                "",
                tk.END,
                values=(enrollment["student_id"], enrollment["course_id"]),
            )

    def _clear_tree(self, tree):
        for item in tree.get_children():
            tree.delete(item)

    def _on_student_select(self, _event):
        selected = self.students_tree.selection()
        if not selected:
            return
        values = self.students_tree.item(selected[0], "values")
        self.student_id_var.set(values[0])
        self.student_name_var.set(values[1])
        self.student_dept_var.set(values[2])
        self.student_image_var.set(values[3])

    def _on_course_select(self, _event):
        selected = self.courses_tree.selection()
        if not selected:
            return
        values = self.courses_tree.item(selected[0], "values")
        self.course_id_var.set(values[0])
        self.course_name_var.set(values[1])
        self.course_start_var.set(values[2])
        self.course_end_var.set(values[3])
        self.course_days_var.set(values[4])

    def _on_enrollment_select(self, _event):
        selected = self.enrollments_tree.selection()
        if not selected:
            return
        values = self.enrollments_tree.item(selected[0], "values")
        self.enroll_student_var.set(values[0])
        self.enroll_course_var.set(values[1])

    def _clear_student_form(self):
        self.student_id_var.set("")
        self.student_name_var.set("")
        self.student_dept_var.set("")
        self.student_image_var.set("")

    def _clear_course_form(self):
        self.course_id_var.set("")
        self.course_name_var.set("")
        self.course_start_var.set("")
        self.course_end_var.set("")
        self.course_days_var.set("sat,sun,mon,tue,wed,thu")

    def _clear_enroll_form(self):
        self.enroll_student_var.set("")
        self.enroll_course_var.set("")

    def _valid_time(self, value):
        try:
            __import__("datetime").datetime.strptime(value, "%H:%M")
            return True
        except (TypeError, ValueError):
            return False

    def _on_close(self):
        self.db.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    AttendanceGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
