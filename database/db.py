import sqlite3
from datetime import datetime
from pathlib import Path

from utils.logging import get_logger

try:
    import pymysql
except ImportError:
    pymysql = None


class Database:
    def __init__(self, config):
        self.config = config
        self.conn = None
        self.logger = get_logger(__name__)
        self._is_mysql = config.db_type.lower() == "mysql"

    def connect(self):
        if self._is_mysql:
            if pymysql is None:
                raise RuntimeError("pymysql is required for MySQL connections")
            self.conn = pymysql.connect(
                host=self.config.db_host,
                port=self.config.db_port,
                user=self.config.db_user,
                password=self.config.db_password,
                database=self.config.db_name,
                autocommit=False,
            )
        else:
            self.conn = sqlite3.connect(self.config.db_path)
            self.conn.execute("PRAGMA foreign_keys = ON;")

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

    def _format_query(self, query):
        if not self._is_mysql:
            return query
        query = query.replace("?", "%s")
        query = query.replace("INSERT OR IGNORE", "INSERT IGNORE")
        return query

    def execute(self, query, params=None, commit=True):
        if params is None:
            params = ()
        if self.conn is None:
            self.connect()
        if self._is_mysql:
            try:
                # Keep long-running recognition sessions alive.
                self.conn.ping(reconnect=True)
            except Exception as exc:
                self.logger.warning("MySQL ping failed, reconnecting: %s", exc)
                self.connect()
        cursor = self.conn.cursor()
        cursor.execute(self._format_query(query), params)
        if commit:
            self.conn.commit()
        return cursor

    def fetchone(self, query, params=None):
        cursor = self.execute(query, params, commit=False)
        return cursor.fetchone()

    def fetchall(self, query, params=None):
        cursor = self.execute(query, params, commit=False)
        return cursor.fetchall()

    def init_schema(self):
        schema_file = "schema_mysql.sql" if self._is_mysql else "schema_sqlite.sql"
        schema_path = Path(self.config.base_dir) / "database" / schema_file
        with open(schema_path, "r", encoding="utf-8") as f:
            statements = f.read().split(";")
        for stmt in statements:
            stmt = stmt.strip()
            if stmt:
                self.execute(stmt + ";")

    def ensure_course_schedule_columns(self):
        if self._is_mysql:
            rows = self.fetchall(
                "SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA = ? AND TABLE_NAME = 'courses'",
                (self.config.db_name,),
            )
            columns = {row[0].lower() for row in rows}
        else:
            rows = self.fetchall("PRAGMA table_info(courses)")
            columns = {row[1].lower() for row in rows}

        if "start_time" not in columns:
            self.execute("ALTER TABLE courses ADD COLUMN start_time VARCHAR(5)")
        if "end_time" not in columns:
            self.execute("ALTER TABLE courses ADD COLUMN end_time VARCHAR(5)")
        if "days" not in columns:
            self.execute("ALTER TABLE courses ADD COLUMN days VARCHAR(64)")

    def add_student(self, student_id, name, department=None, image_path=None):
        self.execute(
            "INSERT OR IGNORE INTO students (id, name, department, image_path) VALUES (?, ?, ?, ?)",
            (student_id, name, department, image_path),
        )

    def add_course(self, course_id, course_name, start_time=None, end_time=None, days=None):
        self.ensure_course_schedule_columns()
        if self._is_mysql:
            query = (
                "INSERT INTO courses (course_id, course_name, start_time, end_time, days) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON DUPLICATE KEY UPDATE course_name=VALUES(course_name), "
                "start_time=VALUES(start_time), end_time=VALUES(end_time), days=VALUES(days)"
            )
        else:
            query = (
                "INSERT INTO courses (course_id, course_name, start_time, end_time, days) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(course_id) DO UPDATE SET course_name=excluded.course_name, "
                "start_time=excluded.start_time, end_time=excluded.end_time, days=excluded.days"
            )
        self.execute(query, (course_id, course_name, start_time, end_time, days))

    def get_courses(self):
        self.ensure_course_schedule_columns()
        rows = self.fetchall(
            "SELECT course_id, course_name, start_time, end_time, days FROM courses"
        )
        return [
            {
                "course_id": row[0],
                "course_name": row[1],
                "start_time": row[2],
                "end_time": row[3],
                "days": row[4],
            }
            for row in rows
        ]

    def get_course(self, course_id):
        self.ensure_course_schedule_columns()
        row = self.fetchone(
            "SELECT course_id, course_name, start_time, end_time, days FROM courses WHERE course_id = ?",
            (course_id,),
        )
        if not row:
            return None
        return {
            "course_id": row[0],
            "course_name": row[1],
            "start_time": row[2],
            "end_time": row[3],
            "days": row[4],
        }

    def enroll_student(self, student_id, course_id):
        self.execute(
            "INSERT OR IGNORE INTO enrollments (student_id, course_id) VALUES (?, ?)",
            (student_id, course_id),
        )

    def is_enrolled(self, student_id, course_id):
        row = self.fetchone(
            "SELECT 1 FROM enrollments WHERE student_id = ? AND course_id = ?",
            (student_id, course_id),
        )
        return row is not None

    def attendance_exists(self, student_id, course_id, session_key):
        row = self.fetchone(
            "SELECT 1 FROM attendance WHERE student_id = ? AND course_id = ? AND session_key = ?",
            (student_id, course_id, session_key),
        )
        return row is not None

    def record_attendance(self, student_id, course_id, confidence, session_key, status="on_time"):
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        created_at = now.isoformat()
        self.execute(
            "INSERT INTO attendance (student_id, course_id, date, time, confidence, status, session_key, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (student_id, course_id, date_str, time_str, confidence, status, session_key, created_at),
        )

    def get_student(self, student_id):
        row = self.fetchone(
            "SELECT id, name, department, image_path FROM students WHERE id = ?",
            (student_id,),
        )
        if not row:
            return None
        return {
            "id": row[0],
            "name": row[1],
            "department": row[2],
            "image_path": row[3],
        }

    def get_students(self):
        rows = self.fetchall("SELECT id, name, department, image_path FROM students ORDER BY id")
        return [
            {
                "id": row[0],
                "name": row[1],
                "department": row[2],
                "image_path": row[3],
            }
            for row in rows
        ]

    def get_next_student_id(self):
        if self._is_mysql:
            row = self.fetchone("SELECT MAX(CAST(id AS UNSIGNED)) FROM students")
        else:
            row = self.fetchone("SELECT MAX(CAST(id AS INTEGER)) FROM students")
        max_id = row[0] if row and row[0] is not None else 0
        return str(int(max_id) + 1)

    def get_next_course_id(self, prefix="CSC", start=101):
        if self._is_mysql:
            row = self.fetchone(
                "SELECT MAX(CAST(SUBSTRING(course_id, ? + 1) AS UNSIGNED)) FROM courses WHERE course_id LIKE ?",
                (len(prefix), f"{prefix}%"),
            )
        else:
            row = self.fetchone(
                "SELECT MAX(CAST(SUBSTR(course_id, ? + 1) AS INTEGER)) FROM courses WHERE course_id LIKE ?",
                (len(prefix), f"{prefix}%"),
            )
        max_num = row[0] if row and row[0] is not None else (start - 1)
        return f"{prefix}{int(max_num) + 1}"

    def update_student(self, student_id, name, department=None, image_path=None):
        self.execute(
            "UPDATE students SET name = ?, department = ?, image_path = ? WHERE id = ?",
            (name, department, image_path, student_id),
        )

    def delete_student(self, student_id):
        self.execute("DELETE FROM students WHERE id = ?", (student_id,))

    def update_course(self, course_id, course_name, start_time=None, end_time=None, days=None):
        self.ensure_course_schedule_columns()
        self.execute(
            "UPDATE courses SET course_name = ?, start_time = ?, end_time = ?, days = ? WHERE course_id = ?",
            (course_name, start_time, end_time, days, course_id),
        )

    def delete_course(self, course_id):
        self.execute("DELETE FROM courses WHERE course_id = ?", (course_id,))

    def get_enrollments(self):
        rows = self.fetchall(
            "SELECT student_id, course_id FROM enrollments ORDER BY course_id, student_id"
        )
        return [
            {"student_id": row[0], "course_id": row[1]}
            for row in rows
        ]

    def get_enrolled_student_ids(self, course_id):
        rows = self.fetchall(
            "SELECT student_id FROM enrollments WHERE course_id = ? ORDER BY student_id",
            (course_id,),
        )
        return [row[0] for row in rows]

    def mark_absent_for_course_session(self, course_id, session_key):
        absent_count = 0
        for student_id in self.get_enrolled_student_ids(course_id):
            if self.attendance_exists(student_id, course_id, session_key):
                continue
            self.record_attendance(
                student_id,
                course_id,
                confidence=0.0,
                session_key=session_key,
                status="absent",
            )
            absent_count += 1
        return absent_count

    def delete_enrollment(self, student_id, course_id):
        self.execute(
            "DELETE FROM enrollments WHERE student_id = ? AND course_id = ?",
            (student_id, course_id),
        )
