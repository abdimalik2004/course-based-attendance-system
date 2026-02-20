-- MySQL schema

CREATE TABLE IF NOT EXISTS students (
    id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    department VARCHAR(255),
    image_path VARCHAR(1024)
);

CREATE TABLE IF NOT EXISTS courses (
    course_id VARCHAR(64) PRIMARY KEY,
    course_name VARCHAR(255) NOT NULL,
    start_time VARCHAR(5),
    end_time VARCHAR(5),
    days VARCHAR(64)
);

CREATE TABLE IF NOT EXISTS enrollments (
    student_id VARCHAR(64) NOT NULL,
    course_id VARCHAR(64) NOT NULL,
    PRIMARY KEY (student_id, course_id),
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
    FOREIGN KEY (course_id) REFERENCES courses(course_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    student_id VARCHAR(64) NOT NULL,
    course_id VARCHAR(64) NOT NULL,
    date VARCHAR(10) NOT NULL,
    time VARCHAR(8) NOT NULL,
    confidence DOUBLE NOT NULL,
    status VARCHAR(16) NOT NULL DEFAULT 'on_time' CHECK (status IN ('on_time', 'late', 'absent', 'excused')),
    session_key VARCHAR(255) NOT NULL,
    created_at VARCHAR(32) NOT NULL,
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
    FOREIGN KEY (course_id) REFERENCES courses(course_id) ON DELETE CASCADE,
    UNIQUE (student_id, course_id, session_key)
);
