/**
 * Teacher-scoped API methods (#23 — consolidate scattered calls).
 *
 * Before: all 4 teacher pages imported courseService AND attendanceService
 *         separately to perform teacher-specific lookups.
 * After:  teacher pages import teacherService for teacher-scoped reads;
 *         the underlying service calls stay here as single source of truth.
 */

import { api } from "./api";
import attendanceService from "./attendanceService";

export interface TeacherProfile {
  id: number;
  teacher_number: string;
  full_name: string;
  role: string;
  status: string;
  faculty_id: number;
  department_id: number;
  faculty_name: string | null;
  department_name: string | null;
  user_id: number | null;
  linked_username: string | null;
  phone: string | null;
  email: string | null;
  hire_date: string | null;
}

export interface CourseStats {
  total_records: number;
  present: number;
  late: number;
  absent: number;
}

export interface CourseStudentStat {
  student_id: number;
  student_number: string;
  student_name: string;
  present: number;
  late: number;
  absent: number;
  total: number;
}

export interface CourseSession {
  session_id: number;
  session_date: string | null;
  start_time: string | null;
  end_time: string | null;
  status: string;
  present: number;
  late: number;
  absent: number;
  total: number;
}

export interface EnrolledStudent {
  id: number;
  student_number: string;
  full_name: string;
  class_id: number | null;
  faculty_id: number | null;
  department_id: number | null;
  status: string;
}

export const teacherService = {
  /**
   * GET /teachers/me — returns the Teacher record linked to the currently
   * authenticated teacher account. Raises 404 if no profile is linked yet.
   */
  getMyProfile: async (): Promise<TeacherProfile> => {
    return api.get("/teachers/me").then((r) => r.data);
  },

  /**
   * GET /courses/assignments?teacher_id=<id>
   * Returns all course assignments for the given teacher, including
   * course_title and course_code from the backend join.
   *
   * Replaces the scattered `courseService.listAssignments({ teacher_id })` calls
   * in Dashboard, Attendance, AttendanceList and Schedule (#23).
   */
  getAssignedCourses: async (
    teacherId: number,
    opts: { skip?: number; limit?: number } = {},
  ): Promise<any> => {
    return api
      .get("/courses/assignments", {
        params: { teacher_id: teacherId, skip: opts.skip ?? 0, limit: opts.limit ?? 200 },
      })
      .then((r) => r.data);
  },

  /**
   * Fetch schedules for all supplied course IDs in parallel, annotating
   * each schedule object with its `course_id` (same pattern Dashboard and
   * Schedule pages use inline — centralised here to avoid duplication).
   */
  getSchedulesForCourses: async (
    courseIds: number[],
  ): Promise<any[]> => {
    if (courseIds.length === 0) return [];
    const perCourse = await Promise.all(
      courseIds.map((courseId) =>
        attendanceService
          .getSchedulesForCourse(courseId)
          .then((schedules: any[]) =>
            schedules.map((s: any) => ({ ...s, course_id: courseId })),
          ),
      ),
    );
    return perCourse.flat();
  },
  /**
   * GET /reports/course/{course_id}
   * Returns aggregate attendance stats for the course (total, present, late, absent).
   */
  getCourseStats: async (courseId: number): Promise<CourseStats> => {
    return api.get(`/reports/course/${courseId}`).then((r) => r.data);
  },

  /**
   * GET /reports/course/{course_id}/students
   * Per-student attendance breakdown for the course.
   */
  getCourseStudentStats: async (courseId: number): Promise<{ students: CourseStudentStat[] }> => {
    return api.get(`/reports/course/${courseId}/students`).then((r) => r.data);
  },

  /**
   * GET /reports/course/{course_id}/sessions
   * Session-level breakdown for the course.
   */
  getCourseSessions: async (courseId: number): Promise<{ sessions: CourseSession[] }> => {
    return api.get(`/reports/course/${courseId}/sessions`).then((r) => r.data);
  },

  /**
   * GET /courses/{course_id}/students
   * Full enrolled student roster for the course.
   */
  getCourseEnrolledStudents: async (courseId: number): Promise<EnrolledStudent[]> => {
    return api.get(`/courses/${courseId}/students`).then((r) => r.data);
  },

  /**
   * PUT /attendance/records/{record_id}  body: { status: "EXCUSED" }
   * Teachers can only change ABSENT → EXCUSED.
   */
  excuseAttendanceRecord: async (recordId: number): Promise<any> => {
    return api.put(`/attendance/records/${recordId}`, { status: 'EXCUSED' }).then((r) => r.data);
  },
};

export default teacherService;
