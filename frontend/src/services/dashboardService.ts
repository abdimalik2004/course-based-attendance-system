import { api } from "./api";

// ── Shared types used by multiple student portal pages ────────────────────────

export interface StudentAttendanceCourse {
  id: number;
  course_name: string;
  course_code: string;
  classes_attended: number;
  classes_absent: number;
  classes_excused: number;
  total_classes: number;
  attendance_percentage: number;
  /** "Good" | "Warning" | "Low" */
  status: string;
  last_updated: string | null;
}

export interface ExcuseRequestItem {
  id: number;
  student_id: number;
  course_id: number | null;
  course_name: string | null;
  course_code: string | null;
  request_date: string;    // "YYYY-MM-DD"
  reason: string | null;
  status: 'PENDING' | 'APPROVED' | 'DENIED';
  created_at: string;
  reviewed_at: string | null;
}

export interface FacultyExcuseRequestItem extends ExcuseRequestItem {
  student_name: string | null;
  student_number: string | null;
}

export interface StudentSessionRecord {
  record_id: number;
  session_id: number;
  date: string;          // ISO "YYYY-MM-DD"
  start_time: string;    // "HH:MM"
  session_type: string;  // "Lecture" | "Lab" | "Tutorial"
  status: string;        // "PRESENT" | "LATE" | "ABSENT" | "EXCUSED"
  recognized_at: string | null; // "HH:MM" or null
}

export interface StudentScheduleItem {
  id: number;
  course_id: number;
  course_name: string;
  course_code: string;
  /** Display labels e.g. ["Mon", "Wed", "Fri"] */
  weekdays: string[];
  start_time: string;
  end_time: string;
  grace_period_minutes: number;
  class_name: string | null;
  has_active_session: boolean;
}

// ─────────────────────────────────────────────────────────────────────────────

export const dashboardService = {
  adminOverview: async () => {
    const summary = await api
      .get("/reports/summary")
      .then((response) => response.data);

    return {
      totalStudents: summary.totalStudents ?? 0,
      totalTeachers: summary.totalTeachers ?? 0,
      totalFaculties: summary.totalFaculties ?? 0,
      totalAttendanceRecords: summary.totalAttendanceRecords ?? 0,
      attendanceRate: summary.attendanceRate ?? 0,
    };
  },

  facultyOverview: async () =>
    api.get("/reports/summary").then((response) => response.data),

  /** Attendance-only fetch — use queryKey ['studentAttendance'] */
  studentAttendanceData: async (): Promise<StudentAttendanceCourse[]> => {
    const response = await api.get('/student-portal/me/attendance');
    return response.data ?? [];
  },

  /** Schedule-only fetch — use queryKey ['studentSchedule'] */
  studentScheduleData: async (): Promise<StudentScheduleItem[]> => {
    const response = await api.get('/student-portal/me/schedule');
    return response.data ?? [];
  },

  /** Per-session history for one course — use queryKey ['studentSessions', courseId] */
  studentSessionHistory: async (courseId: number): Promise<StudentSessionRecord[]> => {
    const response = await api.get(`/student-portal/me/attendance/${courseId}/sessions`);
    return response.data ?? [];
  },

  studentProfile: async () => {
    const response = await api.get('/student-portal/me/profile');
    return response.data as {
      id?: number;
      student_number?: string;
      full_name?: string;
      email?: string | null;
      phone?: string | null;
      date_of_birth?: string | null;
      status?: string;
      faculty_id?: number;
      faculty_name?: string | null;
      department_id?: number;
      department_name?: string | null;
      enrolled_at?: string | null;
      username?: string;
    };
  },

  /** Submit an excuse request (student) */
  submitExcuseRequest: async (payload: {
    request_date: string;
    course_id?: number | null;
    reason?: string | null;
  }): Promise<ExcuseRequestItem> => {
    const response = await api.post('/student-portal/me/excuse-requests', payload);
    return response.data;
  },

  /** List this student's own excuse requests */
  myExcuseRequests: async (): Promise<ExcuseRequestItem[]> => {
    const response = await api.get('/student-portal/me/excuse-requests');
    return response.data ?? [];
  },

  studentOverview: async () => {
    const [attendanceRes, scheduleRes] = await Promise.all([
      api
        .get("/student-portal/me/attendance")
        .then((response) => response.data)
        .catch(() => []),
      api
        .get("/student-portal/me/schedule")
        .then((response) => response.data)
        .catch(() => []),
    ]);

    return { attendance: attendanceRes, schedule: scheduleRes };
  },
};

export default dashboardService;
