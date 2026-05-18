import { api } from "./api";

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

  facultyOverview: async (facultyId?: number) =>
    api
      .get("/reports", { params: { faculty_id: facultyId } })
      .then((response) => response.data),

  studentOverview: async (studentId?: number) => {
    if (!studentId) return null;

    const [attendanceRes, scheduleRes] = await Promise.all([
      api
        .get(`/student-portal/students/${studentId}/attendance`)
        .then((response) => response.data)
        .catch(() => []),
      api
        .get(`/student-portal/students/${studentId}/schedule`)
        .then((response) => response.data)
        .catch(() => []),
    ]);

    return { attendance: attendanceRes, schedule: scheduleRes };
  },
};

export default dashboardService;
