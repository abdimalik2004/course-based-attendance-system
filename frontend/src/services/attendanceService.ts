import { api } from "./api";

export const attendanceService = {
  startSession: async (payload: {
    course_id: number;
    schedule_id?: number | null;
    session_type?: string;
  }) => {
    return api.post("/sessions/start", payload).then((r) => r.data);
  },

  endSession: async (sessionId: number) => {
    return api
      .post("/sessions/end", { session_id: sessionId })
      .then((r) => r.data);
  },

  processFrame: async (sessionId: number, imageBase64: string) => {
    return api
      .post("/attendance/frame", { session_id: sessionId, image: imageBase64 })
      .then((r) => r.data);
  },

  listSessions: async (params?: {
    course_id?: number;
    skip?: number;
    limit?: number;
  }) => {
    return api.get("/sessions", { params }).then((r) => r.data);
  },

  listActiveSessions: async (params?: {
    course_id?: number;
    faculty_id?: number;
  }) => {
    return api.get("/sessions/active", { params }).then((r) => r.data);
  },

  getSessionRecords: async (sessionId: number) => {
    return api.get(`/sessions/${sessionId}/records`).then((r) => r.data);
  },

  getSchedulesForCourse: async (courseId: number) => {
    return api
      .get("/schedules", { params: { course_id: courseId, limit: 50 } })
      .then((r) => {
        const items = r.data?.items ?? r.data ?? [];
        return items.map((s: any) => {
          // Backend returns weekday as an array of day codes e.g. ["sat", "sun"]
          const weekdayArr: string[] = Array.isArray(s.weekday)
            ? s.weekday.map((d: string) => String(d).toLowerCase())
            : s.weekday
            ? String(s.weekday).toLowerCase().split(",").map((d: string) => d.trim()).filter(Boolean)
            : [];
          return {
            id: s.id,
            weekday: weekdayArr.map((d) => d.toUpperCase()).join(" / "), // display e.g. "SAT / SUN"
            weekday_raw: weekdayArr,  // raw lowercase codes for day-of-week validation
            start_time: s.start_time ?? "",
            end_time: s.end_time ?? "",
            grace_period_minutes: s.grace_period_minutes ?? 0,
          };
        });
      });
  },

  listClassesForCourse: async (courseId: number): Promise<{ id: number; name: string }[]> => {
    return api
      .get("/classes", { params: { course_id: courseId, limit: 50 } })
      .then((r) => {
        const items = r.data?.items ?? r.data ?? [];
        return items.map((c: any) => ({ id: Number(c.id), name: String(c.name) }));
      });
  },

  getAttendanceList: async (params?: {
    page?: number;
    limit?: number;
    search?: string;
    faculty?: string;
    department?: string;
    course?: string;
    course_id?: number;
    status?: string;
  }) => {
    const response = await api.get("/attendance/records", {
      params: {
        page: params?.page,
        limit: params?.limit,
        search: params?.search,
        faculty: params?.faculty,
        department: params?.department,
        course: params?.course,
        course_id: params?.course_id,
        status: params?.status,
      },
    });

    const records = response.data.data ?? [];
    const data = records.map((record: any) => {
      const normalizedStatus = String(record.status || "").toUpperCase();
      let statusLabel = "Unknown";
      if (normalizedStatus === "PRESENT") statusLabel = "Present";
      if (normalizedStatus === "LATE") statusLabel = "Late";
      if (normalizedStatus === "ABSENT") statusLabel = "Absent";
      if (normalizedStatus === "EXCUSED") statusLabel = "Excused";

      return {
        id: String(record.id),
        courseId:
          record.courseId != null
            ? String(record.courseId)
            : record.course_id != null
              ? String(record.course_id)
              : "",
        studentName: record.studentName,
        faculty: record.faculty || "",
        department: record.department || "",
        course: record.course || "",
        sessionId: record.sessionId || `SES-${record.id}`,
        status: statusLabel,
        confidence: record.confidence,
        recognizedAt: record.recognizedAt,
        attendedSessions: record.attendedSessions ?? 0,
        totalSessions: record.totalSessions ?? 1,
      };
    });

    return { data, total: response.data.total ?? data.length };
  },

  updateAttendanceStatus: async (recordId: string, status: string) =>
    api.put(`/attendance/records/${recordId}`, { status }).then((r) => r.data),
};

export default attendanceService;
