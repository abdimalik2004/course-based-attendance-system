import { api } from "./api";

export const attendanceService = {
  startSession: async (payload: {
    course_id: number;
    schedule_id?: number | null;
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

  getSessionRecords: async (sessionId: number) => {
    return api.get(`/sessions/${sessionId}/records`).then((r) => r.data);
  },

  getAttendanceList: async (params?: {
    page?: number;
    limit?: number;
    search?: string;
    faculty?: string;
    department?: string;
    course?: string;
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
