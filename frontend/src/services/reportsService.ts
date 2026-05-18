import { api } from "./api";
import type {
  ReportSummary,
  AbsenceRecord,
  ChartDataPoint,
  DistributionData,
} from "../types/reports.types";

export const reportsService = {
  getSummaryMetrics: async (): Promise<ReportSummary> => {
    const response = await api.get("/reports/summary");
    return {
      totalStudents: response.data.totalStudents ?? 0,
      totalTeachers: response.data.totalTeachers ?? 0,
      totalFaculties: response.data.totalFaculties ?? 0,
      attendanceRate: response.data.attendanceRate ?? 0,
    };
  },

  getAbsenceRanking: async (params?: {
    page?: number;
    limit?: number;
    search?: string;
    type?: string;
    faculty?: string;
    department?: string;
    course?: string;
  }): Promise<{ data: AbsenceRecord[]; total: number }> => {
    const response = await api.get("/reports/absence-ranking", {
      params: {
        page: params?.page,
        limit: params?.limit,
        search: params?.search,
        type: params?.type,
        faculty: params?.faculty,
        department: params?.department,
        course: params?.course,
      },
    });

    return {
      data: response.data.data ?? [],
      total: response.data.total ?? 0,
    };
  },

  getAttendanceChartData: async (): Promise<ChartDataPoint[]> => {
    const response = await api.get("/sessions", {
      params: { skip: 0, limit: 200 },
    });
    const sessions = response.data ?? [];
    const byMonth = new Map<string, number>();

    sessions.forEach((session: any) => {
      const key = new Date(
        session.start_time ?? session.created_at ?? session.session_date,
      ).toLocaleString("en", { month: "short" });
      byMonth.set(key, (byMonth.get(key) ?? 0) + 1);
    });

    return Array.from(byMonth.entries()).map(([name, value]) => ({
      name,
      value,
    }));
  },

  getDistributionSummary: async (): Promise<DistributionData> => {
    const [studentsRes, teachersRes, facultiesRes] = await Promise.all([
      api.get("/students", { params: { skip: 0, limit: 1 } }),
      api.get("/teachers", { params: { skip: 0, limit: 1 } }),
      api.get("/faculties", { params: { skip: 0, limit: 1 } }),
    ]);

    return {
      students: studentsRes.data?.total ?? 0,
      teachers: teachersRes.data?.total ?? 0,
      faculties: facultiesRes.data?.total ?? 0,
    };
  },
};
