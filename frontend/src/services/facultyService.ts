import { api } from "./api";
import { attendanceService } from "./attendanceService";
import { courseService } from "./courseService";
import { hrService } from "./hrService";

export const facultyService = {
  getSummary: async () =>
    api.get("/reports/summary").then((response) => response.data),

  getCourses: async () => courseService.listCourses({ skip: 0, limit: 200 }),

  getTeachers: async () => hrService.getTeachers(),

  getDepartments: async () => hrService.getDepartments(),

  getClasses: async () =>
    api
      .get("/classes", { params: { skip: 0, limit: 200 } })
      .then((response) => response.data),

  listAssignments: async () =>
    courseService.listAssignments({ skip: 0, limit: 200 }),

  createAssignment: async (payload: {
    course_id: number;
    teacher_id: number;
    is_primary?: boolean;
  }) => courseService.assignTeacher(payload),

  updateAssignment: async (
    id: string,
    payload: { teacher_id?: number; is_primary?: boolean },
  ) => courseService.updateAssignment(id, payload),

  deleteAssignment: async (id: string) => courseService.deleteAssignment(id),

  listSchedules: async () =>
    api
      .get("/schedules", { params: { skip: 0, limit: 200 } })
      .then((response) => response.data),

  createSchedule: async (payload: {
    course_id: number;
    weekday: string[];
    start_time: string;
    end_time: string;
    grace_period_minutes: number;
  }) => {
    try {
      console.log("SENDING PAYLOAD:", JSON.stringify(payload, null, 2));

      const response = await api.post("/schedules", payload);

      return response.data;
    } catch (error: any) {
      console.log(
        "FASTAPI VALIDATION:",
        JSON.stringify(error.response?.data, null, 2),
      );

      throw error;
    }
  },

  updateSchedule: async (
    id: string,
    payload: Partial<{
      course_id: number;
      weekday: string[];
      start_time: string;
      end_time: string;
      grace_period_minutes: number;
    }>,
  ) => api.put(`/schedules/${id}`, payload).then((response) => response.data),

  deleteSchedule: async (id: string) =>
    api.delete(`/schedules/${id}`).then((response) => response.data),

  listAttendanceRecords: async (params?: {
    page?: number;
    limit?: number;
    search?: string;
    faculty?: string;
    department?: string;
    course?: string;
    status?: string;
  }) => attendanceService.getAttendanceList(params),

  updateAttendanceStatus: async (recordId: string) =>
    attendanceService.updateAttendanceStatus(recordId, "EXCUSED"),
};

export default facultyService;
