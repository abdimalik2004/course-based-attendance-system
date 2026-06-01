import { create } from "zustand";
import { facultyService } from "@/services/facultyService";

export interface TeacherAssignment {
  id: string;
  courseId: string;
  teacherId: string;
  isPrimary: boolean;
  createdAt: string;
}

export interface AssignmentFormData {
  courseId: string;
  teacherId: string;
  status?: "active" | "inactive";
}

export interface ScheduleFormData {
  courseId: string;
  weekdays: string[];
  startTime: string;
  endTime: string;
  gracePeriod: number;
}

export interface CourseSchedule {
  id: string;
  courseId: string;
  weekdays: string[];
  startTime: string;
  endTime: string;
  gracePeriod: number;
  createdAt: string;
}

export interface Course {
  id: string;
  title: string;
  code: string;
}

export type ModalMode = "create" | "edit" | "view";

interface ModalState<T> {
  isOpen: boolean;
  mode: ModalMode;
  record: T | null;
}

interface FacultyState {
  courses: Course[];
  assignments: TeacherAssignment[];
  schedules: CourseSchedule[];

  stats: {
    totalStudents: number;
    totalTeachers: number;
    totalDepartments: number;
    totalClasses: number;
    totalCourses: number;
  };

  isLoading: boolean;
  error: string | null;

  assignModal: ModalState<TeacherAssignment>;
  scheduleModal: ModalState<CourseSchedule>;

  fetchData: () => Promise<void>;

  openModal: (
    type: "assign" | "schedule",
    mode: ModalMode,
    record?: any,
  ) => void;
  closeModal: (type: "assign" | "schedule") => void;

  addAssignment: (data: AssignmentFormData) => Promise<void>;
  updateAssignment: (
    id: string,
    data: Partial<AssignmentFormData>,
  ) => Promise<void>;
  deleteAssignment: (id: string) => Promise<void>;

  addSchedule: (data: ScheduleFormData) => Promise<void>;
  updateSchedule: (
    id: string,
    data: Partial<ScheduleFormData>,
  ) => Promise<void>;
  deleteSchedule: (id: string) => Promise<void>;
}

const defaultModalState = {
  isOpen: false,
  mode: "create" as ModalMode,
  record: null,
};

const mapCourse = (course: any): Course => ({
  id: String(course.id),
  title: course.title,
  code: course.code,
});

const mapAssignment = (assignment: any): TeacherAssignment => ({
  id: String(assignment.id),
  courseId: String(assignment.course_id),
  teacherId: String(assignment.teacher_id ?? ""),
  isPrimary: Boolean(assignment.is_primary),
  createdAt: assignment.created_at ?? new Date().toISOString(),
});

// Normalize a time string from the backend ("HH:MM:SS") to "HH:MM" for HTML time inputs
const normalizeTime = (t: string | null | undefined): string => {
  if (!t) return "";
  // If seconds are present (HH:MM:SS), strip them
  const parts = t.split(":");
  return parts.length >= 2 ? `${parts[0]}:${parts[1]}` : t;
};

const mapSchedule = (schedule: any): CourseSchedule => ({
  id: String(schedule.id),
  courseId: String(schedule.course_id),
  weekdays: Array.isArray(schedule.weekday) ? schedule.weekday.map(String) : [],
  startTime: normalizeTime(schedule.start_time),
  endTime: normalizeTime(schedule.end_time),
  gracePeriod: schedule.grace_period_minutes ?? 0,
  createdAt: schedule.created_at ?? new Date().toISOString(),
});

export const useFacultyStore = create<FacultyState>((set, get) => ({
  courses: [],
  assignments: [],
  schedules: [],

  stats: {
    totalStudents: 1250,
    totalTeachers: 45,
    totalDepartments: 4,
    totalClasses: 32,
    totalCourses: 48,
  },

  isLoading: false,
  error: null,

  assignModal: { ...defaultModalState },
  scheduleModal: { ...defaultModalState },

  openModal: (type, mode, record = null) =>
    set(() => ({
      [`${type}Modal`]: { isOpen: true, mode, record },
    })),

  closeModal: (type) =>
    set((state) => ({
      [`${type}Modal`]: {
        ...(state[`${type}Modal` as keyof FacultyState] as any),
        isOpen: false,
      },
    })),

  fetchData: async () => {
    set({ isLoading: true, error: null });
    try {
      const [summary, coursesRes, assignmentsRes, schedulesRes] =
        await Promise.all([
          facultyService.getSummary(),
          facultyService.getCourses(),
          facultyService.listAssignments(),
          facultyService.listSchedules(),
        ]);

      set({
        courses: (coursesRes.items ?? []).map(mapCourse),
        assignments: (assignmentsRes.items ?? []).map(mapAssignment),
        schedules: (schedulesRes.items ?? []).map(mapSchedule),
        stats: {
          totalStudents: Number(summary.totalStudents ?? 0),
          totalTeachers: Number(summary.totalTeachers ?? 0),
          totalDepartments: Number(summary.totalDepartments ?? 0),
          totalClasses: Number(summary.totalClasses ?? 0),
          totalCourses: Number(summary.totalCourses ?? coursesRes.total ?? 0),
        },
        isLoading: false,
      });
    } catch (error) {
      set({
        isLoading: false,
        error:
          error instanceof Error
            ? error.message
            : "Failed to load faculty data",
      });
    }
  },

  addAssignment: async (data) => {
    try {
      await facultyService.createAssignment({
        course_id: Number(data.courseId),
        teacher_id: Number(data.teacherId),
        is_primary: data.status !== "inactive",
      });
      await get().fetchData();
    } catch (error) {
      set({
        error:
          error instanceof Error
            ? error.message
            : "Failed to create assignment",
      });
      throw error;
    }
  },
  updateAssignment: async (id, data) => {
    try {
      await facultyService.updateAssignment(id, {
        teacher_id: data.teacherId ? Number(data.teacherId) : undefined,
        is_primary:
          data.status !== undefined ? data.status !== "inactive" : undefined,
      });
      await get().fetchData();
    } catch (error) {
      set({
        error:
          error instanceof Error
            ? error.message
            : "Failed to update assignment",
      });
      throw error;
    }
  },
  deleteAssignment: async (id) => {
    try {
      await facultyService.deleteAssignment(id);
      await get().fetchData();
    } catch (error) {
      set({
        error:
          error instanceof Error
            ? error.message
            : "Failed to delete assignment",
      });
      throw error;
    }
  },

  addSchedule: async (data) => {
    try {
      console.log("STORE RECEIVED:", data);

      await facultyService.createSchedule(data);

      await get().fetchData();
    } catch (error) {
      set({
        error:
          error instanceof Error ? error.message : "Failed to create schedule",
      });
      throw error;
    }
  },
  updateSchedule: async (id, data) => {
    const payload: Record<string, unknown> = {};
    if (data.courseId !== undefined) payload.course_id = Number(data.courseId);
    if (data.weekdays !== undefined) payload.weekday = data.weekdays;
    // Ensure seconds are always included — backend expects "HH:MM:SS"
    if (data.startTime !== undefined)
      payload.start_time = data.startTime.length === 5 ? `${data.startTime}:00` : data.startTime;
    if (data.endTime !== undefined)
      payload.end_time = data.endTime.length === 5 ? `${data.endTime}:00` : data.endTime;
    if (data.gracePeriod !== undefined) {
      payload.grace_period_minutes = Number(data.gracePeriod);
    }

    try {
      await facultyService.updateSchedule(id, payload as any);
      await get().fetchData();
    } catch (error) {
      set({
        error:
          error instanceof Error ? error.message : "Failed to update schedule",
      });
      throw error;
    }
  },
  deleteSchedule: async (id) => {
    try {
      await facultyService.deleteSchedule(id);
      await get().fetchData();
    } catch (error) {
      set({
        error:
          error instanceof Error ? error.message : "Failed to delete schedule",
      });
      throw error;
    }
  },
}));
