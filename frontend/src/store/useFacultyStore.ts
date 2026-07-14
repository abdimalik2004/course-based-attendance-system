import { create } from "zustand";
import { facultyService } from "@/services/facultyService";
import { hrService } from "@/services/hrService";
import { useAuthStore } from "@/store/useAuthStore";

const STALE_MS = 30_000; // 30 seconds

export interface FacultyTeacher {
  id: string;
  fullName: string;
}

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
  teachers: FacultyTeacher[];
  assignments: TeacherAssignment[];
  schedules: CourseSchedule[];
  lastFetchedAt: number | null;

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

  /** Fetch all data. Skips if data is < 30 s old. */
  fetchData: () => Promise<void>;
  /** Force-refetch all 5 collections — bypasses stale window. */
  refetchAll: () => Promise<void>;
  /** Targeted: re-fetch only the assignments list (1 API call). Used after assignment mutations. */
  refetchAssignments: () => Promise<void>;
  /** Targeted: re-fetch only the schedules list (1 API call). Used after schedule mutations. */
  refetchSchedules: () => Promise<void>;

  openModal: (type: "assign" | "schedule", mode: ModalMode, record?: any) => void;
  closeModal: (type: "assign" | "schedule") => void;

  addAssignment: (data: AssignmentFormData) => Promise<void>;
  updateAssignment: (id: string, data: Partial<AssignmentFormData>) => Promise<void>;
  deleteAssignment: (id: string) => Promise<void>;

  addSchedule: (data: ScheduleFormData) => Promise<void>;
  updateSchedule: (id: string, data: Partial<ScheduleFormData>) => Promise<void>;
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

// Normalize "HH:MM:SS" → "HH:MM" for HTML time inputs
const normalizeTime = (t: string | null | undefined): string => {
  if (!t) return "";
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

// Ensure seconds suffix for backend ("HH:MM" → "HH:MM:SS")
const toBackendTime = (t: string): string =>
  t.length === 5 ? `${t}:00` : t;

export const useFacultyStore = create<FacultyState>((set, get) => ({
  courses: [],
  teachers: [],
  assignments: [],
  schedules: [],
  lastFetchedAt: null,

  stats: {
    totalStudents: 0,
    totalTeachers: 0,
    totalDepartments: 0,
    totalClasses: 0,
    totalCourses: 0,
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
    const { lastFetchedAt } = get();
    if (lastFetchedAt && Date.now() - lastFetchedAt < STALE_MS) return;
    await get().refetchAll();
  },

  refetchAll: async () => {
    set({ isLoading: true, error: null });
    const facultyId = useAuthStore.getState().user?.facultyId;
    const fid = facultyId ? Number(facultyId) : undefined;
    try {
      const [summary, coursesRes, assignmentsRes, schedulesRes, teachersData] =
        await Promise.all([
          facultyService.getSummary(),
          facultyService.getCourses(fid),
          facultyService.listAssignments(),
          facultyService.listSchedules(),
          hrService.getTeachers(),
        ]);

      set({
        courses: (coursesRes.items ?? []).map(mapCourse),
        teachers: teachersData.map((t) => ({ id: t.id, fullName: t.fullName })),
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
        lastFetchedAt: Date.now(),
      });
    } catch (error) {
      set({
        isLoading: false,
        error: error instanceof Error ? error.message : "Failed to load faculty data",
      });
    }
  },

  refetchAssignments: async () => {
    try {
      const assignmentsRes = await facultyService.listAssignments();
      set({
        assignments: (assignmentsRes.items ?? []).map(mapAssignment),
        lastFetchedAt: Date.now(),
      });
    } catch (error) {
      set({ error: error instanceof Error ? error.message : "Failed to refresh assignments" });
      throw error;
    }
  },

  refetchSchedules: async () => {
    try {
      const schedulesRes = await facultyService.listSchedules();
      set({
        schedules: (schedulesRes.items ?? []).map(mapSchedule),
        lastFetchedAt: Date.now(),
      });
    } catch (error) {
      set({ error: error instanceof Error ? error.message : "Failed to refresh schedules" });
      throw error;
    }
  },

  addAssignment: async (data) => {
    try {
      await facultyService.createAssignment({
        course_id: Number(data.courseId),
        teacher_id: Number(data.teacherId),
        is_primary: data.status !== "inactive",
      });
      await get().refetchAssignments();
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : "Failed to create assignment",
      });
      throw error;
    }
  },

  updateAssignment: async (id, data) => {
    try {
      await facultyService.updateAssignment(id, {
        teacher_id: data.teacherId ? Number(data.teacherId) : undefined,
        is_primary: data.status !== undefined ? data.status !== "inactive" : undefined,
      });
      await get().refetchAssignments();
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : "Failed to update assignment",
      });
      throw error;
    }
  },

  deleteAssignment: async (id) => {
    try {
      await facultyService.deleteAssignment(id);
      await get().refetchAssignments();
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : "Failed to delete assignment",
      });
      throw error;
    }
  },

  addSchedule: async (data) => {
    try {
      await facultyService.createSchedule({
        course_id: Number(data.courseId),
        weekday: data.weekdays,
        start_time: toBackendTime(data.startTime),
        end_time: toBackendTime(data.endTime),
        grace_period_minutes: Number(data.gracePeriod),
      });
      await get().refetchSchedules();
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : "Failed to create schedule",
      });
      throw error;
    }
  },

  updateSchedule: async (id, data) => {
    const payload: Record<string, unknown> = {};
    if (data.courseId !== undefined) payload.course_id = Number(data.courseId);
    if (data.weekdays !== undefined) payload.weekday = data.weekdays;
    if (data.startTime !== undefined) payload.start_time = toBackendTime(data.startTime);
    if (data.endTime !== undefined) payload.end_time = toBackendTime(data.endTime);
    if (data.gracePeriod !== undefined) payload.grace_period_minutes = Number(data.gracePeriod);

    try {
      await facultyService.updateSchedule(id, payload as any);
      await get().refetchSchedules();
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : "Failed to update schedule",
      });
      throw error;
    }
  },

  deleteSchedule: async (id) => {
    try {
      await facultyService.deleteSchedule(id);
      await get().refetchSchedules();
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : "Failed to delete schedule",
      });
      throw error;
    }
  },
}));
