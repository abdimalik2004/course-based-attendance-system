import { create } from "zustand";
import { api } from "@/services/api";
import { courseService } from "@/services/courseService";

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

const mapSchedule = (schedule: any): CourseSchedule => ({
  id: String(schedule.id),
  courseId: String(schedule.course_id),
  weekdays: Array.isArray(schedule.weekday) ? schedule.weekday.map(String) : [],
  startTime: schedule.start_time,
  endTime: schedule.end_time,
  gracePeriod: schedule.grace_period_minutes ?? 0,
  createdAt: schedule.created_at ?? new Date().toISOString(),
});

export const useFacultyStore = create<FacultyState>((set) => ({
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
    set({ isLoading: true });
    try {
      const [
        coursesRes,
        assignmentsRes,
        schedulesRes,
        studentsRes,
        teachersRes,
        departmentsRes,
        classesRes,
      ] = await Promise.all([
        api.get("/courses", { params: { skip: 0, limit: 200 } }),
        courseService.listAssignments(),
        api.get("/schedules", { params: { skip: 0, limit: 200 } }),
        api.get("/students", { params: { skip: 0, limit: 1 } }),
        api.get("/teachers", { params: { skip: 0, limit: 1 } }),
        api.get("/departments", { params: { skip: 0, limit: 1 } }),
        api.get("/classes", { params: { skip: 0, limit: 1 } }),
      ]);

      const schedules = schedulesRes.data?.items ?? [];

      set({
        courses: (coursesRes.data?.items ?? []).map(mapCourse),
        assignments: (assignmentsRes ?? []).map(mapAssignment),
        schedules: schedules.map(mapSchedule),
        stats: {
          totalStudents: studentsRes.data?.total ?? 0,
          totalTeachers: teachersRes.data?.total ?? 0,
          totalDepartments: departmentsRes.data?.total ?? 0,
          totalClasses: classesRes.data?.total ?? 0,
          totalCourses: coursesRes.data?.total ?? 0,
        },
        isLoading: false,
      });
    } catch {
      set({ isLoading: false });
    }
  },

  addAssignment: async (data) => {
    const assignment = await courseService.assignTeacher({
      course_id: Number(data.courseId),
      teacher_id: Number(data.teacherId),
      is_primary: data.status !== "inactive",
    });
    set((state) => ({
      assignments: [
        ...state.assignments,
        {
          id: String(assignment?.id),
          courseId: data.courseId,
          teacherId: data.teacherId,
          isPrimary: data.status !== "inactive",
          createdAt: new Date().toISOString(),
        },
      ],
    }));
  },
  updateAssignment: async (id, data) => {
    const assignment = await courseService.updateAssignment(id, {
      teacher_id: data.teacherId ? Number(data.teacherId) : undefined,
      is_primary:
        data.status !== undefined ? data.status !== "inactive" : undefined,
    });
    set((state) => ({
      assignments: state.assignments.map((a) =>
        a.id === id ? mapAssignment(assignment) : a,
      ),
    }));
  },
  deleteAssignment: async (id) => {
    await courseService.deleteAssignment(id);
    set((state) => ({
      assignments: state.assignments.filter((a) => a.id !== id),
    }));
  },

  addSchedule: async (data) => {
    const schedule = await api.post("/schedules", {
      course_id: Number(data.courseId),
      weekday: data.weekdays,
      start_time: data.startTime,
      end_time: data.endTime,
      grace_period_minutes: Number(data.gracePeriod),
    });
    set((state) => ({
      schedules: [...state.schedules, mapSchedule(schedule.data)],
    }));
  },
  updateSchedule: async (id, data) => {
    const payload: Record<string, unknown> = {};
    if (data.courseId !== undefined) payload.course_id = Number(data.courseId);
    if (data.weekdays !== undefined) payload.weekday = data.weekdays;
    if (data.startTime !== undefined) payload.start_time = data.startTime;
    if (data.endTime !== undefined) payload.end_time = data.endTime;
    if (data.gracePeriod !== undefined) {
      payload.grace_period_minutes = Number(data.gracePeriod);
    }

    const schedule = await api.put(`/schedules/${id}`, payload);
    set((state) => ({
      schedules: state.schedules.map((s) =>
        s.id === id ? mapSchedule(schedule.data) : s,
      ),
    }));
  },
  deleteSchedule: async (id) => {
    await api.delete(`/schedules/${id}`);
    set((state) => ({ schedules: state.schedules.filter((s) => s.id !== id) }));
  },
}));
