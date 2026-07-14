import { create } from 'zustand';
import { hrService, type Teacher, type Faculty, type Department } from '@/services/hrService';

const STALE_MS = 30_000; // 30 seconds

interface HrState {
  teachers: Teacher[];
  faculties: Faculty[];
  departments: Department[];
  isLoading: boolean;
  error: string | null;
  lastFetchedAt: number | null;

  /** Fetch all HR data in one parallel shot. Skips if data is < 30 s old. */
  fetchAll: () => Promise<void>;
  /** Force-refetch (ignores stale window — use after mutations). */
  refetchAll: () => Promise<void>;

  // Kept for components that need individual slices (e.g. TeacherModal dropdowns)
  fetchTeachers: () => Promise<void>;
  fetchFaculties: () => Promise<void>;
  fetchDepartments: () => Promise<void>;

  addTeacher: (data: Omit<Teacher, 'id' | 'status' | 'userId' | 'linkedUsername'>) => Promise<void>; // phone/email/hireDate are optional inside Teacher
  updateTeacher: (id: string, data: Partial<Teacher>) => Promise<Teacher>;
  deleteTeacher: (id: string) => Promise<void>;
  linkUser: (teacherId: string, userId: string | null) => Promise<void>;
}

export const useHrStore = create<HrState>((set, get) => ({
  teachers: [],
  faculties: [],
  departments: [],
  isLoading: false,
  error: null,
  lastFetchedAt: null,

  fetchAll: async () => {
    const { lastFetchedAt } = get();
    if (lastFetchedAt && Date.now() - lastFetchedAt < STALE_MS) return;
    await get().refetchAll();
  },

  refetchAll: async () => {
    set({ isLoading: true, error: null });
    try {
      const [teachers, faculties, departments] = await Promise.all([
        hrService.getTeachers(),
        hrService.getFaculties(),
        hrService.getDepartments(),
      ]);
      set({ teachers, faculties, departments, isLoading: false, lastFetchedAt: Date.now() });
    } catch {
      set({ error: 'Failed to load HR data', isLoading: false });
    }
  },

  fetchTeachers: async () => {
    set({ isLoading: true, error: null });
    try {
      const teachers = await hrService.getTeachers();
      set({ teachers, isLoading: false });
    } catch {
      set({ error: 'Failed to fetch teachers', isLoading: false });
    }
  },

  fetchFaculties: async () => {
    try {
      const faculties = await hrService.getFaculties();
      set({ faculties });
    } catch {
      set({ error: 'Failed to fetch faculties' });
    }
  },

  fetchDepartments: async () => {
    try {
      const departments = await hrService.getDepartments();
      set({ departments });
    } catch {
      set({ error: 'Failed to fetch departments' });
    }
  },

  addTeacher: async (data) => {
    set({ isLoading: true, error: null });
    try {
      const newTeacher = await hrService.addTeacher(data);
      set((state) => ({
        teachers: [...state.teachers, newTeacher],
        isLoading: false,
        lastFetchedAt: null, // bust cache so next visit re-fetches
      }));
    } catch (error) {
      set({ isLoading: false });
      throw error;
    }
  },

  updateTeacher: async (id, data) => {
    set({ isLoading: true, error: null });
    try {
      const updatedTeacher = await hrService.updateTeacher(id, data);
      set((state) => ({
        teachers: state.teachers.map((t) => (t.id === id ? updatedTeacher : t)),
        isLoading: false,
      }));
      return updatedTeacher;
    } catch (error) {
      set({ isLoading: false });
      throw error;
    }
  },

  deleteTeacher: async (id) => {
    set({ isLoading: true, error: null });
    try {
      await hrService.deleteTeacher(id);
      set((state) => ({
        teachers: state.teachers.filter(t => t.id !== id),
        isLoading: false,
        lastFetchedAt: null,
      }));
    } catch (error) {
      set({ isLoading: false });
      throw error;
    }
  },

  linkUser: async (teacherId, userId) => {
    set({ isLoading: true, error: null });
    try {
      const updatedTeacher = await hrService.linkUser(teacherId, userId);
      set((state) => ({
        teachers: state.teachers.map((t) => (t.id === teacherId ? updatedTeacher : t)),
        isLoading: false,
      }));
    } catch (error) {
      set({ isLoading: false });
      throw error;
    }
  },
}));
