import { create } from 'zustand';
import { hrService, type Teacher, type Faculty, type Department } from '@/services/hrService';

interface HrState {
  teachers: Teacher[];
  faculties: Faculty[];
  departments: Department[];
  isLoading: boolean;
  error: string | null;
  
  fetchTeachers: () => Promise<void>;
  fetchFaculties: () => Promise<void>;
  fetchDepartments: () => Promise<void>;
  
  addTeacher: (data: Omit<Teacher, 'id' | 'status' | 'userId'>) => Promise<void>;
  updateTeacher: (id: string, data: Partial<Teacher>) => Promise<void>;
  deleteTeacher: (id: string) => Promise<void>;
}

export const useHrStore = create<HrState>((set) => ({
  teachers: [],
  faculties: [],
  departments: [],
  isLoading: false,
  error: null,

  fetchTeachers: async () => {
    set({ isLoading: true, error: null });
    try {
      const teachers = await hrService.getTeachers();
      set({ teachers, isLoading: false });
    } catch (error) {
      set({ error: 'Failed to fetch teachers', isLoading: false });
    }
  },

  fetchFaculties: async () => {
    try {
      const faculties = await hrService.getFaculties();
      set({ faculties });
    } catch (error) {
      set({ error: 'Failed to fetch faculties' });
    }
  },

  fetchDepartments: async () => {
    try {
      const departments = await hrService.getDepartments();
      set({ departments });
    } catch (error) {
      set({ error: 'Failed to fetch departments' });
    }
  },

  addTeacher: async (data) => {
    set({ isLoading: true, error: null });
    try {
      const newTeacher = await hrService.addTeacher(data);
      set((state) => ({ 
        teachers: [...state.teachers, newTeacher],
        isLoading: false 
      }));
    } catch (error) {
      set({ error: (error as Error).message, isLoading: false });
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
    } catch (error) {
      set({ error: (error as Error).message, isLoading: false });
    }
  },

  deleteTeacher: async (id) => {
    set({ isLoading: true, error: null });
    try {
      await hrService.deleteTeacher(id);
      set((state) => ({ 
        teachers: state.teachers.filter(t => t.id !== id),
        isLoading: false 
      }));
    } catch (error) {
      set({ error: 'Failed to delete teacher', isLoading: false });
    }
  }
}));
