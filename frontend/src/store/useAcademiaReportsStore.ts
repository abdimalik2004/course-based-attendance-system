import { create } from 'zustand';
import {
  academiaReportsService,
  type FacultyComparisonResponse,
  type DepartmentComparisonResponse,
  type TrendsResponse,
  type CourseRankingResponse,
} from '@/services/academiaReportsService';

export type DatePreset = 'all' | 'week' | 'month' | 'semester' | 'custom';
export type TrendPeriod = 'weekly' | 'monthly';

interface Filters {
  preset: DatePreset;
  startDate: string;
  endDate: string;
  trendPeriod: TrendPeriod;
  drillFacultyId: number | null;
  courseFilterFacultyId: number | null;
}

interface AcademiaReportsState {
  filters: Filters;
  comparison: FacultyComparisonResponse | null;
  departments: DepartmentComparisonResponse | null;
  trends: TrendsResponse | null;
  courseRanking: CourseRankingResponse | null;
  loading: { comparison: boolean; departments: boolean; trends: boolean; courses: boolean };
  error: string | null;

  setFilters: (patch: Partial<Filters>) => void;
  fetchAll: () => Promise<void>;
  fetchComparison: () => Promise<void>;
  fetchDepartments: (facultyId: number) => Promise<void>;
  fetchTrends: () => Promise<void>;
  fetchCourseRanking: () => Promise<void>;
}

export const useAcademiaReportsStore = create<AcademiaReportsState>((set, get) => ({
  filters: {
    preset: 'all',
    startDate: '',
    endDate: '',
    trendPeriod: 'monthly',
    drillFacultyId: null,
    courseFilterFacultyId: null,
  },
  comparison: null,
  departments: null,
  trends: null,
  courseRanking: null,
  loading: { comparison: false, departments: false, trends: false, courses: false },
  error: null,

  setFilters: (patch) => {
    set(s => ({ filters: { ...s.filters, ...patch } }));
  },

  fetchAll: async () => {
    const { fetchComparison, fetchTrends, fetchCourseRanking } = get();
    await Promise.all([fetchComparison(), fetchTrends(), fetchCourseRanking()]);
  },

  fetchComparison: async () => {
    const { filters } = get();
    set(s => ({ loading: { ...s.loading, comparison: true }, error: null }));
    try {
      const data = await academiaReportsService.getFacultyComparison(
        filters.startDate || undefined,
        filters.endDate || undefined,
      );
      set(s => ({ comparison: data, loading: { ...s.loading, comparison: false } }));
    } catch {
      set(s => ({ loading: { ...s.loading, comparison: false }, error: 'Failed to load faculty comparison' }));
    }
  },

  fetchDepartments: async (facultyId: number) => {
    const { filters } = get();
    set(s => ({ loading: { ...s.loading, departments: true }, error: null }));
    try {
      const data = await academiaReportsService.getDepartmentComparison(
        facultyId,
        filters.startDate || undefined,
        filters.endDate || undefined,
      );
      set(s => ({
        departments: data,
        filters: { ...s.filters, drillFacultyId: facultyId },
        loading: { ...s.loading, departments: false },
      }));
    } catch {
      set(s => ({ loading: { ...s.loading, departments: false }, error: 'Failed to load department data' }));
    }
  },

  fetchTrends: async () => {
    const { filters } = get();
    set(s => ({ loading: { ...s.loading, trends: true }, error: null }));
    try {
      const data = await academiaReportsService.getTrends(
        filters.trendPeriod,
        filters.startDate || undefined,
        filters.endDate || undefined,
      );
      set(s => ({ trends: data, loading: { ...s.loading, trends: false } }));
    } catch {
      set(s => ({ loading: { ...s.loading, trends: false }, error: 'Failed to load trend data' }));
    }
  },

  fetchCourseRanking: async () => {
    const { filters } = get();
    set(s => ({ loading: { ...s.loading, courses: true }, error: null }));
    try {
      const data = await academiaReportsService.getCourseRanking(
        filters.courseFilterFacultyId ?? undefined,
        filters.startDate || undefined,
        filters.endDate || undefined,
      );
      set(s => ({ courseRanking: data, loading: { ...s.loading, courses: false } }));
    } catch {
      set(s => ({ loading: { ...s.loading, courses: false }, error: 'Failed to load course ranking' }));
    }
  },
}));
