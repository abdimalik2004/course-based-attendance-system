import { api } from './api';

export interface FacultyStat {
  faculty_id: number;
  faculty_name: string;
  faculty_code: string;
  total_students: number;
  total_sessions: number;
  total_records: number;
  present: number;
  late: number;
  absent: number;
  attendance_pct: number;
  at_risk_students: number;
}

export interface FacultyComparisonResponse {
  faculties: FacultyStat[];
  institution_avg: number;
  start_date: string | null;
  end_date: string | null;
}

export interface DepartmentStat {
  department_id: number;
  department_name: string;
  department_code: string;
  total_students: number;
  total_records: number;
  present: number;
  late: number;
  absent: number;
  attendance_pct: number;
}

export interface DepartmentComparisonResponse {
  faculty_id: number;
  faculty_name: string;
  departments: DepartmentStat[];
  start_date: string | null;
  end_date: string | null;
}

export interface TrendPoint {
  period: string;
  faculties: { faculty_id: number; faculty_name: string; pct: number | null }[];
}

export interface TrendsResponse {
  period: string;
  series: TrendPoint[];
  faculty_names: string[];
  start_date: string | null;
  end_date: string | null;
}

export interface CourseStat {
  course_id: number;
  course_title: string;
  course_code: string;
  faculty_name: string;
  total_records: number;
  present: number;
  attendance_pct: number;
  status: 'good' | 'warning' | 'low';
}

export interface CourseRankingResponse {
  courses: CourseStat[];
  total: number;
}

function buildParams(params: Record<string, string | number | undefined | null>) {
  const p = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== '') p.append(k, String(v));
  }
  return p.toString() ? '?' + p.toString() : '';
}

export const academiaReportsService = {
  getFacultyComparison(start_date?: string, end_date?: string): Promise<FacultyComparisonResponse> {
    return api
      .get<FacultyComparisonResponse>('/reports/faculty-comparison' + buildParams({ start_date, end_date }))
      .then(r => r.data);
  },

  getDepartmentComparison(faculty_id: number, start_date?: string, end_date?: string): Promise<DepartmentComparisonResponse> {
    return api
      .get<DepartmentComparisonResponse>('/reports/department-comparison' + buildParams({ faculty_id, start_date, end_date }))
      .then(r => r.data);
  },

  getTrends(period: 'weekly' | 'monthly', start_date?: string, end_date?: string): Promise<TrendsResponse> {
    return api
      .get<TrendsResponse>('/reports/attendance-trends' + buildParams({ period, start_date, end_date }))
      .then(r => r.data);
  },

  getCourseRanking(faculty_id?: number, start_date?: string, end_date?: string): Promise<CourseRankingResponse> {
    return api
      .get<CourseRankingResponse>('/reports/course-ranking' + buildParams({ faculty_id, start_date, end_date }))
      .then(r => r.data);
  },
};
