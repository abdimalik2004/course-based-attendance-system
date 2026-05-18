import { api } from "./api";

export type AdmissionStatus = "pending" | "approved" | "rejected";

export interface AdmissionStudentDto {
  id: number;
  student_number: string;
  full_name: string;
  faculty_id: number;
  department_id: number;
  embedding_ref?: string | null;
  status: AdmissionStatus;
  created_at: string;
}

export interface AdmissionDashboardStatsDto {
  total_students: number;
  new_admissions: number;
  pending_admissions: number;
  rejected_applications: number;
}

export interface FacultyDto {
  id: number;
  name: string;
  code: string;
}

export interface DepartmentDto {
  id: number;
  faculty_id: number;
  name: string;
  code: string;
}

export interface StudentCapturedImageDto {
  file_name: string;
  url: string;
}

export interface StudentCapturedImagesDto {
  student_id: number;
  student_number: string;
  image_count: number;
  images: StudentCapturedImageDto[];
}

interface PaginatedResponse<T> {
  items: T[];
  total: number;
  skip: number;
  limit: number;
}

interface StudentListParams {
  skip?: number;
  limit?: number;
  search?: string;
  faculty_id?: number;
  department_id?: number;
  status?: AdmissionStatus;
}

export const admissionService = {
  getDashboardStats: async (): Promise<AdmissionDashboardStatsDto> => {
    const response =
      await api.get<AdmissionDashboardStatsDto>("/students/stats");
    return response.data;
  },

  listStudents: async (
    params?: StudentListParams,
  ): Promise<PaginatedResponse<AdmissionStudentDto>> => {
    const response = await api.get<PaginatedResponse<AdmissionStudentDto>>(
      "/students",
      {
        params: { skip: 0, limit: 200, ...params },
      },
    );
    return response.data;
  },

  listFaculties: async (): Promise<FacultyDto[]> => {
    const response = await api.get<PaginatedResponse<FacultyDto>>(
      "/faculties",
      {
        params: { skip: 0, limit: 200 },
      },
    );
    return response.data?.items ?? [];
  },

  listDepartments: async (): Promise<DepartmentDto[]> => {
    const response = await api.get<PaginatedResponse<DepartmentDto>>(
      "/departments",
      {
        params: { skip: 0, limit: 200 },
      },
    );
    return response.data?.items ?? [];
  },

  createStudent: async (payload: {
    full_name: string;
    faculty_id: number;
    department_id: number;
  }): Promise<AdmissionStudentDto> => {
    const response = await api.post<AdmissionStudentDto>("/students", payload);
    return response.data;
  },

  updateStudent: async (
    studentId: number,
    payload: {
      full_name?: string;
      faculty_id?: number;
      department_id?: number;
      status?: AdmissionStatus;
    },
  ): Promise<AdmissionStudentDto> => {
    const response = await api.put<AdmissionStudentDto>(
      `/students/${studentId}`,
      payload,
    );
    return response.data;
  },

  deleteStudent: async (studentId: number): Promise<void> => {
    await api.delete(`/students/${studentId}`);
  },

  getStudentCapturedImages: async (
    studentId: number,
  ): Promise<StudentCapturedImagesDto> => {
    const response = await api.get<StudentCapturedImagesDto>(
      `/students/${studentId}/captured-images`,
    );
    return response.data;
  },

  getStudentCapturedImageBlob: async (
    studentId: number,
    fileName: string,
  ): Promise<Blob> => {
    const response = await api.get(
      `/students/${studentId}/captured-images/${encodeURIComponent(fileName)}`,
      {
        responseType: "blob",
      },
    );
    return response.data;
  },

  listRecentStudents: async (params?: { skip?: number; limit?: number }) => {
    const data = await admissionService.listStudents(params);
    return {
      items: data.items,
      total: Number(data.total ?? 0),
    };
  },
};

export default admissionService;
