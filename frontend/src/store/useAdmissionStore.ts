import { create } from "zustand";
import admissionService, {
  type AdmissionDashboardStatsDto,
  type AdmissionStatus,
  type AdmissionStudentDto,
  type DepartmentDto,
  type FacultyDto,
} from "@/services/admissionService";

export interface Student {
  id: string;
  studentNumber: string;
  fullName: string;
  facultyId: string;
  departmentId: string;
  faculty: string;
  department: string;
  class: string;
  faceImagesCount: number;
  status: AdmissionStatus;
  createdAt: string;
  dateOfBirth?: string | null;
  phone?: string | null;
  email?: string | null;
}

interface AdmissionState {
  students: Student[];
  /** Human-readable display lists (derived from raw DTOs) */
  faculties: string[];
  departments: Record<string, string[]>;
  /** Raw DTO caches — used by addStudent/updateStudent to avoid re-fetching */
  facultyDtos: FacultyDto[];
  departmentDtos: DepartmentDto[];
  /** Pagination */
  total: number;
  currentPage: number;
  pageSize: number;
  /** Last applied server-side filters — persisted so page navigation preserves them */
  currentSearch: string;
  currentStatus: string;
  isLoading: boolean;
  isSaving: boolean;
  error: string | null;
  dashboardStats: {
    totalStudents: number;
    newAdmissions: number;
    pendingApplications: number;
    rejectedApplications: number;
  };

  fetchAdmissionData: (opts?: { page?: number; pageSize?: number; search?: string; status?: string }) => Promise<void>;
  addStudent: (student: {
    fullName: string;
    faculty: string;
    department: string;
    dateOfBirth?: string | null;
    phone?: string | null;
    email?: string | null;
  }) => Promise<{ studentNumber: string; generatedPassword: string | null } | null>;
  updateStudent: (
    id: string,
    updates: {
      fullName?: string;
      faculty?: string;
      department?: string;
      status?: AdmissionStatus;
      dateOfBirth?: string | null;
      phone?: string | null;
      email?: string | null;
    },
  ) => Promise<void>;
  deleteStudent: (id: string) => Promise<void>;
  approveStudent: (id: string) => Promise<{ studentNumber: string; generatedPassword: string | null } | null>;
  rejectStudent: (id: string) => Promise<void>;
}

const mapDashboardStats = (
  stats: AdmissionDashboardStatsDto,
): AdmissionState["dashboardStats"] => ({
  totalStudents: Number(stats.total_students ?? 0),
  newAdmissions: Number(stats.new_admissions ?? 0),
  pendingApplications: Number(stats.pending_admissions ?? 0),
  rejectedApplications: Number(stats.rejected_applications ?? 0),
});

const toClassLabel = (createdAt: string) => {
  const year = new Date(createdAt).getFullYear();
  return Number.isNaN(year) ? "Class N/A" : `Class ${year}`;
};

const normalizeAdmissionState = (
  students: AdmissionStudentDto[],
  faculties: FacultyDto[],
  departments: DepartmentDto[],
) => {
  const facultyIdToName = new Map(
    faculties.map((faculty) => [faculty.id, faculty.name]),
  );
  const departmentIdToName = new Map(
    departments.map((department) => [department.id, department.name]),
  );

  const mappedStudents: Student[] = students.map((student) => {
    const facultyName =
      facultyIdToName.get(student.faculty_id) ??
      `Faculty ${student.faculty_id}`;
    const departmentName =
      departmentIdToName.get(student.department_id) ??
      `Department ${student.department_id}`;
    return {
      id: String(student.id),
      studentNumber: student.student_number,
      fullName: student.full_name,
      facultyId: String(student.faculty_id),
      departmentId: String(student.department_id),
      faculty: facultyName,
      department: departmentName,
      class: toClassLabel(student.created_at),
      faceImagesCount: student.face_images_count ?? 0,
      status: student.status,
      createdAt: student.created_at,
      dateOfBirth: student.date_of_birth ?? null,
      phone: student.phone ?? null,
      email: student.email ?? null,
    };
  });

  const facultyNames = faculties.map((faculty) => faculty.name);
  const departmentsByFaculty: Record<string, string[]> = {};
  departments.forEach((department) => {
    const facultyName = facultyIdToName.get(department.faculty_id);
    if (!facultyName) return;
    if (!departmentsByFaculty[facultyName]) {
      departmentsByFaculty[facultyName] = [];
    }
    departmentsByFaculty[facultyName].push(department.name);
  });

  return {
    mappedStudents,
    facultyNames,
    departmentsByFaculty,
  };
};

const resolveFacultyIdByName = (
  faculties: FacultyDto[],
  facultyName: string,
): number => {
  const match = faculties.find((faculty) => faculty.name === facultyName);
  if (!match) {
    throw new Error(`Faculty '${facultyName}' was not found in the database.`);
  }
  return match.id;
};

const resolveDepartmentIdByName = (
  departments: DepartmentDto[],
  departmentName: string,
  facultyId: number,
): number => {
  const match = departments.find(
    (department) =>
      department.name === departmentName && department.faculty_id === facultyId,
  );
  if (!match) {
    throw new Error(
      `Department '${departmentName}' was not found for the selected faculty.`,
    );
  }
  return match.id;
};

const DEFAULT_PAGE_SIZE = 50;

export const useAdmissionStore = create<AdmissionState>((set, get) => ({
  students: [],
  faculties: [],
  departments: {},
  facultyDtos: [],
  departmentDtos: [],
  total: 0,
  currentPage: 1,
  pageSize: DEFAULT_PAGE_SIZE,
  currentSearch: '',
  currentStatus: 'All',
  isLoading: false,
  isSaving: false,
  error: null,
  dashboardStats: {
    totalStudents: 0,
    newAdmissions: 0,
    pendingApplications: 0,
    rejectedApplications: 0,
  },

  fetchAdmissionData: async (opts) => {
    const page = opts?.page ?? get().currentPage;
    const pageSize = opts?.pageSize ?? get().pageSize;
    const skip = (page - 1) * pageSize;
    // Preserve existing search/status if not explicitly overridden
    const search = opts?.search !== undefined ? opts.search : get().currentSearch;
    const status = opts?.status !== undefined ? opts.status : get().currentStatus;

    set({ isLoading: true, error: null, currentPage: page, pageSize, currentSearch: search, currentStatus: status });
    try {
      // Only re-fetch faculties/departments if the cache is empty
      const state = get();
      const needsDtos =
        state.facultyDtos.length === 0 || state.departmentDtos.length === 0;

      const resolvedStatus = status && status !== 'All' ? (status.toLowerCase() as AdmissionStatus) : undefined;

      const [studentsResult, statsResult, facultiesResult, departmentsResult] =
        await Promise.all([
          admissionService.listStudents({
            skip,
            limit: pageSize,
            search: search || undefined,
            status: resolvedStatus,
          }),
          admissionService.getDashboardStats(),
          needsDtos ? admissionService.listFaculties() : Promise.resolve(state.facultyDtos),
          needsDtos ? admissionService.listDepartments() : Promise.resolve(state.departmentDtos),
        ]);

      const normalized = normalizeAdmissionState(
        studentsResult.items,
        facultiesResult,
        departmentsResult,
      );

      set({
        students: normalized.mappedStudents,
        faculties: normalized.facultyNames,
        departments: normalized.departmentsByFaculty,
        facultyDtos: facultiesResult,
        departmentDtos: departmentsResult,
        total: studentsResult.total,
        dashboardStats: mapDashboardStats(statsResult),
        isLoading: false,
        error: null,
      });
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "Failed to load admission data.";
      set({ isLoading: false, error: message });
    }
  },

  addStudent: async (student) => {
    set({ isSaving: true, error: null });
    try {
      // Use cached DTOs — no extra network request
      const { facultyDtos, departmentDtos } = get();
      if (facultyDtos.length === 0 || departmentDtos.length === 0) {
        throw new Error("Faculty and department data not loaded. Please refresh the page.");
      }

      const facultyId = resolveFacultyIdByName(facultyDtos, student.faculty);
      const departmentId = resolveDepartmentIdByName(
        departmentDtos,
        student.department,
        facultyId,
      );

      const created = await admissionService.createStudent({
        full_name: student.fullName,
        faculty_id: facultyId,
        department_id: departmentId,
        date_of_birth: student.dateOfBirth ?? null,
        phone: student.phone ?? null,
        email: student.email ?? null,
      });

      await get().fetchAdmissionData();
      return {
        studentNumber: created.student_number,
        generatedPassword: created.generated_password ?? null,
      };
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Failed to create student.";
      set({ error: message });
      throw error;
    } finally {
      set({ isSaving: false });
    }
  },

  updateStudent: async (id, updates) => {
    set({ isSaving: true, error: null });
    try {
      const studentId = Number(id);
      if (Number.isNaN(studentId)) {
        throw new Error("Invalid student id.");
      }

      // Use cached DTOs — no extra network request
      const { facultyDtos, departmentDtos } = get();

      let facultyId: number | undefined;
      if (updates.faculty) {
        facultyId = resolveFacultyIdByName(facultyDtos, updates.faculty);
      }

      let departmentId: number | undefined;
      if (updates.department) {
        const existingStudent = get().students.find(
          (student) => student.id === id,
        );
        const resolvedFacultyId =
          facultyId ??
          (existingStudent ? Number(existingStudent.facultyId) : undefined);
        if (!resolvedFacultyId) {
          throw new Error(
            "Unable to resolve selected faculty for department update.",
          );
        }
        departmentId = resolveDepartmentIdByName(
          departmentDtos,
          updates.department,
          resolvedFacultyId,
        );
      }

      await admissionService.updateStudent(studentId, {
        full_name: updates.fullName,
        faculty_id: facultyId,
        department_id: departmentId,
        status: updates.status,
        date_of_birth: updates.dateOfBirth,
        phone: updates.phone,
        email: updates.email,
      });

      await get().fetchAdmissionData();
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Failed to update student.";
      set({ error: message });
      throw error;
    } finally {
      set({ isSaving: false });
    }
  },

  deleteStudent: async (id) => {
    set({ isSaving: true, error: null });
    try {
      const studentId = Number(id);
      if (Number.isNaN(studentId)) {
        throw new Error("Invalid student id.");
      }

      await admissionService.deleteStudent(studentId);
      await get().fetchAdmissionData();
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Failed to delete student.";
      set({ error: message });
      throw error;
    } finally {
      set({ isSaving: false });
    }
  },

  approveStudent: async (id) => {
    set({ isSaving: true, error: null });
    try {
      const studentId = Number(id);
      const result = await admissionService.updateStudent(studentId, { status: "approved" });
      await get().fetchAdmissionData();
      return {
        studentNumber: result.student_number,
        generatedPassword: result.generated_password ?? null,
      };
    } catch (error) {
      const message = error instanceof Error ? error.message : "Failed to approve student.";
      set({ error: message });
      throw error;
    } finally {
      set({ isSaving: false });
    }
  },

  rejectStudent: async (id) => {
    await get().updateStudent(id, { status: "rejected" });
  },
}));
