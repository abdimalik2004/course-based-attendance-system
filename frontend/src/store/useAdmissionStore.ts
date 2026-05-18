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
  imagesCaptured?: number;
  status: AdmissionStatus;
  createdAt: string;
}

interface AdmissionState {
  students: Student[];
  faculties: string[];
  departments: Record<string, string[]>;
  isLoading: boolean;
  isSaving: boolean;
  error: string | null;
  dashboardStats: {
    totalStudents: number;
    newAdmissions: number;
    pendingApplications: number;
    rejectedApplications: number;
  };

  fetchAdmissionData: () => Promise<void>;
  addStudent: (student: {
    fullName: string;
    faculty: string;
    department: string;
  }) => Promise<void>;
  updateStudent: (
    id: string,
    updates: {
      fullName?: string;
      faculty?: string;
      department?: string;
      status?: AdmissionStatus;
    },
  ) => Promise<void>;
  deleteStudent: (id: string) => Promise<void>;
  approveStudent: (id: string) => Promise<void>;
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
      status: student.status,
      createdAt: student.created_at,
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

export const useAdmissionStore = create<AdmissionState>((set, get) => ({
  students: [],
  faculties: [],
  departments: {},
  isLoading: false,
  isSaving: false,
  error: null,
  dashboardStats: {
    totalStudents: 0,
    newAdmissions: 0,
    pendingApplications: 0,
    rejectedApplications: 0,
  },

  fetchAdmissionData: async () => {
    set({ isLoading: true, error: null });
    try {
      const [studentsResult, facultiesResult, departmentsResult, statsResult] =
        await Promise.all([
          admissionService.listStudents({ skip: 0, limit: 200 }),
          admissionService.listFaculties(),
          admissionService.listDepartments(),
          admissionService.getDashboardStats(),
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
      const [facultiesResult, departmentsResult] = await Promise.all([
        admissionService.listFaculties(),
        admissionService.listDepartments(),
      ]);
      const facultyId = resolveFacultyIdByName(
        facultiesResult,
        student.faculty,
      );
      const departmentId = resolveDepartmentIdByName(
        departmentsResult,
        student.department,
        facultyId,
      );

      await admissionService.createStudent({
        full_name: student.fullName,
        faculty_id: facultyId,
        department_id: departmentId,
      });
      await get().fetchAdmissionData();
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

      const [facultiesResult, departmentsResult] = await Promise.all([
        admissionService.listFaculties(),
        admissionService.listDepartments(),
      ]);

      let facultyId: number | undefined;
      if (updates.faculty) {
        facultyId = resolveFacultyIdByName(facultiesResult, updates.faculty);
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
          departmentsResult,
          updates.department,
          resolvedFacultyId,
        );
      }

      await admissionService.updateStudent(studentId, {
        full_name: updates.fullName,
        faculty_id: facultyId,
        department_id: departmentId,
        status: updates.status,
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
    await get().updateStudent(id, { status: "approved" });
  },

  rejectStudent: async (id) => {
    await get().updateStudent(id, { status: "rejected" });
  },
}));
