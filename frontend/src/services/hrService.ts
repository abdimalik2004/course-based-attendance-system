import { api } from "./api";

export interface Teacher {
  id: string;
  fullName: string;
  facultyId: string;
  departmentId: string;
  userId: string;
  status: "Active" | "On Leave" | "Inactive";
  role: string;
}

export interface Faculty {
  id: string;
  name: string;
}

export interface Department {
  id: string;
  facultyId: string;
  name: string;
}

export const hrService = {
  getTeachers: async (): Promise<Teacher[]> => {
    const response = await api.get("/teachers", {
      params: { skip: 0, limit: 200 },
    });
    return (response.data?.items ?? []).map((teacher: any) => ({
      id: String(teacher.id),
      fullName: teacher.full_name ?? teacher.name ?? "",
      facultyId: String(teacher.faculty_id ?? ""),
      departmentId: String(teacher.department_id ?? ""),
      userId: teacher.user_id != null ? String(teacher.user_id) : "",
      status: teacher.status ?? "Active",
      role: teacher.role ?? "Lecturer",
    }));
  },

  addTeacher: async (
    data: Omit<Teacher, "id" | "status" | "userId">,
  ): Promise<Teacher> => {
    const response = await api.post("/teachers", {
      full_name: data.fullName,
      faculty_id: Number(data.facultyId),
      department_id: Number(data.departmentId),
      role: data.role,
      status: "Active",
    });
    const teacher = response.data;
    return {
      id: String(teacher.id),
      fullName: teacher.full_name ?? teacher.name ?? data.fullName,
      facultyId: String(teacher.faculty_id ?? data.facultyId),
      departmentId: String(teacher.department_id ?? data.departmentId),
      userId: teacher.user_id != null ? String(teacher.user_id) : "",
      status: teacher.status ?? "Active",
      role: teacher.role ?? data.role,
    };
  },

  deleteTeacher: async (id: string): Promise<boolean> => {
    await api.delete(`/teachers/${id}`);
    return true;
  },

  updateTeacher: async (
    id: string,
    data: Partial<Teacher>,
  ): Promise<Teacher> => {
    const response = await api.put(`/teachers/${id}`, {
      full_name: data.fullName,
      faculty_id: data.facultyId ? Number(data.facultyId) : undefined,
      department_id: data.departmentId ? Number(data.departmentId) : undefined,
      status: data.status,
      role: data.role,
    });
    const teacher = response.data;
    return {
      id: String(teacher.id),
      fullName: teacher.full_name ?? teacher.name ?? data.fullName ?? "",
      facultyId: String(teacher.faculty_id ?? data.facultyId ?? ""),
      departmentId: String(teacher.department_id ?? data.departmentId ?? ""),
      userId: teacher.user_id != null ? String(teacher.user_id) : "",
      status: teacher.status ?? data.status ?? "Active",
      role: teacher.role ?? data.role ?? "Lecturer",
    };
  },

  getFaculties: async (): Promise<Faculty[]> => {
    const response = await api.get("/faculties", {
      params: { skip: 0, limit: 200 },
    });
    return (response.data?.items ?? []).map((faculty: any) => ({
      id: String(faculty.id),
      name: faculty.name,
    }));
  },

  getDepartments: async (): Promise<Department[]> => {
    const response = await api.get("/departments", {
      params: { skip: 0, limit: 200 },
    });
    return (response.data?.items ?? []).map((department: any) => ({
      id: String(department.id),
      facultyId: String(department.faculty_id),
      name: department.name,
    }));
  },

  getDepartmentsByFaculty: async (facultyId: string): Promise<Department[]> => {
    const response = await api.get("/departments", {
      params: { skip: 0, limit: 200, faculty_id: Number(facultyId) },
    });
    return (response.data?.items ?? []).map((department: any) => ({
      id: String(department.id),
      facultyId: String(department.faculty_id),
      name: department.name,
    }));
  },
};
