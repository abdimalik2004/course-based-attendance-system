import { api } from "./api";

export interface Teacher {
  id: string;
  teacherNumber: string;
  fullName: string;
  facultyId: string;
  departmentId: string;
  userId: string;
  linkedUsername: string | null;
  status: "Active" | "On Leave" | "Inactive";
  role: string;
  phone: string | null;
  email: string | null;
  hireDate: string | null; // ISO date string "YYYY-MM-DD"
}

export interface AvailableUser {
  id: number;
  username: string;
  email: string | null;
}

export interface Faculty {
  id: string;
  name: string;
}

export interface Department {
  id: string;
  facultyId: string;
  name: string;
  code: string;
}

function mapTeacher(t: any): Teacher {
  return {
    id: String(t.id),
    teacherNumber: t.teacher_number ?? "",
    fullName: t.full_name ?? t.name ?? "",
    facultyId: String(t.faculty_id ?? ""),
    departmentId: String(t.department_id ?? ""),
    userId: t.user_id != null ? String(t.user_id) : "",
    linkedUsername: t.linked_username ?? null,
    status: t.status ?? "Active",
    role: t.role ?? "Lecturer",
    phone: t.phone ?? null,
    email: t.email ?? null,
    hireDate: t.hire_date ?? null,
  };
}

export const hrService = {
  getTeachers: async (): Promise<Teacher[]> => {
    const response = await api.get("/teachers", {
      params: { skip: 0, limit: 200 },
    });
    return (response.data?.items ?? []).map((teacher: any) => mapTeacher(teacher));
  },

  addTeacher: async (
    data: Omit<Teacher, "id" | "status" | "userId" | "linkedUsername">,
  ): Promise<Teacher> => {
    const response = await api.post("/teachers", {
      full_name: data.fullName,
      faculty_id: Number(data.facultyId),
      department_id: Number(data.departmentId),
      role: data.role,
      status: "Active",
      phone: data.phone || null,
      email: data.email || null,
      hire_date: data.hireDate || null,
    });
    return mapTeacher(response.data);
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
      phone: data.phone !== undefined ? (data.phone || null) : undefined,
      email: data.email !== undefined ? (data.email || null) : undefined,
      hire_date: data.hireDate !== undefined ? (data.hireDate || null) : undefined,
    });
    return mapTeacher(response.data);
  },

  linkUser: async (teacherId: string, userId: string | null): Promise<Teacher> => {
    const response = await api.patch(`/teachers/${teacherId}/link-user`, {
      user_id: userId != null ? Number(userId) : null,
    });
    return mapTeacher(response.data);
  },

  getAvailableUsers: async (): Promise<AvailableUser[]> => {
    const response = await api.get("/teachers/available-users");
    return response.data ?? [];
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
      code: department.code ?? "",
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
      code: department.code ?? "",
    }));
  },
};
