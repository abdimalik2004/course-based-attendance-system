import type {
  User,
  CreateUserPayload,
  Role,
  Faculty,
} from "@/types/users.types";
import { api } from "./api";

export const usersService = {
  getUsers: async (params?: {
    skip?: number;
    limit?: number;
    search?: string;
    faculty_id?: number;
  }) => {
    const res = await api.get("/users", { params });
    const payload = res.data || { items: [], total: 0 };
    const items = (payload.items || []).map((u: any) => ({
      id: String(u.id),
      username: u.username,
      email: u.email || "",
      role: (u.role_names?.[0] || "STAFF") as any,
      facultyId: u.faculty_id ? String(u.faculty_id) : null,
      status: u.is_active ? "Active" : "Inactive",
      createdAt: u.created_at || new Date().toISOString(),
    }));
    return { items, total: Number(payload.total || items.length) };
  },

  createUser: async (data: CreateUserPayload) => {
    const roleName = data.role?.toString?.().trim().toUpperCase() || "STAFF";
    const payload: any = {
      username: data.username,
      email: data.email,
      password: data.password,
      role_names: [roleName],
      faculty_id: data.facultyId ? Number(data.facultyId) : undefined,
    };
    if (data.teacherId) {
      payload.teacher_id = Number(data.teacherId);
    }
    if (data.studentId) {
      payload.student_id = Number(data.studentId);
    }
    const res = await api.post("/auth/register", payload);
    const u = res.data;
    return {
      id: String(u.id || u.user_id || u.username),
      username: u.username || String(u.id),
      email: u.email || "",
      role: (u.role_names?.[0] || roleName) as any,
      facultyId: u.faculty_id ? String(u.faculty_id) : null,
      status: u.is_active ? "Active" : "Inactive",
      createdAt: u.created_at || new Date().toISOString(),
    };
  },

  updateUser: async (id: string | number, data: Partial<User>) => {
    const payload: any = {};
    if (data.username !== undefined) payload.username = data.username;
    if (data.email !== undefined) payload.email = data.email;
    if (data.facultyId !== undefined) {
      payload.faculty_id = data.facultyId ? Number(data.facultyId) : null;
    }
    if (data.status !== undefined) payload.is_active = data.status === "Active";
    if (data.role !== undefined)
      payload.role_names = [String(data.role).toUpperCase()];

    const res = await api.put(`/users/${id}`, payload);
    const u = res.data;
    return {
      id: String(u.id),
      username: u.username,
      email: u.email || "",
      role: (u.role_names?.[0] || "STAFF") as any,
      facultyId: u.faculty_id ? String(u.faculty_id) : null,
      status: u.is_active ? "Active" : "Inactive",
      createdAt: u.created_at || new Date().toISOString(),
    };
  },

  deleteUser: async (id: string | number) => {
    await api.delete(`/users/${id}`);
    return;
  },

  getRoles: async () => {
    const res = await api.get<Role[]>("/auth/roles");
    return res.data;
  },

  createRole: async (name: string) => {
    const res = await api.post<Role>("/auth/roles", { name });
    return res.data;
  },

  updateRole: async (id: string | number, name: string) => {
    const res = await api.put<Role>(`/auth/roles/${id}`, { name });
    return res.data;
  },

  deleteRole: async (id: string | number) => {
    await api.delete(`/auth/roles/${id}`);
    return;
  },

  getFaculties: async () => {
    const res = await api.get<{ items?: Faculty[] } | Faculty[]>("/faculties", {
      params: { skip: 0, limit: 200 },
    });
    return Array.isArray(res.data) ? res.data : (res.data.items ?? []);
  },
};

export default usersService;
