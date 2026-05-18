import { api } from "./api";

export const facultyService = {
  list: async (params?: { skip?: number; limit?: number }) =>
    api.get("/faculties", { params }).then((r) => r.data),
  get: async (id: number) => api.get(`/faculties/${id}`).then((r) => r.data),
  create: async (payload: any) =>
    api.post("/faculties", payload).then((r) => r.data),
  update: async (id: number, payload: any) =>
    api.put(`/faculties/${id}`, payload).then((r) => r.data),
  remove: async (id: number) =>
    api.delete(`/faculties/${id}`).then((r) => r.data),
};

export default facultyService;
