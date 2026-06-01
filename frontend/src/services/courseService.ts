import { api } from "./api";

export const courseService = {
  listCourses: async (params?: {
    faculty_id?: number;
    department_id?: number;
    skip?: number;
    limit?: number;
  }) => {
    return api.get("/courses", { params }).then((r) => r.data);
  },

  getCourse: async (id: number) =>
    api.get(`/courses/${id}`).then((r) => r.data),

  createCourse: async (payload: any) =>
    api.post("/courses", payload).then((r) => r.data),

  updateCourse: async (id: number, payload: any) =>
    api.put(`/courses/${id}`, payload).then((r) => r.data),

  deleteCourse: async (id: number) =>
    api.delete(`/courses/${id}`).then((r) => r.data),

  listAssignments: async (params?: {
    course_id?: number;
    faculty_id?: number;
    department_id?: number;
    teacher_id?: number;
    skip?: number;
    limit?: number;
  }) =>
    api
      .get("/courses/assignments", {
        params,
      })
      .then((r) => r.data),

  assignTeacher: async (payload: {
    course_id: number;
    teacher_id: number;
    is_primary?: boolean;
  }) => api.post("/courses/assign-teacher", payload).then((r) => r.data),

  updateAssignment: async (id: string, payload: any) =>
    api.put(`/courses/assignments/${id}`, payload).then((r) => r.data),

  deleteAssignment: async (id: string) =>
    api.delete(`/courses/assignments/${id}`).then((r) => r.data),
};

export default courseService;
