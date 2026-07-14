import { create } from "zustand";
import { api } from "@/services/api";
import type {
  Faculty,
  Department,
  Course,
  Class,
  AcademicStructure,
  CourseAssignment,
  ClassAssignment,
} from "@/types/academia.types";

export type ModalMode = "create" | "edit" | "view";

interface ModalState<T> {
  isOpen: boolean;
  mode: ModalMode;
  record: T | null;
}

interface AcademiaState {
  faculties: Faculty[];
  departments: Department[];
  courses: Course[];
  classes: Class[];
  structures: AcademicStructure[];
  courseAssignments: CourseAssignment[];
  classAssignments: ClassAssignment[];

  isLoading: boolean;
  error: string | null;
  lastFetchedAt: number | null;

  facultyModal: ModalState<Faculty>;
  departmentModal: ModalState<Department>;
  courseModal: ModalState<Course>;
  classModal: ModalState<Class>;
  structureModal: ModalState<AcademicStructure>;
  courseAssignModal: ModalState<CourseAssignment>;
  classAssignModal: ModalState<ClassAssignment>;

  openModal: <
    K extends
      | "faculty"
      | "department"
      | "course"
      | "class"
      | "structure"
      | "courseAssign"
      | "classAssign",
  >(
    type: K,
    mode: ModalMode,
    record?: any,
  ) => void;

  closeModal: (
    type:
      | "faculty"
      | "department"
      | "course"
      | "class"
      | "structure"
      | "courseAssign"
      | "classAssign",
  ) => void;

  fetchData: () => Promise<void>;

  // Faculty
  addFaculty: (faculty: Omit<Faculty, "id" | "createdAt">) => Promise<void>;
  updateFaculty: (id: string, updates: Partial<Faculty>) => Promise<void>;
  deleteFaculty: (id: string) => Promise<void>;

  // Department
  addDepartment: (dept: Omit<Department, "id" | "createdAt">) => Promise<void>;
  updateDepartment: (id: string, updates: Partial<Department>) => Promise<void>;
  deleteDepartment: (id: string) => Promise<void>;

  // Course
  addCourse: (course: Omit<Course, "id" | "createdAt">) => Promise<void>;
  updateCourse: (id: string, updates: Partial<Course>) => Promise<void>;
  deleteCourse: (id: string) => Promise<void>;

  // Class
  addClass: (cls: Omit<Class, "id" | "createdAt">) => Promise<void>;
  updateClass: (id: string, updates: Partial<Class>) => Promise<void>;
  deleteClass: (id: string) => Promise<void>;

  // Structure
  addStructure: (
    structure: Omit<AcademicStructure, "id" | "createdAt">,
  ) => Promise<void>;
  updateStructure: (
    id: string,
    updates: Partial<AcademicStructure>,
  ) => Promise<void>;
  deleteStructure: (id: string) => Promise<void>;

  addCourseAssignment: (
    data: Omit<CourseAssignment, "id" | "createdAt">,
  ) => Promise<void>;
  updateCourseAssignment: (
    id: string,
    updates: Partial<CourseAssignment>,
  ) => Promise<void>;
  deleteCourseAssignment: (id: string) => Promise<void>;

  addClassAssignment: (
    data: Omit<ClassAssignment, "id" | "createdAt">,
  ) => Promise<void>;
  updateClassAssignment: (
    id: string,
    updates: Partial<ClassAssignment>,
  ) => Promise<void>;
  deleteClassAssignment: (id: string) => Promise<void>;
}

const defaultModalState = {
  isOpen: false,
  mode: "create" as ModalMode,
  record: null,
};

const mapFaculty = (faculty: any): Faculty => ({
  id: String(faculty.id),
  name: faculty.name,
  code: faculty.code ?? faculty.short_code ?? "",
  years: faculty.years ?? faculty.duration_years ?? 0,
  createdAt: faculty.created_at ?? new Date().toISOString(),
});

const mapDepartment = (department: any): Department => ({
  id: String(department.id),
  facultyId: String(department.faculty_id),
  name: department.name,
  code: department.code ?? department.short_code ?? "",
  createdAt: department.created_at ?? new Date().toISOString(),
});

const mapCourse = (course: any): Course => ({
  id: String(course.id),
  facultyId: String(course.faculty_id),
  departmentId: String(course.department_id),
  title: course.title,
  code: course.code,
  createdAt: course.created_at ?? new Date().toISOString(),
});

const mapClass = (cls: any): Class => ({
  id: String(cls.id),
  facultyId: String(cls.faculty_id),
  departmentId: String(cls.department_id),
  name: cls.name ?? cls.class_name ?? "",
  year: cls.year ?? cls.batch_year ?? 0,
  createdAt: cls.created_at ?? new Date().toISOString(),
});

const mapAcademicYear = (year: any): AcademicStructure => ({
  id: String(year.id),
  academicYear: year.academic_year ?? year.term_name ?? "",
  term: year.term_name ?? year.term ?? "",
  startDate: year.start_date ?? new Date().toISOString(),
  endDate: year.end_date ?? new Date().toISOString(),
  status:
    year.status === "active" || year.status === "Active"
      ? "Active"
      : year.status === "inactive" || year.status === "Inactive"
        ? "Inactive"
        : "Draft",
  createdAt: year.created_at ?? new Date().toISOString(),
});


const mapCourseAssignment = (assignment: any): CourseAssignment => ({
  id: String(assignment.id),
  courseId: String(assignment.course_id),
  facultyId: String(assignment.faculty_id),
  departmentId: String(assignment.department_id),
  semester: assignment.semester ?? 0,
  academicYearId: String(assignment.academic_year_id ?? ""),
  createdAt: assignment.created_at ?? new Date().toISOString(),
});

const mapClassAssignment = (assignment: any): ClassAssignment => ({
  id: String(assignment.id),
  classId: String(assignment.class_id),
  courseId: String(assignment.course_id),
  facultyId: String(assignment.faculty_id),
  departmentId: String(assignment.department_id),
  createdAt: assignment.created_at ?? new Date().toISOString(),
});

export const useAcademiaStore = create<AcademiaState>((set, get) => ({
  faculties: [],
  departments: [],
  courses: [],
  classes: [],
  structures: [],
  courseAssignments: [],
  classAssignments: [],

  isLoading: false,
  error: null,
  lastFetchedAt: null,

  facultyModal: { ...defaultModalState },
  departmentModal: { ...defaultModalState },
  courseModal: { ...defaultModalState },
  classModal: { ...defaultModalState },
  structureModal: { ...defaultModalState },
  courseAssignModal: { ...defaultModalState },
  classAssignModal: { ...defaultModalState },

  openModal: (type, mode, record = null) =>
    set(() => ({
      [`${type}Modal`]: { isOpen: true, mode, record },
    })),

  closeModal: (type) =>
    set((state) => ({
      [`${type}Modal`]: {
        ...(state[`${type}Modal` as keyof AcademiaState] as any),
        isOpen: false,
      },
    })),

  fetchData: async () => {
    // Skip if data was fetched within the last 30 seconds
    const { lastFetchedAt } = get();
    if (lastFetchedAt && Date.now() - lastFetchedAt < 30_000) return;
    set({ isLoading: true, error: null });
    try {
      const [
        facultiesRes,
        departmentsRes,
        coursesRes,
        classesRes,
        academicYearsRes,
        courseAssignmentsRes,
        classAssignmentsRes,
      ] = await Promise.all([
        api.get("/faculties", { params: { skip: 0, limit: 200 } }),
        api.get("/departments", { params: { skip: 0, limit: 200 } }),
        api.get("/courses", { params: { skip: 0, limit: 200 } }),
        api.get("/classes", { params: { skip: 0, limit: 200 } }),
        api.get("/academic-structure/academic-years", {
          params: { skip: 0, limit: 200 },
        }),
        api.get("/academic-structure/course-semester-assignments", {
          params: { skip: 0, limit: 200 },
        }),
        api.get("/academic-structure/class-course-assignments", {
          params: { skip: 0, limit: 200 },
        }),
      ]);

      set({
        faculties: (facultiesRes.data?.items ?? []).map(mapFaculty),
        departments: (departmentsRes.data?.items ?? []).map(mapDepartment),
        courses: (coursesRes.data?.items ?? []).map(mapCourse),
        classes: (classesRes.data?.items ?? []).map(mapClass),
        structures: (academicYearsRes.data?.items ?? []).map(mapAcademicYear),
        courseAssignments: (courseAssignmentsRes.data?.items ?? []).map(
          mapCourseAssignment,
        ),
        classAssignments: (classAssignmentsRes.data?.items ?? []).map(
          mapClassAssignment,
        ),
        isLoading: false,
        lastFetchedAt: Date.now(),
      });
    } catch (error) {
      set({
        error:
          error instanceof Error
            ? error.message
            : "Failed to fetch academia data",
        isLoading: false,
      });
    }
  },

  // Faculty Actions
  addFaculty: async (data) => {
    const response = await api.post("/faculties", {
      name: data.name,
      code: data.code,
      years: data.years,
    });
    set((state) => ({
      faculties: [...state.faculties, mapFaculty(response.data)],
    }));
  },
  updateFaculty: async (id, updates) => {
    const response = await api.put(`/faculties/${id}`, updates);
    set((state) => ({
      faculties: state.faculties.map((faculty) =>
        faculty.id === id ? mapFaculty(response.data) : faculty,
      ),
    }));
  },
  deleteFaculty: async (id) => {
    await api.delete(`/faculties/${id}`);
    set((state) => ({
      faculties: state.faculties.filter((faculty) => faculty.id !== id),
    }));
  },

  // Department Actions
  addDepartment: async (data) => {
    const response = await api.post("/departments", {
      faculty_id: Number(data.facultyId),
      name: data.name,
      code: data.code,
    });
    set((state) => ({
      departments: [...state.departments, mapDepartment(response.data)],
    }));
  },
  updateDepartment: async (id, updates) => {
    const response = await api.put(`/departments/${id}`, {
      faculty_id: updates.facultyId ? Number(updates.facultyId) : undefined,
      name: updates.name,
      code: updates.code,
    });
    set((state) => ({
      departments: state.departments.map((department) =>
        department.id === id ? mapDepartment(response.data) : department,
      ),
    }));
  },
  deleteDepartment: async (id) => {
    await api.delete(`/departments/${id}`);
    set((state) => ({
      departments: state.departments.filter(
        (department) => department.id !== id,
      ),
    }));
  },

  // Course Actions
  addCourse: async (data) => {
    const response = await api.post("/courses", {
      faculty_id: Number(data.facultyId),
      department_id: Number(data.departmentId),
      // support both `title` (Course type) and `name` (legacy form field)
      title: (data as any).title ?? (data as any).name,
    });
    set((state) => ({ courses: [...state.courses, mapCourse(response.data)] }));
  },
  updateCourse: async (id, updates) => {
    const response = await api.put(`/courses/${id}`, {
      faculty_id: updates.facultyId ? Number(updates.facultyId) : undefined,
      department_id: updates.departmentId
        ? Number(updates.departmentId)
        : undefined,
      title: updates.title,
    });
    set((state) => ({
      courses: state.courses.map((course) =>
        course.id === id ? mapCourse(response.data) : course,
      ),
    }));
  },
  deleteCourse: async (id) => {
    await api.delete(`/courses/${id}`);
    set((state) => ({
      courses: state.courses.filter((course) => course.id !== id),
    }));
  },

  // Class Actions
  addClass: async (data) => {
    const response = await api.post("/classes", {
      faculty_id: Number(data.facultyId),
      department_id: Number(data.departmentId),
      year: data.year,
    });
    set((state) => ({ classes: [...state.classes, mapClass(response.data)] }));
  },
  updateClass: async (id, updates) => {
    const response = await api.put(`/classes/${id}`, {
      faculty_id: updates.facultyId ? Number(updates.facultyId) : undefined,
      department_id: updates.departmentId
        ? Number(updates.departmentId)
        : undefined,
      year: updates.year,
    });
    set((state) => ({
      classes: state.classes.map((cls) =>
        cls.id === id ? mapClass(response.data) : cls,
      ),
    }));
  },
  deleteClass: async (id) => {
    await api.delete(`/classes/${id}`);
    set((state) => ({ classes: state.classes.filter((cls) => cls.id !== id) }));
  },

  // Structure Actions
  addStructure: async (data) => {
    const response = await api.post("/academic-structure/academic-years", {
      academic_year: data.academicYear,
      term_name: data.term,
      start_date: data.startDate,
      end_date: data.endDate,
      // status is intentionally omitted — the backend derives it from the dates
    });
    set((state) => ({
      structures: [...state.structures, mapAcademicYear(response.data)],
    }));
  },
  updateStructure: async (id, updates) => {
    const response = await api.put(`/academic-structure/academic-years/${id}`, {
      academic_year: updates.academicYear,
      term_name: updates.term,
      start_date: updates.startDate,
      end_date: updates.endDate,
      // status is intentionally omitted — the backend derives it from the dates
    });
    set((state) => ({
      structures: state.structures.map((structure) =>
        structure.id === id ? mapAcademicYear(response.data) : structure,
      ),
    }));
  },
  deleteStructure: async (id) => {
    await api.delete(`/academic-structure/academic-years/${id}`);
    set((state) => ({
      structures: state.structures.filter((structure) => structure.id !== id),
    }));
  },

  addCourseAssignment: async (data) => {
    const response = await api.post(
      "/academic-structure/course-semester-assignments",
      {
        course_id: Number(data.courseId),
        faculty_id: Number(data.facultyId),
        department_id: Number(data.departmentId),
        semester: Number(data.semester),
      },
    );
    set((state) => ({
      courseAssignments: [
        ...state.courseAssignments,
        mapCourseAssignment(response.data),
      ],
    }));
  },
  updateCourseAssignment: async (id, updates) => {
    // Only `semester` is updatable on an existing course-semester assignment
    const response = await api.put(
      `/academic-structure/course-semester-assignments/${id}`,
      { semester: Number(updates.semester) },
    );
    set((state) => ({
      courseAssignments: state.courseAssignments.map((assignment) =>
        assignment.id === id ? mapCourseAssignment(response.data) : assignment,
      ),
    }));
  },
  deleteCourseAssignment: async (id) => {
    await api.delete(`/academic-structure/course-semester-assignments/${id}`);
    set((state) => ({
      courseAssignments: state.courseAssignments.filter(
        (assignment) => assignment.id !== id,
      ),
    }));
  },

  addClassAssignment: async (data) => {
    const response = await api.post(
      "/academic-structure/class-course-assignments",
      {
        class_id: Number(data.classId),
        course_id: Number(data.courseId),
        faculty_id: Number(data.facultyId),
        department_id: Number(data.departmentId),
      },
    );
    set((state) => ({
      classAssignments: [
        ...state.classAssignments,
        mapClassAssignment(response.data),
      ],
    }));
  },
  updateClassAssignment: async (id, updates) => {
    const response = await api.put(
      `/academic-structure/class-course-assignments/${id}`,
      {
        class_id: updates.classId ? Number(updates.classId) : undefined,
        course_id: updates.courseId ? Number(updates.courseId) : undefined,
        faculty_id: updates.facultyId ? Number(updates.facultyId) : undefined,
        department_id: updates.departmentId
          ? Number(updates.departmentId)
          : undefined,
      },
    );
    set((state) => ({
      classAssignments: state.classAssignments.map((assignment) =>
        assignment.id === id ? mapClassAssignment(response.data) : assignment,
      ),
    }));
  },
  deleteClassAssignment: async (id) => {
    await api.delete(`/academic-structure/class-course-assignments/${id}`);
    set((state) => ({
      classAssignments: state.classAssignments.filter(
        (assignment) => assignment.id !== id,
      ),
    }));
  },
}));
