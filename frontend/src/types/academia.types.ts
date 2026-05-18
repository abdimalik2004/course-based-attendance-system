export interface Faculty {
  id: string;
  name: string;
  code: string;
  years: number;
  createdAt: string;
}

export interface Department {
  id: string;
  facultyId: string;
  name: string;
  code: string;
  createdAt: string;
}

export interface Course {
  id: string;
  facultyId: string;
  departmentId: string;
  title: string;
  code: string;
  createdAt: string;
}

export interface Class {
  id: string;
  facultyId: string;
  departmentId: string;
  name: string;
  year: number;
  createdAt: string;
}

// Keeping Structure generic since it wasn't specified to change, but ensuring createdAt is tracked if needed.
export interface AcademicStructure {
  id: string;
  academicYear: string;
  term: string;
  startDate: string;
  endDate: string;
  status: 'Active' | 'Inactive' | 'Draft';
  createdAt: string;
}

export interface CourseAssignment {
  id: string;
  courseId: string;
  facultyId: string;
  departmentId: string;
  semester: number;
  createdAt: string;
}

export interface ClassAssignment {
  id: string;
  classId: string;
  courseId: string;
  facultyId: string;
  departmentId: string;
  createdAt: string;
}
