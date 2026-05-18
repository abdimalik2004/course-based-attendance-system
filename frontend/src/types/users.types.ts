export type UserRole =
  | "Admin"
  | "Faculty"
  | "Teacher"
  | "Student"
  | "Staff"
  | "SUPER_ADMIN"
  | "ACADEMIA"
  | "FACULTY"
  | "TEACHER"
  | "HR"
  | "ADMISSIONS"
  | "STUDENT"
  | "FACULTY_ADMIN"
  | string;
export type UserStatus = "Active" | "Inactive";

export interface Faculty {
  id: string;
  name: string;
}

export interface Role {
  id: string;
  name: UserRole;
}

export interface User {
  id: string;
  username: string;
  email: string;
  role: UserRole;
  facultyId: string | null;
  identifier?: string | null;
  status: UserStatus;
  createdAt: string;
}

export interface CreateUserPayload {
  username: string;
  email: string;
  password?: string;
  role: UserRole;
  facultyId?: string | null;
  teacherId?: string;
  studentId?: string;
}
