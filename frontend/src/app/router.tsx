import { lazy } from 'react';
import { createBrowserRouter, Navigate } from 'react-router-dom';
const Login = lazy(() => import('@/pages/auth/Login'));
import { ProtectedRoute, AccessDenied } from '@/components/ProtectedRoute';

const AdminLayout = lazy(() => import('@/layouts/AdminLayout'));
const AdminDashboard = lazy(() => import('@/pages/admin/Dashboard'));
const UsersManagement = lazy(() => import('@/pages/admin/UsersManagement'));
const SystemReports = lazy(() => import('@/pages/admin/SystemReports'));
const Attendance = lazy(() => import('@/pages/admin/Attendance'));
const AdminAttendanceList = lazy(() => import('@/pages/admin/AttendanceList'));
const AdminActivityLog = lazy(() => import('@/pages/admin/ActivityLog'));
const RolesManagement = lazy(() => import('@/pages/admin/RolesManagement'));
const SettingsLayout = lazy(() => import('@/pages/admin/settings/SettingsLayout'));
// Admin cross-module views (reuse existing page components)
const AdminStudents = lazy(() => import('@/pages/admission/Students'));
const AdminTeachers = lazy(() => import('@/pages/hr/Teachers'));
const AdminFaculties = lazy(() => import('@/pages/academia/Faculties'));

// Shared profile page (Admin, HR, Academia, Admissions, Faculty)
const UserProfile = lazy(() => import('@/pages/shared/UserProfile'));

// Academia Imports
const AcademiaLayout = lazy(() => import('@/components/academia/Layout'));
const AcademiaDashboard = lazy(() => import('@/pages/academia/Dashboard'));
const Faculties = lazy(() => import('@/pages/academia/Faculties'));
const Departments = lazy(() => import('@/pages/academia/Departments'));
const Courses = lazy(() => import('@/pages/academia/Courses'));
const Classes = lazy(() => import('@/pages/academia/Classes'));
const AcademicStructure = lazy(() => import('@/pages/academia/Academic-Structure'));
const AcademiaReports = lazy(() => import('@/pages/academia/Reports'));

// HR Imports
const HRLayout = lazy(() => import('@/components/hr/Layout'));
const HRDashboard = lazy(() => import('@/pages/hr/Dashboard'));
const HRTeachers = lazy(() => import('@/pages/hr/Teachers'));
const HRReports = lazy(() => import('@/pages/hr/Reports'));
const HRSettings = lazy(() => import('@/pages/hr/Settings'));

// Admission Imports
const AdmissionLayout = lazy(() => import('@/components/admission/Layout'));
const AdmissionDashboard = lazy(() => import('@/pages/admission/Dashboard'));
const AdmissionStudents = lazy(() => import('@/pages/admission/Students'));
const AdmissionStudentsApproval = lazy(() => import('@/pages/admission/StudentsApproval'));
const AdmissionFaceRegistration = lazy(() => import('@/pages/admission/FaceRegistration'));

// Faculty Imports
const FacultyLayout = lazy(() => import('@/components/faculty/Layout'));
const FacultyDashboard = lazy(() => import('@/pages/faculty/Dashboard'));
const AssignTeacher = lazy(() => import('@/pages/faculty/AssignTeacher'));
const ScheduleCourse = lazy(() => import('@/pages/faculty/ScheduleCourse'));
const FacultyAttendanceList = lazy(() => import('@/pages/faculty/AttendanceList'));
// Faculty scoped list views (clicked from dashboard stat cards)
const FacultyStudents = lazy(() => import('@/pages/faculty/Students'));
const FacultyTeachers = lazy(() => import('@/pages/faculty/Teachers'));
const FacultyDepartments = lazy(() => import('@/pages/faculty/Departments'));
const FacultyClasses = lazy(() => import('@/pages/faculty/Classes'));
const FacultyCourses = lazy(() => import('@/pages/faculty/Courses'));
const FacultyExcuseRequests = lazy(() => import('@/pages/faculty/ExcuseRequests'));

// Teacher Imports
const TeacherLayout = lazy(() => import('@/components/teacher/Layout'));
const TeacherDashboard = lazy(() => import('@/pages/teacher/Dashboard'));
const TeacherAttendance = lazy(() => import('@/pages/teacher/Attendance'));
const TeacherAttendanceList = lazy(() => import('@/pages/teacher/AttendanceList'));
const TeacherSchedule = lazy(() => import('@/pages/teacher/Schedule'));
const TeacherProfile = lazy(() => import('@/pages/teacher/Profile'));
const TeacherCourses = lazy(() => import('@/pages/teacher/Courses'));

// Student Imports
const StudentLayout = lazy(() => import('@/components/student/Layout'));
const StudentDashboard = lazy(() => import('@/pages/student/Dashboard'));
const StudentAttendance = lazy(() => import('@/pages/student/Attendance'));
const StudentSchedule = lazy(() => import('@/pages/student/Schedule'));
const StudentProfile = lazy(() => import('@/pages/student/Profile'));

export const router = createBrowserRouter([
  {
    path: '/login',
    element: <Login />,
  },
  {
    path: '/faculty',
    element: <ProtectedRoute allowedRole="faculty" />,
    errorElement: <AccessDenied />,
    children: [
      {
        path: '',
        element: <FacultyLayout />,
        children: [
          {
            index: true,
            element: <Navigate to="dashboard" replace />,
          },
          {
            path: 'dashboard',
            element: <FacultyDashboard />,
          },
          {
            path: 'assign-teacher',
            element: <AssignTeacher />,
          },
          {
            path: 'schedule',
            element: <ScheduleCourse />,
          },
          {
            path: 'attendance-list',
            element: <FacultyAttendanceList />,
          },
          {
            path: 'students',
            element: <FacultyStudents />,
          },
          {
            path: 'teachers',
            element: <FacultyTeachers />,
          },
          {
            path: 'departments',
            element: <FacultyDepartments />,
          },
          {
            path: 'classes',
            element: <FacultyClasses />,
          },
          {
            path: 'courses',
            element: <FacultyCourses />,
          },
          {
            path: 'excuse-requests',
            element: <FacultyExcuseRequests />,
          },
          {
            path: 'profile',
            element: <UserProfile />,
          },
          {
            path: '*',
            element: <AccessDenied />,
          },
        ]
      }
    ],
  },
  {
    path: '/admin',
    element: <ProtectedRoute allowedRole="admin" />,
    errorElement: <AccessDenied />,
    children: [
      {
        path: '',
        element: <AdminLayout />,
        children: [
          {
            index: true,
            element: <Navigate to="dashboard" replace />,
          },
          {
            path: 'dashboard',
            element: <AdminDashboard />,
          },
          {
            path: 'users',
            element: <UsersManagement />,
          },
          {
            path: 'reports',
            element: <SystemReports />,
          },
          {
            path: 'roles',
            element: <RolesManagement />,
          },
          {
            path: 'attendance',
            element: <Attendance />,
          },
          {
            path: 'attendance-list',
            element: <AdminAttendanceList />,
          },
          {
            path: 'activity-log',
            element: <AdminActivityLog />,
          },
          {
            path: 'settings',
            element: <SettingsLayout />,
          },
          // Cross-module views — admin can browse any module's data
          {
            path: 'students',
            element: <AdminStudents />,
          },
          {
            path: 'teachers',
            element: <AdminTeachers />,
          },
          {
            path: 'faculties',
            element: <AdminFaculties />,
          },
          {
            path: 'profile',
            element: <UserProfile />,
          },
          {
            path: '*',
            element: <AccessDenied />,
          },
        ]
      }
    ],
  },
  {
    path: '/academia',
    element: <ProtectedRoute allowedRole="academia" />,
    errorElement: <AccessDenied />,
    children: [
      {
        path: '',
        element: <AcademiaLayout />,
        children: [
          {
            index: true,
            element: <Navigate to="dashboard" replace />,
          },
          {
            path: 'dashboard',
            element: <AcademiaDashboard />,
          },
          {
            path: 'faculties',
            element: <Faculties />,
          },
          {
            path: 'departments',
            element: <Departments />,
          },
          {
            path: 'courses',
            element: <Courses />,
          },
          {
            path: 'classes',
            element: <Classes />,
          },
          {
            path: 'academic-structure',
            element: <AcademicStructure />,
          },
          {
            path: 'reports',
            element: <AcademiaReports />,
          },
          {
            path: 'profile',
            element: <UserProfile />,
          },
          {
            path: '*',
            element: <AccessDenied />,
          },
        ]
      }
    ],
  },
  {
    path: '/',
    // Root redirects to login. Login handles route delegation based on role.
    element: <Navigate to="/login" replace />,
  },
  {
    path: '/hr',
    element: <ProtectedRoute allowedRole="hr" />,
    errorElement: <AccessDenied />,
    children: [
      {
        path: '',
        element: <HRLayout />,
        children: [
          {
            index: true,
            element: <Navigate to="dashboard" replace />,
          },
          {
            path: 'dashboard',
            element: <HRDashboard />,
          },
          {
            path: 'teachers',
            element: <HRTeachers />,
          },
          {
            path: 'reports',
            element: <HRReports />,
          },
          {
            path: 'settings',
            element: <HRSettings />,
          },
          {
            path: 'profile',
            element: <UserProfile />,
          },
          {
            path: '*',
            element: <AccessDenied />,
          },
        ]
      }
    ],
  },
  {
    path: '/admission',
    element: <ProtectedRoute allowedRole="admission" />,
    errorElement: <AccessDenied />,
    children: [
      {
        path: '',
        element: <AdmissionLayout />,
        children: [
          {
            index: true,
            element: <Navigate to="dashboard" replace />,
          },
          {
            path: 'dashboard',
            element: <AdmissionDashboard />,
          },
          {
            path: 'students',
            element: <AdmissionStudents />,
          },
          {
            path: 'approval',
            element: <AdmissionStudentsApproval />,
          },
          {
            path: 'face-registration',
            element: <AdmissionFaceRegistration />,
          },
          {
            path: 'profile',
            element: <UserProfile />,
          },
          {
            path: '*',
            element: <AccessDenied />,
          },
        ]
      }
    ],
  },
  {
    path: '/teacher',
    element: <ProtectedRoute allowedRole="teacher" />,
    errorElement: <AccessDenied />,
    children: [
      {
        path: '',
        element: <TeacherLayout />,
        children: [
          {
            index: true,
            element: <Navigate to="dashboard" replace />,
          },
          {
            path: 'dashboard',
            element: <TeacherDashboard />,
          },
          {
            path: 'attendance',
            element: <TeacherAttendance />,
          },
          {
            path: 'attendance-list',
            element: <TeacherAttendanceList />,
          },
          {
            path: 'schedule',
            element: <TeacherSchedule />,
          },
          {
            path: 'profile',
            element: <TeacherProfile />,
          },
          {
            path: 'courses',
            element: <TeacherCourses />,
          },
          {
            path: '*',
            element: <AccessDenied />,
          },
        ]
      }
    ],
  },
  {
    path: '/student',
    element: <ProtectedRoute allowedRole="student" />,
    errorElement: <AccessDenied />,
    children: [
      {
        path: '',
        element: <StudentLayout />,
        children: [
          {
            index: true,
            element: <Navigate to="dashboard" replace />,
          },
          {
            path: 'dashboard',
            element: <StudentDashboard />,
          },
          {
            path: 'attendance',
            element: <StudentAttendance />,
          },
          {
            path: 'schedule',
            element: <StudentSchedule />,
          },
          {
            path: 'profile',
            element: <StudentProfile />,
          },
          {
            path: '*',
            element: <AccessDenied />,
          },
        ]
      }
    ],
  },
  // Global catch-all: any unrecognised URL shows the Access Denied screen
  {
    path: '*',
    element: <AccessDenied />,
  },
]);
