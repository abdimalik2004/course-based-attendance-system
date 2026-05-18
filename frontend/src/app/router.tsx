import { lazy } from 'react';
import { createBrowserRouter, Navigate } from 'react-router-dom';
const Login = lazy(() => import('@/pages/auth/Login'));
import { ProtectedRoute } from '@/components/ProtectedRoute';

const AdminLayout = lazy(() => import('@/layouts/AdminLayout'));
const AdminDashboard = lazy(() => import('@/pages/admin/Dashboard'));
const UsersManagement = lazy(() => import('@/pages/admin/UsersManagement'));
const SystemReports = lazy(() => import('@/pages/admin/SystemReports'));
const Attendance = lazy(() => import('@/pages/admin/Attendance'));
const AdminAttendanceList = lazy(() => import('@/pages/admin/AttendanceList'));
const RolesManagement = lazy(() => import('@/pages/admin/RolesManagement'));
const SettingsLayout = lazy(() => import('@/pages/admin/settings/SettingsLayout'));

// Academia Imports
const AcademiaLayout = lazy(() => import('@/components/academia/Layout'));
const AcademiaDashboard = lazy(() => import('@/pages/academia/Dashboard'));
const Faculties = lazy(() => import('@/pages/academia/Faculties'));
const Departments = lazy(() => import('@/pages/academia/Departments'));
const Courses = lazy(() => import('@/pages/academia/Courses'));
const Classes = lazy(() => import('@/pages/academia/Classes'));
const AcademicStructure = lazy(() => import('@/pages/academia/Academic-Structure'));

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

// Teacher Imports
const TeacherLayout = lazy(() => import('@/components/teacher/Layout'));
const TeacherDashboard = lazy(() => import('@/pages/teacher/Dashboard'));
const TeacherAttendance = lazy(() => import('@/pages/teacher/Attendance'));
const TeacherAttendanceList = lazy(() => import('@/pages/teacher/AttendanceList'));
const TeacherSchedule = lazy(() => import('@/pages/teacher/Schedule'));

// Student Imports
const StudentLayout = lazy(() => import('@/components/student/Layout'));
const StudentDashboard = lazy(() => import('@/pages/student/Dashboard'));
const StudentAttendance = lazy(() => import('@/pages/student/Attendance'));
const StudentSchedule = lazy(() => import('@/pages/student/Schedule'));

export const router = createBrowserRouter([
  {
    path: '/login',
    element: <Login />,
  },
  {
    path: '/faculty',
    element: <ProtectedRoute allowedRole="faculty" />,
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
        ]
      }
    ],
  },
  {
    path: '/admin',
    element: <ProtectedRoute allowedRole="admin" />,
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
            path: 'settings',
            element: <SettingsLayout />,
          },
        ]
      }
    ],
  },
  {
    path: '/academia',
    element: <ProtectedRoute allowedRole="academia" />,
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
        ]
      }
    ],
  },
  {
    path: '/admission',
    element: <ProtectedRoute allowedRole="admission" />,
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
        ]
      }
    ],
  },
  {
    path: '/teacher',
    element: <ProtectedRoute allowedRole="teacher" />,
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
        ]
      }
    ],
  },
  {
    path: '/student',
    element: <ProtectedRoute allowedRole="student" />,
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
        ]
      }
    ],
  },
]);
