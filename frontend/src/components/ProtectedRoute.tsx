import { Navigate, Outlet } from "react-router-dom";
import { useAuthStore } from "@/store/useAuthStore";

type Role =
  | "SUPER_ADMIN"
  | "ACADEMIA"
  | "TEACHER"
  | "HR"
  | "ADMISSIONS"
  | "FACULTY"
  | "FACULTY_ADMIN"
  | "STUDENT"
  | null;

interface ProtectedRouteProps {
  allowedRole?: Role;
}

function RouteLoader() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50 dark:bg-dark-bg">
      <div className="h-10 w-10 animate-spin rounded-full border-4 border-primary border-t-transparent" />
    </div>
  );
}

export function ProtectedRoute({ allowedRole }: ProtectedRouteProps) {
  const { isAuthenticated, user } = useAuthStore();

  // App-level auth initialization already runs before routes are rendered.
  // Avoid blocking indefinitely on hydration flags inside the route guard.
  if (!isAuthenticated || !user) return <Navigate to="/login" replace />;

  if (allowedRole) {
    const role = (user.role || "").toString().toUpperCase();
    const expected = allowedRole.toString().toUpperCase();
    // Accept broader role matches (e.g. `SUPER_ADMIN` should satisfy `admin` routes)
    const matches =
      role === expected || role.includes(expected) || expected.includes(role);
    if (!matches) {
      // Redirect users to their appropriate dashboards if they attempt to access unauthorized routes
      switch (role) {
        case "SUPER_ADMIN":
        case "ADMIN":
          return <Navigate to="/admin/dashboard" replace />;
        case "ACADEMIA":
          return <Navigate to="/academia/dashboard" replace />;
        case "FACULTY":
        case "FACULTY_ADMIN":
          return <Navigate to="/faculty/dashboard" replace />;
        case "TEACHER":
          return <Navigate to="/teacher/dashboard" replace />;
        case "STUDENT":
          return <Navigate to="/student/dashboard" replace />;
        default:
          return <Navigate to="/login" replace />;
      }
    }
  }

  return <Outlet />;
}
