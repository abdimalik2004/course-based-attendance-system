import { Navigate, Outlet, useNavigate } from "react-router-dom";

import { useAuthStore } from "@/store/useAuthStore";
import { ShieldX } from "lucide-react";

type Role =
  | "SUPER_ADMIN"
  | "ACADEMIA"
  | "TEACHER"
  | "HR"
  | "ADMISSIONS"
  | "FACULTY"
  | "STUDENT"
  | null;

interface ProtectedRouteProps {
  allowedRole?: Role;
}

/** Map a role string to its home dashboard path. */
function homePath(role: string): string {
  switch (role.toUpperCase()) {
    case "SUPER_ADMIN":
    case "ADMIN":
      return "/admin/dashboard";
    case "ACADEMIA":
      return "/academia/dashboard";
    case "FACULTY":
      return "/faculty/dashboard";
    case "TEACHER":
      return "/teacher/dashboard";
    case "HR":
      return "/hr/dashboard";
    case "ADMISSIONS":
      return "/admission/dashboard";
    case "STUDENT":
      return "/student/dashboard";
    default:
      return "/login";
  }
}

/** Friendly label for each role. */
function roleLabel(role: string): string {
  switch (role.toUpperCase()) {
    case "SUPER_ADMIN": return "Super Administrator";
    case "ADMIN": return "Administrator";
    case "ACADEMIA": return "Academia";
    case "FACULTY": return "Faculty";
    case "TEACHER": return "Teacher";
    case "HR": return "HR";
    case "ADMISSIONS": return "Admissions";
    case "STUDENT": return "Student";
    default: return role;
  }
}

export function AccessDenied() {
  const { user } = useAuthStore();
  const navigate = useNavigate();
  const role = (user?.role ?? "").toString().toUpperCase();
  const isAdmin = role === "SUPER_ADMIN" || role === "ADMIN";

  // Admin hit an unknown URL — silently redirect to their dashboard
  if (isAdmin) {
    return <Navigate to="/admin/dashboard" replace />;
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50 dark:bg-dark-bg px-4">
      <div className="max-w-md w-full text-center space-y-6">
        {/* Icon */}
        <div className="flex justify-center">
          <div className="w-20 h-20 rounded-full bg-rose-100 dark:bg-rose-500/10 flex items-center justify-center">
            <ShieldX className="w-10 h-10 text-rose-500" />
          </div>
        </div>

        {/* Heading */}
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            Access Denied
          </h1>
          <p className="mt-2 text-gray-500 dark:text-gray-400">
            You don't have permission to view this page.
          </p>
        </div>

        {/* Role chip */}
        {role && (
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-gray-100 dark:bg-white/5 border border-gray-200 dark:border-white/10 text-sm text-gray-700 dark:text-gray-300">
            <span className="w-2 h-2 rounded-full bg-rose-400 shrink-0" />
            Your role: <span className="font-semibold">{roleLabel(role)}</span>
          </div>
        )}

        <p className="text-sm text-gray-400 dark:text-gray-500">
          This section is restricted to a different role. If you believe this is
          a mistake, please contact your system administrator.
        </p>

        {/* Back button */}
        <button
          onClick={() => navigate(homePath(role), { replace: true })}
          className="inline-flex items-center gap-2 px-6 py-2.5 rounded-xl bg-primary text-white text-sm font-medium hover:bg-primary/90 transition-colors shadow-sm"
        >
          Go to My Dashboard
        </button>
      </div>
    </div>
  );
}

export function ProtectedRoute({ allowedRole }: ProtectedRouteProps) {
  const { isAuthenticated, user } = useAuthStore();

  if (!isAuthenticated || !user) return <Navigate to="/login" replace />;

  const role = (user.role || "").toString().toUpperCase();

  // Admins have unrestricted access to every section of the system
  const isAdmin = role === "SUPER_ADMIN" || role === "ADMIN";

  if (!isAdmin && allowedRole) {
    const expected = allowedRole.toString().toUpperCase();
    const matches =
      role === expected || role.includes(expected) || expected.includes(role);
    if (!matches) {
      return <AccessDenied />;
    }
  }

  return <Outlet />;
}
