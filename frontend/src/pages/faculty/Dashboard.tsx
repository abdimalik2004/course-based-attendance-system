import { useEffect, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import {
  Users,
  Building2,
  BookOpen,
  GraduationCap,
  Calendar,
  ClipboardList,
  ChevronRight,
} from "lucide-react";
import { StatCard } from "@/components/ui/StatCard";
import { useFacultyStore } from "@/store/useFacultyStore";

export default function FacultyDashboard() {
  const navigate = useNavigate();
  const {
    stats,
    fetchData,
    isLoading,
    error,
    courses,
    assignments,
    schedules,
  } = useFacultyStore();

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const avgClassSize = useMemo(
    () =>
      stats.totalClasses > 0
        ? Math.round(stats.totalStudents / stats.totalClasses)
        : 0,
    [stats.totalClasses, stats.totalStudents],
  );

  const liveSnapshot = useMemo(
    () => [
      { label: "Loaded Courses", value: courses.length },
      { label: "Teacher Assignments", value: assignments.length },
      { label: "Active Schedules", value: schedules.length },
      { label: "Average Class Size", value: avgClassSize },
    ],
    [avgClassSize, assignments.length, courses.length, schedules.length],
  );

  if (isLoading) {
    return (
      <div className="space-y-6 max-w-7xl mx-auto">
        <div className="h-8 w-64 rounded-full bg-gray-200 dark:bg-white/10 animate-pulse" />
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
          {Array.from({ length: 5 }).map((_, index) => (
            <div
              key={index}
              className="h-28 rounded-2xl bg-gray-200/80 dark:bg-white/10 animate-pulse"
            />
          ))}
        </div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-8">
          <div className="h-72 rounded-2xl bg-gray-200/80 dark:bg-white/10 animate-pulse" />
          <div className="h-72 rounded-2xl bg-gray-200/80 dark:bg-white/10 animate-pulse" />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-gray-100 flex items-center gap-3">
            <Building2 className="text-primary" size={28} />
            Faculty Dashboard
          </h1>
          <p className="text-gray-500 dark:text-gray-400 mt-1">
            Overview of your department's academic activities and resources.
          </p>
        </div>
      </div>

      {error ? (
        <div className="rounded-2xl border border-rose-200 dark:border-rose-500/20 bg-rose-50 dark:bg-rose-500/10 p-4 text-sm text-rose-700 dark:text-rose-200">
          {error}
        </div>
      ) : null}

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        <StatCard
          title="Total Students"
          value={stats.totalStudents}
          icon={Users}
          iconColor="primary"
          onClick={() => navigate("/faculty/students")}
        />
        <StatCard
          title="Total Teachers"
          value={stats.totalTeachers}
          icon={GraduationCap}
          iconColor="success"
          onClick={() => navigate("/faculty/teachers")}
        />
        <StatCard
          title="Departments"
          value={stats.totalDepartments}
          icon={Building2}
          iconColor="warning"
          onClick={() => navigate("/faculty/departments")}
        />
        <StatCard
          title="Classes"
          value={stats.totalClasses}
          icon={Calendar}
          iconColor="success"
          onClick={() => navigate("/faculty/classes")}
        />
        <StatCard
          title="Courses"
          value={stats.totalCourses}
          icon={BookOpen}
          iconColor="danger"
          onClick={() => navigate("/faculty/courses")}
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-8">
        <div className="glass-panel p-6 rounded-2xl border border-white/10 shadow-lg relative overflow-hidden group">
          <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-purple-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
          <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
            <div className="w-2 h-6 bg-blue-500 rounded-full"></div>
            Live Faculty Snapshot
          </h2>
          <div className="space-y-4">
            {liveSnapshot.map((item) => (
              <div
                key={item.label}
                className="flex items-center justify-between p-3 rounded-lg bg-white/5 border border-white/10"
              >
                <span className="text-gray-400">{item.label}</span>
                <span className="text-lg font-semibold text-white">
                  {item.value}
                </span>
              </div>
            ))}
            <div className="flex items-center justify-between p-3 rounded-lg bg-white/5 border border-white/10">
              <span className="text-gray-400">Current Faculty Scope</span>
              <span className="text-lg font-semibold text-emerald-400">
                Database driven
              </span>
            </div>
          </div>
        </div>

        <div className="glass-panel p-6 rounded-2xl border border-white/10 shadow-lg relative overflow-hidden group">
          <div className="absolute inset-0 bg-gradient-to-bl from-blue-500/5 to-purple-500/5 opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
          <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
            <div className="w-2 h-6 bg-purple-500 rounded-full"></div>
            Quick Actions
          </h2>
          <div className="space-y-3">
            {[
              {
                icon: Users,
                label: "Assign Teacher",
                description: "Assign teachers to faculty courses",
                path: "/faculty/assign-teacher",
                color: "text-blue-400",
                bg: "bg-blue-500/10 border-blue-500/20",
              },
              {
                icon: Calendar,
                label: "Schedule Course",
                description: "Set up course schedules and times",
                path: "/faculty/schedule",
                color: "text-purple-400",
                bg: "bg-purple-500/10 border-purple-500/20",
              },
              {
                icon: ClipboardList,
                label: "Attendance List",
                description: "View and manage attendance records",
                path: "/faculty/attendance-list",
                color: "text-emerald-400",
                bg: "bg-emerald-500/10 border-emerald-500/20",
              },
            ].map((action) => (
              <button
                key={action.path}
                onClick={() => navigate(action.path)}
                className="w-full flex items-center gap-3 p-3 rounded-lg bg-white/5 border border-white/10 hover:bg-white/10 hover:border-white/20 transition-all duration-200 text-left group/btn"
              >
                <div className={`flex items-center justify-center w-9 h-9 rounded-lg border ${action.bg} shrink-0`}>
                  <action.icon size={18} className={action.color} />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-semibold text-gray-900 dark:text-white">
                    {action.label}
                  </p>
                  <p className="text-xs text-gray-400 truncate">{action.description}</p>
                </div>
                <ChevronRight
                  size={16}
                  className="text-gray-500 group-hover/btn:text-gray-300 transition-colors shrink-0"
                />
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
