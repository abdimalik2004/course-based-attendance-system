/**
 * Teacher Courses page — combines:
 *  #31: Course roster / student list (GET /courses/{id}/students + /reports/course/{id}/students)
 *  #32: Per-session summary / session history (GET /reports/course/{id}/sessions)
 *  #35: Attendance rate statistics / charts (GET /reports/course/{id})
 */
import { useState, useMemo } from "react";
import { motion } from "framer-motion";
import {
  BookOpen,
  Users,
  BarChart2,
  List,
  ChevronDown,
  TrendingUp,
  TrendingDown,
  CheckCircle2,
  XCircle,
  Clock,
  AlertTriangle,
} from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import { useQuery } from "@tanstack/react-query";
import teacherService from "@/services/teacherService";
import { useTeacherId } from "@/store/useTeacherStore";

type Tab = "stats" | "roster" | "sessions";

const TABS: { id: Tab; label: string; icon: React.ElementType }[] = [
  { id: "stats",    label: "Stats",          icon: BarChart2     },
  { id: "roster",   label: "Student Roster", icon: Users         },
  { id: "sessions", label: "Session History",icon: List          },
];

function pct(num: number, total: number) {
  if (!total) return 0;
  return Math.round((num / total) * 100);
}

export default function TeacherCourses() {
  const { teacherId } = useTeacherId();
  const [selectedCourseId, setSelectedCourseId] = useState<number | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>("stats");

  // ── fetch assigned courses ───────────────────────────────────────────────
  const coursesQuery = useQuery({
    queryKey: ["teacherCourses", teacherId],
    queryFn: () => teacherService.getAssignedCourses(teacherId),
    enabled: !!teacherId,
    staleTime: 60_000,
  });

  const courses = useMemo(() => {
    const list: any[] = coursesQuery.data?.items ?? coursesQuery.data ?? [];
    return list.map((a: any) => ({
      id: Number(a.course_id),
      title: a.course_title ?? `Course ${a.course_id}`,
      code: a.course_code ?? "",
    }));
  }, [coursesQuery.data]);

  // Auto-select first course
  const courseId = selectedCourseId ?? courses[0]?.id ?? null;
  const selectedCourse = courses.find((c) => c.id === courseId);

  // ── course-level stats ───────────────────────────────────────────────────
  const statsQuery = useQuery({
    queryKey: ["teacherCourseStats", courseId],
    queryFn: () => teacherService.getCourseStats(courseId!),
    enabled: !!courseId && activeTab === "stats",
    staleTime: 60_000,
  });

  // ── per-student stats ────────────────────────────────────────────────────
  const studentStatsQuery = useQuery({
    queryKey: ["teacherCourseStudentStats", courseId],
    queryFn: () => teacherService.getCourseStudentStats(courseId!),
    enabled: !!courseId && activeTab === "stats",
    staleTime: 60_000,
  });

  // ── enrolled roster ──────────────────────────────────────────────────────
  const rosterQuery = useQuery({
    queryKey: ["teacherCourseRoster", courseId],
    queryFn: () => teacherService.getCourseEnrolledStudents(courseId!),
    enabled: !!courseId && activeTab === "roster",
    staleTime: 60_000,
  });

  // ── session history ──────────────────────────────────────────────────────
  const sessionsQuery = useQuery({
    queryKey: ["teacherCourseSessions", courseId],
    queryFn: () => teacherService.getCourseSessions(courseId!),
    enabled: !!courseId && activeTab === "sessions",
    staleTime: 60_000,
    retry: false,
  });

  // ── chart data ────────────────────────────────────────────────────────────
  const barData = useMemo(() => {
    const s = statsQuery.data;
    if (!s) return [];
    const total = s.present + s.late + s.absent;
    return [
      { name: "Present", value: s.present, pct: pct(s.present, total), fill: "#10B981" },
      { name: "Late",    value: s.late,    pct: pct(s.late, total),    fill: "#F59E0B" },
      { name: "Absent",  value: s.absent,  pct: pct(s.absent, total),  fill: "#EF4444" },
    ];
  }, [statsQuery.data]);

  const attendanceRate = useMemo(() => {
    const s = statsQuery.data;
    if (!s) return null;
    const total = s.present + s.late + s.absent;
    return pct(s.present + s.late, total);
  }, [statsQuery.data]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">My Courses</h2>
          <p className="text-gray-500 dark:text-gray-400 text-sm mt-1">
            Roster, session history, and attendance stats per course.
          </p>
        </div>

        {/* Course selector */}
        {courses.length > 0 && (
          <div className="relative shrink-0">
            <select
              value={courseId ?? ""}
              onChange={(e) => setSelectedCourseId(Number(e.target.value))}
              className="appearance-none pl-4 pr-10 py-2.5 rounded-xl glass-input text-sm font-medium text-gray-900 dark:text-white bg-transparent cursor-pointer border border-gray-200 dark:border-white/10 focus:outline-none focus:ring-2 focus:ring-primary/50 min-w-[220px]"
            >
              {courses.map((c) => (
                <option key={c.id} value={c.id} className="bg-white dark:bg-dark-bg">
                  {c.code ? `${c.code} — ${c.title}` : c.title}
                </option>
              ))}
            </select>
            <ChevronDown size={16} className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none" />
          </div>
        )}
      </div>

      {/* Loading / no data states */}
      {coursesQuery.isLoading && (
        <div className="glass-card rounded-2xl p-10 text-center">
          <div className="animate-spin w-8 h-8 border-2 border-primary border-t-transparent rounded-full mx-auto mb-3" />
          <p className="text-sm text-gray-500 dark:text-gray-400">Loading courses…</p>
        </div>
      )}

      {!coursesQuery.isLoading && courses.length === 0 && (
        <div className="glass-card rounded-2xl p-10 text-center">
          <div className="w-12 h-12 rounded-2xl bg-gray-100 dark:bg-white/10 text-gray-400 flex items-center justify-center mx-auto mb-3">
            <BookOpen size={22} />
          </div>
          <p className="text-sm text-gray-500 dark:text-gray-400">No courses assigned yet.</p>
        </div>
      )}

      {courses.length > 0 && courseId && (
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.35 }}
          className="glass-card rounded-2xl overflow-hidden"
        >
          {/* Tab bar */}
          <div className="border-b border-gray-200 dark:border-white/10 flex">
            {TABS.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 px-5 py-4 text-sm font-medium border-b-2 transition-colors ${
                    isActive
                      ? "border-primary text-primary dark:text-primary-accent"
                      : "border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-white"
                  }`}
                >
                  <Icon size={15} />
                  {tab.label}
                </button>
              );
            })}
          </div>

          {/* ── Stats tab ── */}
          {activeTab === "stats" && (
            <div className="p-6 space-y-6">
              {statsQuery.isLoading ? (
                <div className="animate-pulse space-y-4">
                  <div className="grid grid-cols-3 gap-4">
                    {[1, 2, 3].map((i) => (
                      <div key={i} className="h-20 bg-gray-200 dark:bg-white/10 rounded-xl" />
                    ))}
                  </div>
                  <div className="h-40 bg-gray-200 dark:bg-white/10 rounded-xl" />
                </div>
              ) : statsQuery.isError ? (
                <div className="text-sm text-rose-500 p-4 bg-rose-50 dark:bg-rose-500/10 rounded-xl">
                  Failed to load course stats.
                </div>
              ) : statsQuery.data ? (
                <>
                  {/* Stat cards */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
                    <div className="bg-gray-50 dark:bg-white/5 rounded-xl p-4 text-center">
                      <p className="text-2xl font-bold text-gray-900 dark:text-white">
                        {statsQuery.data.total_records}
                      </p>
                      <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">Total Records</p>
                    </div>
                    <div className="bg-emerald-50 dark:bg-emerald-500/10 rounded-xl p-4 text-center">
                      <p className="text-2xl font-bold text-emerald-600 dark:text-emerald-400">
                        {statsQuery.data.present}
                      </p>
                      <p className="text-xs text-emerald-600/70 dark:text-emerald-400/70 mt-1">Present</p>
                    </div>
                    <div className="bg-amber-50 dark:bg-amber-500/10 rounded-xl p-4 text-center">
                      <p className="text-2xl font-bold text-amber-600 dark:text-amber-400">
                        {statsQuery.data.late}
                      </p>
                      <p className="text-xs text-amber-600/70 dark:text-amber-400/70 mt-1">Late</p>
                    </div>
                    <div className="bg-rose-50 dark:bg-rose-500/10 rounded-xl p-4 text-center">
                      <p className="text-2xl font-bold text-rose-600 dark:text-rose-400">
                        {statsQuery.data.absent}
                      </p>
                      <p className="text-xs text-rose-600/70 dark:text-rose-400/70 mt-1">Absent</p>
                    </div>
                  </div>

                  {/* Attendance rate + bar chart */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Rate indicator */}
                    <div className="bg-gray-50 dark:bg-white/5 rounded-xl p-5 flex flex-col items-center justify-center gap-3">
                      <p className="text-xs uppercase tracking-wider font-semibold text-gray-400 dark:text-gray-500">
                        Overall Attendance Rate
                      </p>
                      <div className="relative">
                        <p
                          className={`text-5xl font-bold ${
                            (attendanceRate ?? 0) >= 75
                              ? "text-emerald-500"
                              : (attendanceRate ?? 0) >= 50
                                ? "text-amber-500"
                                : "text-rose-500"
                          }`}
                        >
                          {attendanceRate ?? 0}%
                        </p>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-white/10 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full transition-all ${
                            (attendanceRate ?? 0) >= 75
                              ? "bg-emerald-500"
                              : (attendanceRate ?? 0) >= 50
                                ? "bg-amber-500"
                                : "bg-rose-500"
                          }`}
                          style={{ width: `${attendanceRate ?? 0}%` }}
                        />
                      </div>
                      <div className="flex items-center gap-1.5 text-xs text-gray-500 dark:text-gray-400">
                        {(attendanceRate ?? 0) >= 75 ? (
                          <><TrendingUp size={13} className="text-emerald-500" /> Good attendance</>
                        ) : (attendanceRate ?? 0) >= 50 ? (
                          <><AlertTriangle size={13} className="text-amber-500" /> Needs attention</>
                        ) : (
                          <><TrendingDown size={13} className="text-rose-500" /> Poor attendance</>
                        )}
                      </div>
                    </div>

                    {/* Bar chart */}
                    <div className="bg-gray-50 dark:bg-white/5 rounded-xl p-5">
                      <p className="text-xs uppercase tracking-wider font-semibold text-gray-400 dark:text-gray-500 mb-4">
                        Breakdown
                      </p>
                      <div className="h-[160px]">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={barData} barSize={36}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(107,114,128,0.2)" vertical={false} />
                            <XAxis dataKey="name" stroke="#6B7280" fontSize={12} tickLine={false} axisLine={false} />
                            <YAxis stroke="#6B7280" fontSize={12} tickLine={false} axisLine={false} allowDecimals={false} />
                            <Tooltip
                              contentStyle={{
                                backgroundColor: "rgba(17,24,39,0.9)",
                                border: "none",
                                borderRadius: "10px",
                                color: "#fff",
                              }}
                              formatter={(value: number, _name: string, entry: any) => [
                                `${value} (${entry.payload.pct}%)`,
                                entry.payload.name,
                              ]}
                            />
                            <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                              {barData.map((entry, i) => (
                                <Cell key={i} fill={entry.fill} />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  </div>

                  {/* Per-student stats table */}
                  {studentStatsQuery.data?.students && studentStatsQuery.data.students.length > 0 && (
                    <div>
                      <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-200 mb-3">
                        Per-Student Breakdown
                      </h3>
                      <div className="overflow-x-auto rounded-xl border border-gray-100 dark:border-white/5">
                        <table className="w-full text-left text-sm">
                          <thead className="bg-gray-50 dark:bg-white/5 text-xs uppercase text-gray-500 dark:text-gray-400">
                            <tr>
                              <th className="px-4 py-3 font-semibold">Student</th>
                              <th className="px-4 py-3 font-semibold text-center">Present</th>
                              <th className="px-4 py-3 font-semibold text-center">Late</th>
                              <th className="px-4 py-3 font-semibold text-center">Absent</th>
                              <th className="px-4 py-3 font-semibold text-center">Rate</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-gray-100 dark:divide-white/5">
                            {studentStatsQuery.data.students.map((stu) => {
                              const rate = pct(stu.present + stu.late, stu.total);
                              return (
                                <tr key={stu.student_id} className="hover:bg-gray-50 dark:hover:bg-white/5 transition-colors">
                                  <td className="px-4 py-3">
                                    <p className="font-medium text-gray-900 dark:text-white">{stu.student_name}</p>
                                    <p className="text-xs text-gray-400 font-mono">{stu.student_number}</p>
                                  </td>
                                  <td className="px-4 py-3 text-center text-emerald-600 dark:text-emerald-400 font-medium">{stu.present}</td>
                                  <td className="px-4 py-3 text-center text-amber-600 dark:text-amber-400 font-medium">{stu.late}</td>
                                  <td className="px-4 py-3 text-center text-rose-600 dark:text-rose-400 font-medium">{stu.absent}</td>
                                  <td className="px-4 py-3 text-center">
                                    <span
                                      className={`inline-flex px-2 py-0.5 rounded-full text-xs font-semibold ${
                                        rate >= 75
                                          ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-400"
                                          : rate >= 50
                                            ? "bg-amber-100 text-amber-700 dark:bg-amber-500/20 dark:text-amber-400"
                                            : "bg-rose-100 text-rose-700 dark:bg-rose-500/20 dark:text-rose-400"
                                      }`}
                                    >
                                      {rate}%
                                    </span>
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                </>
              ) : null}
            </div>
          )}

          {/* ── Roster tab ── */}
          {activeTab === "roster" && (
            <div className="p-6">
              {rosterQuery.isLoading ? (
                <div className="animate-pulse space-y-3">
                  {Array.from({ length: 5 }).map((_, i) => (
                    <div key={i} className="flex items-center gap-3">
                      <div className="h-8 w-8 rounded-full bg-gray-200 dark:bg-white/10 shrink-0" />
                      <div className="flex-1 space-y-1.5">
                        <div className="h-3.5 bg-gray-200 dark:bg-white/10 rounded w-36" />
                        <div className="h-2.5 bg-gray-200 dark:bg-white/10 rounded w-20" />
                      </div>
                      <div className="h-5 bg-gray-200 dark:bg-white/10 rounded-full w-16" />
                    </div>
                  ))}
                </div>
              ) : rosterQuery.isError ? (
                <div className="text-sm text-rose-500 p-4 bg-rose-50 dark:bg-rose-500/10 rounded-xl">
                  Failed to load student roster.
                </div>
              ) : (rosterQuery.data?.length ?? 0) === 0 ? (
                <div className="py-10 text-center text-sm text-gray-400 dark:text-gray-500">
                  No students enrolled in this course yet.
                </div>
              ) : (
                <div className="overflow-x-auto rounded-xl border border-gray-100 dark:border-white/5">
                  <table className="w-full text-left text-sm">
                    <thead className="bg-gray-50 dark:bg-white/5 text-xs uppercase text-gray-500 dark:text-gray-400">
                      <tr>
                        <th className="px-4 py-3 font-semibold">#</th>
                        <th className="px-4 py-3 font-semibold">Student</th>
                        <th className="px-4 py-3 font-semibold">Status</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-100 dark:divide-white/5">
                      {rosterQuery.data?.map((stu, idx) => (
                        <tr key={stu.id} className="hover:bg-gray-50 dark:hover:bg-white/5 transition-colors">
                          <td className="px-4 py-3 text-gray-400 text-xs">{idx + 1}</td>
                          <td className="px-4 py-3">
                            <div className="flex items-center gap-3">
                              <div className="w-8 h-8 rounded-full bg-primary/15 text-primary flex items-center justify-center text-sm font-semibold shrink-0">
                                {stu.full_name?.charAt(0)?.toUpperCase() ?? "S"}
                              </div>
                              <div>
                                <p className="font-medium text-gray-900 dark:text-white">{stu.full_name}</p>
                                <p className="text-xs text-gray-400 font-mono">{stu.student_number}</p>
                              </div>
                            </div>
                          </td>
                          <td className="px-4 py-3">
                            <span
                              className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${
                                stu.status === "Active"
                                  ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-400"
                                  : "bg-gray-100 text-gray-600 dark:bg-white/10 dark:text-gray-400"
                              }`}
                            >
                              {stu.status === "Active" ? (
                                <CheckCircle2 size={10} />
                              ) : (
                                <XCircle size={10} />
                              )}
                              {stu.status}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                  <div className="px-4 py-3 border-t border-gray-100 dark:border-white/5">
                    <p className="text-xs text-gray-400 dark:text-gray-500">
                      {rosterQuery.data?.length} student{rosterQuery.data?.length !== 1 ? "s" : ""} enrolled
                    </p>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ── Sessions tab ── */}
          {activeTab === "sessions" && (
            <div className="p-6">
              {sessionsQuery.isLoading ? (
                <div className="animate-pulse space-y-3">
                  {Array.from({ length: 5 }).map((_, i) => (
                    <div key={i} className="h-14 bg-gray-200 dark:bg-white/10 rounded-xl" />
                  ))}
                </div>
              ) : sessionsQuery.isError ? (
                <div className="text-sm text-rose-500 p-4 bg-rose-50 dark:bg-rose-500/10 rounded-xl">
                  Failed to load session history.
                </div>
              ) : (sessionsQuery.data?.sessions.length ?? 0) === 0 ? (
                <div className="py-10 text-center text-sm text-gray-400 dark:text-gray-500">
                  No sessions recorded for this course yet.
                </div>
              ) : (
                <div className="overflow-x-auto rounded-xl border border-gray-100 dark:border-white/5">
                  <table className="w-full text-left text-sm">
                    <thead className="bg-gray-50 dark:bg-white/5 text-xs uppercase text-gray-500 dark:text-gray-400">
                      <tr>
                        <th className="px-4 py-3 font-semibold">Session</th>
                        <th className="px-4 py-3 font-semibold">Date</th>
                        <th className="px-4 py-3 font-semibold">Time</th>
                        <th className="px-4 py-3 font-semibold">Status</th>
                        <th className="px-4 py-3 font-semibold text-center">Present</th>
                        <th className="px-4 py-3 font-semibold text-center">Late</th>
                        <th className="px-4 py-3 font-semibold text-center">Absent</th>
                        <th className="px-4 py-3 font-semibold text-center">Rate</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-100 dark:divide-white/5">
                      {sessionsQuery.data?.sessions.map((sess, idx) => {
                        const rate = pct(sess.present + sess.late, sess.total);
                        const fmtTime = (t: string | null) => {
                          if (!t) return "—";
                          const d = new Date(t);
                          return isNaN(d.getTime())
                            ? t.slice(0, 5)
                            : d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
                        };
                        const fmtDate = (d: string | null) => {
                          if (!d) return "—";
                          const dt = new Date(d);
                          return isNaN(dt.getTime())
                            ? d
                            : dt.toLocaleDateString([], { month: "short", day: "numeric", year: "numeric" });
                        };
                        const isClosed = String(sess.status).toUpperCase() === "CLOSED";
                        const isActive = String(sess.status).toUpperCase() === "ACTIVE";
                        return (
                          <tr key={sess.session_id} className="hover:bg-gray-50 dark:hover:bg-white/5 transition-colors">
                            <td className="px-4 py-3 font-mono text-xs text-gray-500 dark:text-gray-400">
                              #{idx + 1}
                            </td>
                            <td className="px-4 py-3 text-gray-700 dark:text-gray-200">
                              {fmtDate(sess.session_date ?? sess.start_time)}
                            </td>
                            <td className="px-4 py-3 text-gray-500 dark:text-gray-400 whitespace-nowrap">
                              {fmtTime(sess.start_time)} – {fmtTime(sess.end_time)}
                            </td>
                            <td className="px-4 py-3">
                              <span
                                className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium ${
                                  isClosed
                                    ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-400"
                                    : isActive
                                      ? "bg-blue-100 text-blue-700 dark:bg-blue-500/20 dark:text-blue-400 animate-pulse"
                                      : "bg-gray-100 text-gray-600 dark:bg-white/10 dark:text-gray-400"
                                }`}
                              >
                                {isClosed ? (
                                  <><CheckCircle2 size={10} /> Done</>
                                ) : isActive ? (
                                  <><Clock size={10} /> Live</>
                                ) : (
                                  sess.status
                                )}
                              </span>
                            </td>
                            <td className="px-4 py-3 text-center text-emerald-600 dark:text-emerald-400 font-medium">{sess.present}</td>
                            <td className="px-4 py-3 text-center text-amber-600 dark:text-amber-400 font-medium">{sess.late}</td>
                            <td className="px-4 py-3 text-center text-rose-600 dark:text-rose-400 font-medium">{sess.absent}</td>
                            <td className="px-4 py-3 text-center">
                              {sess.total > 0 ? (
                                <span
                                  className={`inline-flex px-2 py-0.5 rounded-full text-xs font-semibold ${
                                    rate >= 75
                                      ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-500/20 dark:text-emerald-400"
                                      : rate >= 50
                                        ? "bg-amber-100 text-amber-700 dark:bg-amber-500/20 dark:text-amber-400"
                                        : "bg-rose-100 text-rose-700 dark:bg-rose-500/20 dark:text-rose-400"
                                  }`}
                                >
                                  {rate}%
                                </span>
                              ) : (
                                <span className="text-gray-400">—</span>
                              )}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                  <div className="px-4 py-3 border-t border-gray-100 dark:border-white/5">
                    <p className="text-xs text-gray-400 dark:text-gray-500">
                      {sessionsQuery.data?.sessions.length} session{sessionsQuery.data?.sessions.length !== 1 ? "s" : ""} recorded
                    </p>
                  </div>
                </div>
              )}
            </div>
          )}
        </motion.div>
      )}
    </div>
  );
}
