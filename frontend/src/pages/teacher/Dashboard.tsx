import { motion } from "framer-motion";
import {
  BookOpen,
  Users,
  Calendar,
  Clock,
  Play,
  List,
  ChevronRight,
  CheckCircle2,
} from "lucide-react";
import {
  PieChart,
  Pie,
  Cell,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { useNavigate } from "react-router-dom";
import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import attendanceService from "@/services/attendanceService";
import facultyService from "@/services/facultyService";
import courseService from "@/services/courseService";
import { useAuthStore } from "@/store/useAuthStore";

// Weekday code → JS getDay() (0=Sun … 6=Sat)
const WD_TO_DAY: Record<string, number> = {
  sat: 6, sun: 0, mon: 1, tue: 2, wed: 3, thu: 4, fri: 5,
  saturday: 6, sunday: 0, monday: 1, tuesday: 2, wednesday: 3, thursday: 4, friday: 5,
};

function formatTimeAgo(date: Date): string {
  const diff = Date.now() - date.getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "Just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

function formatTime(timeStr: string): string {
  if (!timeStr || timeStr === "TBA") return "TBA";
  const [hh, mm] = timeStr.split(":").map(Number);
  if (isNaN(hh)) return timeStr;
  const period = hh >= 12 ? "PM" : "AM";
  const h = hh % 12 || 12;
  return `${h}:${String(mm ?? 0).padStart(2, "0")} ${period}`;
}

export default function TeacherDashboard() {
  const navigate = useNavigate();
  const { user } = useAuthStore();
  const teacherId = Number(user?.teacherId ?? user?.id ?? 0);

  const sessionsQuery = useQuery({
    queryKey: ["teacherSessions"],
    queryFn: () => attendanceService.listSessions({ skip: 0, limit: 200 }),
    retry: false,
  });

  const schedulesQuery = useQuery({
    queryKey: ["teacherSchedules", teacherId],
    queryFn: () => facultyService.listSchedules(),
    enabled: !!teacherId,
    retry: false,
  });

  const assignmentsQuery = useQuery({
    queryKey: ["teacherCoursesCount", teacherId],
    queryFn: () =>
      courseService.listAssignments({ teacher_id: teacherId, skip: 0, limit: 200 }),
    enabled: !!teacherId,
    retry: false,
  });

  const sessions: any[] = useMemo(
    () => sessionsQuery.data ?? [],
    [sessionsQuery.data],
  );
  const allSchedules: any[] = useMemo(
    () => schedulesQuery.data?.items ?? schedulesQuery.data ?? [],
    [schedulesQuery.data],
  );
  const assignments: any[] = useMemo(
    () => assignmentsQuery.data?.items ?? assignmentsQuery.data ?? [],
    [assignmentsQuery.data],
  );

  // Map of course_id → course title
  const courseNamesMap = useMemo(() => {
    const map = new Map<string, string>();
    assignments.forEach((a: any) => {
      if (a.course_id) {
        map.set(String(a.course_id), a.course_title ?? `Course ${a.course_id}`);
      }
    });
    return map;
  }, [assignments]);

  // Set of course IDs assigned to this teacher
  const teacherCourseIds = useMemo(
    () => new Set(assignments.map((a: any) => Number(a.course_id))),
    [assignments],
  );

  // Schedules filtered to teacher's assigned courses
  const teacherSchedules = useMemo(
    () => allSchedules.filter((s: any) => teacherCourseIds.has(Number(s.course_id))),
    [allSchedules, teacherCourseIds],
  );

  // ── Week bounds (Sat–Fri) ──────────────────────────────────────────────
  const { weekStart, weekEnd, today } = useMemo(() => {
    const now = new Date();
    now.setHours(0, 0, 0, 0);
    const day = now.getDay(); // 0=Sun … 6=Sat
    const daysFromSat = day === 6 ? 0 : day + 1;
    const wStart = new Date(now);
    wStart.setDate(now.getDate() - daysFromSat);
    const wEnd = new Date(wStart);
    wEnd.setDate(wStart.getDate() + 6);
    wEnd.setHours(23, 59, 59, 999);
    return { weekStart: wStart, weekEnd: wEnd, today: now };
  }, []);

  // ── Today's schedules ─────────────────────────────────────────────────
  const todayScheduleRows = useMemo(() => {
    const todayDay = today.getDay();
    return teacherSchedules
      .filter((s: any) => {
        const weekdays: string[] = Array.isArray(s.weekday) ? s.weekday : [];
        return weekdays.some((w: string) => WD_TO_DAY[w.toLowerCase().trim()] === todayDay);
      })
      .map((s: any) => {
        const courseTitle = courseNamesMap.get(String(s.course_id)) ?? `Course ${s.course_id}`;
        const todayStr = today.toDateString();
        const todaySession = sessions.find(
          (sess: any) =>
            Number(sess.course_id) === Number(s.course_id) &&
            sess.start_time &&
            new Date(sess.start_time).toDateString() === todayStr,
        );
        const status = todaySession
          ? String(todaySession.status ?? "").toUpperCase()
          : "SCHEDULED";
        return {
          id: s.id,
          time: `${formatTime(s.start_time)} – ${formatTime(s.end_time)}`,
          course: courseTitle,
          class_section: Array.isArray(s.weekday) ? s.weekday.map((w: string) => w.toUpperCase()).join(" / ") : "",
          status,
        };
      });
  }, [teacherSchedules, courseNamesMap, sessions, today]);

  // ── Stats calculations ────────────────────────────────────────────────
  const statsData = useMemo(() => {
    const myCoursesCount = teacherCourseIds.size;
    const todaysCount = todayScheduleRows.length;

    // Attendance sessions: CLOSED (not ENDED — SessionStatus enum uses CLOSED)
    const closedSessions = sessions.filter(
      (s: any) => String(s.status ?? "").toUpperCase() === "CLOSED",
    ).length;

    // Build a lookup: "courseId-dateStr" → true for every session that has
    // already been started (ACTIVE or CLOSED) this week.
    const startedSessionKeys = new Set<string>();
    sessions.forEach((s: any) => {
      if (!s.start_time) return;
      const sd = new Date(s.start_time);
      if (sd >= weekStart && sd <= weekEnd) {
        startedSessionKeys.add(`${s.course_id}-${sd.toDateString()}`);
      }
    });

    // Upcoming classes = scheduled slots from today through end-of-week
    // where no session has been started yet for that course on that date.
    let upcomingCount = 0;
    teacherSchedules.forEach((schedule: any) => {
      const weekdays: string[] = Array.isArray(schedule.weekday) ? schedule.weekday : [];
      weekdays.forEach((wd: string) => {
        const dayNum = WD_TO_DAY[wd.toLowerCase().trim()];
        if (dayNum === undefined) return;
        // offset from Saturday (week index 0)
        const offset = (dayNum + 1) % 7;
        const occurrence = new Date(weekStart);
        occurrence.setDate(weekStart.getDate() + offset);
        occurrence.setHours(0, 0, 0, 0);
        // Only count from today onward (past days are done regardless)
        if (occurrence < today || occurrence > weekEnd) return;
        const key = `${schedule.course_id}-${occurrence.toDateString()}`;
        if (!startedSessionKeys.has(key)) {
          upcomingCount++;
        }
      });
    });

    return [
      {
        title: "Today's Classes",
        value: String(todaysCount),
        subtitle: todaysCount === 1 ? "1 class today" : `${todaysCount} classes today`,
        icon: Clock,
        color: "text-blue-500",
        bg: "bg-blue-500/10",
        onClick: () => navigate("/teacher/schedule", { state: { filter: "Today" } }),
      },
      {
        title: "My Courses",
        value: String(myCoursesCount),
        subtitle: "Assigned courses",
        icon: BookOpen,
        color: "text-purple-500",
        bg: "bg-purple-500/10",
        onClick: () => navigate("/teacher/attendance-list"),
      },
      {
        title: "Attendance Sessions",
        value: String(closedSessions),
        subtitle: "Completed sessions",
        icon: Users,
        color: "text-green-500",
        bg: "bg-green-500/10",
        onClick: () => navigate("/teacher/attendance-list"),
      },
      {
        title: "Upcoming Classes",
        value: String(upcomingCount),
        subtitle: "Remaining this week",
        icon: Calendar,
        color: "text-orange-500",
        bg: "bg-orange-500/10",
        onClick: () => navigate("/teacher/schedule"),
      },
    ];
  }, [sessions, teacherCourseIds, todayScheduleRows, teacherSchedules, weekStart, weekEnd, today, navigate]);

  // ── Charts ────────────────────────────────────────────────────────────
  const attendancePieData = useMemo(() => {
    const active = sessions.filter(
      (s: any) => String(s.status ?? "").toUpperCase() === "ACTIVE",
    ).length;
    const closed = sessions.filter(
      (s: any) => String(s.status ?? "").toUpperCase() === "CLOSED",
    ).length;
    return [
      { name: "Active", value: active, color: "#10B981" },
      { name: "Closed", value: closed, color: "#6366F1" },
    ].filter((e) => e.value > 0);
  }, [sessions]);

  const weeklyLineData = useMemo(() => {
    // Count sessions per day of the current week (Sat–Fri)
    const days = ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"];
    const counts: Record<string, number> = {};
    days.forEach((d) => (counts[d] = 0));

    const dayIndexToLabel = (d: number) => {
      // JS getDay() → week day label in our Sat-Fri order
      const map: Record<number, string> = { 6: "Sat", 0: "Sun", 1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri" };
      return map[d] ?? "Sat";
    };

    sessions.forEach((s: any) => {
      if (!s.start_time) return;
      const d = new Date(s.start_time);
      if (d >= weekStart && d <= weekEnd) {
        const label = dayIndexToLabel(d.getDay());
        counts[label] = (counts[label] ?? 0) + 1;
      }
    });

    return days.map((day) => ({ day, sessions: counts[day] ?? 0 }));
  }, [sessions, weekStart, weekEnd]);

  // ── Recent Activity ───────────────────────────────────────────────────
  const recentActivity = useMemo(() => {
    return sessions.slice(0, 5).map((session: any) => {
      const courseTitle =
        courseNamesMap.get(String(session.course_id)) ??
        session.course_name ??
        `Course ${session.course_id}`;
      const status = String(session.status ?? "").toUpperCase();
      const startTime = session.start_time ? new Date(session.start_time) : null;
      const timeAgo = startTime ? formatTimeAgo(startTime) : "Recent";
      const timeLabel = startTime
        ? startTime.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
        : "—";
      const isActive = status === "ACTIVE";
      const isClosed = status === "CLOSED";
      return {
        id: session.id,
        title: isActive ? "Live Session" : isClosed ? "Session Closed" : "Session Started",
        desc: `${courseTitle} • ${timeLabel}`,
        time: timeAgo,
        icon: isActive ? Play : isClosed ? CheckCircle2 : BookOpen,
        color: isActive ? "text-green-500" : isClosed ? "text-blue-500" : "text-purple-500",
      };
    });
  }, [sessions, courseNamesMap]);

  const hasError =
    sessionsQuery.isError || schedulesQuery.isError || assignmentsQuery.isError;
  const errorMsg =
    (sessionsQuery.error as Error)?.message ||
    (schedulesQuery.error as Error)?.message ||
    (assignmentsQuery.error as Error)?.message;

  const isLoading =
    sessionsQuery.isLoading || schedulesQuery.isLoading || assignmentsQuery.isLoading;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
          Dashboard Overview
        </h2>
      </div>

      {hasError && (
        <div className="rounded-2xl border border-rose-200 dark:border-rose-500/20 bg-rose-50 dark:bg-rose-500/10 p-4 text-sm text-rose-700 dark:text-rose-200">
          {errorMsg ?? "Failed to load dashboard data."}
        </div>
      )}

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6">
        {statsData.map((stat, idx) => (
          <motion.div
            key={idx}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: idx * 0.1 }}
            onClick={stat.onClick}
            className="glass-card p-6 rounded-2xl hover:shadow-lg hover:shadow-primary/5 transition-all duration-300 group cursor-pointer"
          >
            <div className="flex items-start justify-between">
              <div>
                <p className="text-sm font-medium text-gray-500 dark:text-gray-400">
                  {stat.title}
                </p>
                <h3 className="text-3xl font-bold text-gray-900 dark:text-white mt-2 group-hover:text-primary transition-colors">
                  {isLoading ? "—" : stat.value}
                </h3>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  {stat.subtitle}
                </p>
              </div>
              <div className={`p-3 rounded-xl ${stat.bg} ${stat.color}`}>
                <stat.icon size={24} />
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Content Area (Left 2 columns) */}
        <div className="lg:col-span-2 space-y-6">
          {/* Charts Row */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Pie Chart — Overall Attendance */}
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.4 }}
              className="glass-card p-6 rounded-2xl"
            >
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Overall Attendance
              </h3>
              <div className="h-[220px] w-full">
                {attendancePieData.length === 0 ? (
                  <div className="h-full flex items-center justify-center text-sm text-gray-400">
                    No session data yet
                  </div>
                ) : (
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={attendancePieData}
                        cx="50%"
                        cy="50%"
                        innerRadius={55}
                        outerRadius={75}
                        paddingAngle={5}
                        dataKey="value"
                      >
                        {attendancePieData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                      </Pie>
                      <Tooltip
                        contentStyle={{
                          backgroundColor: "rgba(17, 24, 39, 0.9)",
                          border: "none",
                          borderRadius: "12px",
                          color: "#fff",
                        }}
                        itemStyle={{ color: "#fff" }}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                )}
              </div>
              <div className="flex justify-center gap-4 mt-2">
                {attendancePieData.map((item, idx) => (
                  <div key={idx} className="flex items-center gap-2">
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: item.color }}
                    />
                    <span className="text-xs text-gray-500 dark:text-gray-400">
                      {item.name} ({item.value})
                    </span>
                  </div>
                ))}
              </div>
            </motion.div>

            {/* Line Chart — Weekly Trend */}
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.4, delay: 0.1 }}
              className="glass-card p-6 rounded-2xl"
            >
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
                Weekly Trend
              </h3>
              <div className="h-[220px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={weeklyLineData}>
                    <CartesianGrid
                      strokeDasharray="3 3"
                      stroke="rgba(255,255,255,0.1)"
                      vertical={false}
                    />
                    <XAxis
                      dataKey="day"
                      stroke="#6B7280"
                      fontSize={12}
                      tickLine={false}
                      axisLine={false}
                    />
                    <YAxis
                      stroke="#6B7280"
                      fontSize={12}
                      tickLine={false}
                      axisLine={false}
                      allowDecimals={false}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "rgba(17, 24, 39, 0.9)",
                        border: "none",
                        borderRadius: "12px",
                        color: "#fff",
                      }}
                    />
                    <Line
                      type="monotone"
                      dataKey="sessions"
                      name="Sessions"
                      stroke="#3B82F6"
                      strokeWidth={3}
                      dot={{ r: 4, fill: "#3B82F6", strokeWidth: 2, stroke: "#fff" }}
                      activeDot={{ r: 6 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </motion.div>
          </div>

          {/* Today's Schedule Table */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, delay: 0.2 }}
            className="glass-card rounded-2xl overflow-hidden"
          >
            <div className="p-6 border-b border-gray-200 dark:border-white/10 flex justify-between items-center">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                Today's Schedule
              </h3>
              <button
                onClick={() => navigate("/teacher/schedule", { state: { filter: "Today" } })}
                className="text-sm text-primary hover:underline"
              >
                View All
              </button>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm text-gray-500 dark:text-gray-400">
                <thead className="bg-gray-50 dark:bg-white/5 text-xs uppercase text-gray-700 dark:text-gray-300">
                  <tr>
                    <th className="px-6 py-4 font-medium">Time</th>
                    <th className="px-6 py-4 font-medium">Course</th>
                    <th className="px-6 py-4 font-medium">Day</th>
                    <th className="px-6 py-4 font-medium">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200 dark:divide-white/10">
                  {isLoading ? (
                    <tr>
                      <td colSpan={4} className="px-6 py-10 text-center text-sm text-gray-500 dark:text-gray-400">
                        Loading schedule…
                      </td>
                    </tr>
                  ) : todayScheduleRows.length === 0 ? (
                    <tr>
                      <td colSpan={4} className="px-6 py-10 text-center text-sm text-gray-500 dark:text-gray-400">
                        No classes scheduled for today.
                      </td>
                    </tr>
                  ) : (
                    todayScheduleRows.map((item) => (
                      <tr
                        key={item.id}
                        className="hover:bg-gray-50 dark:hover:bg-white/5 transition-colors"
                      >
                        <td className="px-6 py-4 whitespace-nowrap font-medium text-gray-900 dark:text-white">
                          {item.time}
                        </td>
                        <td className="px-6 py-4 font-medium text-gray-900 dark:text-white">
                          {item.course}
                        </td>
                        <td className="px-6 py-4 text-gray-500 dark:text-gray-400">
                          {item.class_section}
                        </td>
                        <td className="px-6 py-4">
                          <span
                            className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                              item.status === "CLOSED"
                                ? "bg-green-100 text-green-800 dark:bg-green-500/20 dark:text-green-400"
                                : item.status === "ACTIVE"
                                  ? "bg-blue-100 text-blue-800 dark:bg-blue-500/20 dark:text-blue-400 animate-pulse"
                                  : "bg-gray-100 text-gray-600 dark:bg-white/10 dark:text-gray-400"
                            }`}
                          >
                            {item.status === "CLOSED" ? "Done" : item.status === "ACTIVE" ? "Live" : "Scheduled"}
                          </span>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </motion.div>
        </div>

        {/* Right Panel */}
        <div className="space-y-6">
          {/* Quick Actions */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.4 }}
            className="glass-card p-6 rounded-2xl space-y-4"
          >
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
              Quick Actions
            </h3>

            <button
              onClick={() => navigate("/teacher/attendance")}
              className="w-full group flex items-center justify-between p-4 rounded-xl bg-gradient-brand text-white hover:shadow-lg hover:shadow-primary/20 transition-all duration-300"
            >
              <div className="flex items-center gap-4">
                <div className="p-2 bg-white/20 rounded-lg backdrop-blur-sm group-hover:scale-110 transition-transform">
                  <Play size={20} className="fill-white" />
                </div>
                <div className="text-left">
                  <p className="font-semibold">Start Attendance</p>
                  <p className="text-xs text-white/80">Begin a new session</p>
                </div>
              </div>
              <ChevronRight
                size={20}
                className="text-white/70 group-hover:text-white group-hover:translate-x-1 transition-all"
              />
            </button>

            <button
              onClick={() => navigate("/teacher/schedule")}
              className="w-full group flex items-center justify-between p-4 rounded-xl bg-gray-50 hover:bg-gray-100 dark:bg-white/5 dark:hover:bg-white/10 border border-gray-200 dark:border-white/10 transition-all duration-300"
            >
              <div className="flex items-center gap-4">
                <div className="p-2 bg-gray-200 dark:bg-white/10 rounded-lg text-gray-600 dark:text-gray-300 group-hover:scale-110 transition-transform">
                  <Calendar size={20} />
                </div>
                <div className="text-left">
                  <p className="font-semibold text-gray-900 dark:text-white">
                    View Schedule
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    Check upcoming classes
                  </p>
                </div>
              </div>
              <ChevronRight
                size={20}
                className="text-gray-400 group-hover:text-gray-900 dark:group-hover:text-white group-hover:translate-x-1 transition-all"
              />
            </button>

            <button
              onClick={() => navigate("/teacher/attendance-list")}
              className="w-full group flex items-center justify-between p-4 rounded-xl bg-gray-50 hover:bg-gray-100 dark:bg-white/5 dark:hover:bg-white/10 border border-gray-200 dark:border-white/10 transition-all duration-300"
            >
              <div className="flex items-center gap-4">
                <div className="p-2 bg-gray-200 dark:bg-white/10 rounded-lg text-gray-600 dark:text-gray-300 group-hover:scale-110 transition-transform">
                  <List size={20} />
                </div>
                <div className="text-left">
                  <p className="font-semibold text-gray-900 dark:text-white">
                    Attendance Records
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    View past sessions
                  </p>
                </div>
              </div>
              <ChevronRight
                size={20}
                className="text-gray-400 group-hover:text-gray-900 dark:group-hover:text-white group-hover:translate-x-1 transition-all"
              />
            </button>
          </motion.div>

          {/* Recent Activity */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.4, delay: 0.1 }}
            className="glass-card p-6 rounded-2xl"
          >
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">
              Recent Activity
            </h3>
            {recentActivity.length === 0 ? (
              <p className="text-sm text-gray-500 dark:text-gray-400">
                No recent session activity.
              </p>
            ) : (
              <div className="space-y-6">
                {recentActivity.map((activity, idx) => (
                  <div key={activity.id} className="relative pl-6">
                    {idx !== recentActivity.length - 1 && (
                      <div className="absolute left-2.5 top-8 bottom-[-24px] w-[1px] bg-gray-200 dark:bg-white/10" />
                    )}
                    <div
                      className={`absolute left-0 top-1 w-5 h-5 rounded-full ${activity.color} bg-white dark:bg-dark-card flex items-center justify-center border-2 border-current shadow-sm`}
                    >
                      <activity.icon size={10} className="text-current" />
                    </div>
                    <div>
                      <h4 className="text-sm font-semibold text-gray-900 dark:text-white">
                        {activity.title}
                      </h4>
                      <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                        {activity.desc}
                      </p>
                      <p className="text-[10px] text-gray-400 dark:text-gray-500 mt-2">
                        {activity.time}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </motion.div>
        </div>
      </div>
    </div>
  );
}
