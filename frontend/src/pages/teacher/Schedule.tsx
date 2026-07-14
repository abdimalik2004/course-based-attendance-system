import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  Search,
  Filter,
  Calendar as CalendarIcon,
  Play,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { useLocation, useNavigate } from "react-router-dom";
import attendanceService from "@/services/attendanceService";
import teacherService from "@/services/teacherService";
import { useTeacherId } from "@/store/useTeacherStore";
import { WD_TO_DAY, formatTime } from "@/utils/scheduleUtils";

export default function TeacherSchedule() {
  const location = useLocation();
  const navigate = useNavigate();
  const initialFilter = (location.state as any)?.filter ?? "All";
  const [searchTerm, setSearchTerm] = useState("");
  const [filterDate, setFilterDate] = useState<string>(initialFilter);

  const { teacherId, isUnlinked: isTeacherUnlinked } = useTeacherId();

  // weekOffset: 0 = current week, -1 = last week, +1 = next week …
  const [weekOffset, setWeekOffset] = useState(0);

  // Poll active sessions every 10 s so "Ongoing Now" appears/disappears
  // as soon as a teacher or admin starts/ends a session.
  const activeSessionsQuery = useQuery({
    queryKey: ["teacherScheduleActiveSessions"],
    queryFn: () => attendanceService.listActiveSessions(),
    refetchInterval: 10_000,
    staleTime: 5_000,
  });

  const coursesQuery = useQuery({
    queryKey: ["teacherCourses", teacherId],
    queryFn: () => teacherService.getAssignedCourses(teacherId),
    enabled: !!teacherId,
    retry: false,
  });

  // Derive the teacher's course IDs so we can fetch schedules per-course.
  // Doing this rather than pulling the entire catalogue avoids paginated
  // truncation and removes the wrong-layer facultyService dependency.
  const courseIds = useMemo(() => {
    const assignments: any[] = coursesQuery.data?.items ?? coursesQuery.data ?? [];
    return assignments
      .map((a: any) => Number(a.course_id))
      .filter((id) => Number.isFinite(id) && id > 0);
  }, [coursesQuery.data]);

  // Fetch schedules for each assigned course in parallel, annotating
  // each result with its course_id so the memo below can match them.
  const schedulesQuery = useQuery({
    queryKey: ["teacherSchedules", teacherId, courseIds],
    queryFn: () => teacherService.getSchedulesForCourses(courseIds),
    enabled: !!teacherId && courseIds.length > 0,
    staleTime: 60_000,
  });

  // Computed week bounds based on offset (Sat–Fri grid)
  const { displayWeekStart, displayWeekEnd, displayWeekLabel } = useMemo(() => {
    const now = new Date();
    const day = now.getDay();
    const daysFromSat = day === 6 ? 0 : day + 1;
    const ws = new Date(now);
    ws.setDate(now.getDate() - daysFromSat + weekOffset * 7);
    ws.setHours(0, 0, 0, 0);
    const we = new Date(ws);
    we.setDate(ws.getDate() + 6);
    we.setHours(23, 59, 59, 999);

    const fmt = (d: Date) =>
      d.toLocaleDateString([], { month: "short", day: "numeric" });
    const label =
      weekOffset === 0
        ? "This Week"
        : weekOffset === -1
          ? "Last Week"
          : weekOffset === 1
            ? "Next Week"
            : `${fmt(ws)} – ${fmt(we)}`;

    return { displayWeekStart: ws, displayWeekEnd: we, displayWeekLabel: label };
  }, [weekOffset]);

  const scheduleData = useMemo(() => {
    // schedulesQuery now returns only this teacher's course schedules (fetched
    // per-course via Promise.all) — no client-side course filter is needed.
    const teacherSchedules: any[] = schedulesQuery.data ?? [];
    const assignments: any[] =
      coursesQuery.data?.items ?? coursesQuery.data ?? [];

    // Build a set of course IDs that currently have an active session.
    const activeSessions: any[] =
      activeSessionsQuery.data?.items ?? activeSessionsQuery.data ?? [];
    const activeSessionCourseIds = new Set(
      activeSessions.map((s: any) => String(s.course_id)),
    );

    // Build a map: course_id → course title
    const courseNames = new Map<string, string>();
    assignments.forEach((a: any) => {
      if (a.course_id) {
        courseNames.set(
          String(a.course_id),
          a.course_title ?? `Course ${a.course_id}`,
        );
      }
    });

    if (assignments.length === 0) return [];

    // Use the display week bounds (shifted by weekOffset)
    const weekStart = displayWeekStart;
    const now = new Date();

    const rows: any[] = [];
    teacherSchedules.forEach((schedule: any) => {
      // getSchedulesForCourse returns weekday_raw (lowercase array like
      // ["sat", "sun"]) and weekday (display string). Prefer weekday_raw for
      // day-of-week math; fall back to legacy array or summary formats.
      const weekdays: string[] = Array.isArray(schedule.weekday_raw)
        ? schedule.weekday_raw
        : Array.isArray(schedule.weekday)
          ? schedule.weekday
          : schedule.weekday_summary
            ? schedule.weekday_summary.split(",")
            : [];

      weekdays.forEach((wdRaw: string) => {
        const wd = wdRaw.trim().toLowerCase();
        const dayNum = WD_TO_DAY[wd];
        if (dayNum === undefined) return;

        // offset from Saturday
        const offset = (dayNum + 1) % 7;
        const occurrence = new Date(weekStart);
        occurrence.setDate(weekStart.getDate() + offset);
        occurrence.setHours(0, 0, 0, 0);

        const startTimeStr = schedule.start_time ?? "TBA";
        const endTimeStr = schedule.end_time ?? "TBA";

        // Determine if "Ongoing Now"
        let startDateTime = new Date(occurrence);
        if (startTimeStr && startTimeStr !== "TBA") {
          const [hh, mm] = startTimeStr.split(":").map((p: string) => Number(p));
          if (!isNaN(hh)) startDateTime.setHours(hh, mm || 0, 0, 0);
        }
        let endDateTime = new Date(occurrence);
        if (endTimeStr && endTimeStr !== "TBA") {
          const [hh, mm] = endTimeStr.split(":").map((p: string) => Number(p));
          if (!isNaN(hh)) endDateTime.setHours(hh, mm || 0, 0, 0);
        }

        // Only flag statuses on today's occurrence — active sessions always run
        // on the current date, and "Up Next" only makes sense for today.
        const isToday = occurrence.toDateString() === now.toDateString();

        // "Ongoing Now" only when a teacher/admin has actually started the session
        // AND this specific occurrence row is for today.
        const isCurrent =
          isToday && activeSessionCourseIds.has(String(schedule.course_id));

        // "Up Next" only for upcoming sessions on today — not other days of the week.
        const isNext = isToday && !isCurrent && startDateTime > now;

        rows.push({
          id: `${schedule.id}-${wd}`,
          scheduleId: schedule.id,
          course_id: schedule.course_id,
          course:
            courseNames.get(String(schedule.course_id)) ??
            `Course ${schedule.course_id}`,
          class_section: wd.charAt(0).toUpperCase() + wd.slice(1),
          date: occurrence.toLocaleDateString(),
          occurrenceDate: occurrence,
          start: startTimeStr,
          end: endTimeStr,
          grace: schedule.grace_period_minutes ?? 0,
          isCurrent,
          isNext,
          isToday,
        });
      });
    });

    // Sort by occurrence date then start time
    rows.sort((a, b) => {
      const dateDiff = a.occurrenceDate.getTime() - b.occurrenceDate.getTime();
      if (dateDiff !== 0) return dateDiff;
      return (a.start ?? "").localeCompare(b.start ?? "");
    });

    return rows;
  }, [schedulesQuery.data, coursesQuery.data, activeSessionsQuery.data, displayWeekStart]);

  // Date filter helpers — these always reference the real today regardless of weekOffset
  const todayDate = useMemo(() => {
    const d = new Date();
    d.setHours(0, 0, 0, 0);
    return d;
  }, []);
  const tomorrowDate = useMemo(() => {
    const d = new Date(todayDate);
    d.setDate(d.getDate() + 1);
    return d;
  }, [todayDate]);

  const filteredSchedule = useMemo(() => {
    return scheduleData.filter((item) => {
      const matchesSearch =
        item.course.toLowerCase().includes(searchTerm.toLowerCase()) ||
        item.class_section.toLowerCase().includes(searchTerm.toLowerCase());

      let matchesDate = true;
      if (filterDate === "Today") {
        matchesDate =
          item.occurrenceDate.toDateString() === todayDate.toDateString();
      } else if (filterDate === "Tomorrow") {
        matchesDate =
          item.occurrenceDate.toDateString() === tomorrowDate.toDateString();
      }

      return matchesSearch && matchesDate;
    });
  }, [scheduleData, searchTerm, filterDate, todayDate, tomorrowDate]);

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            My Schedule
          </h2>
          <p className="text-gray-500 dark:text-gray-400 text-sm mt-1">
            Your weekly class schedule based on assigned courses.
          </p>
        </div>

        {/* Week navigation */}
        <div className="flex items-center gap-1 bg-gray-100 dark:bg-white/5 rounded-xl p-1 border border-gray-200 dark:border-white/10 shrink-0">
          <button
            type="button"
            onClick={() => setWeekOffset((o) => o - 1)}
            className="p-1.5 rounded-lg text-gray-500 hover:bg-white dark:hover:bg-white/10 hover:text-gray-900 dark:hover:text-white transition-colors"
            title="Previous week"
          >
            <ChevronLeft size={16} />
          </button>
          <span className="px-3 text-sm font-medium text-gray-700 dark:text-gray-200 min-w-[100px] text-center">
            {displayWeekLabel}
          </span>
          <button
            type="button"
            onClick={() => setWeekOffset((o) => o + 1)}
            className="p-1.5 rounded-lg text-gray-500 hover:bg-white dark:hover:bg-white/10 hover:text-gray-900 dark:hover:text-white transition-colors"
            title="Next week"
          >
            <ChevronRight size={16} />
          </button>
          {weekOffset !== 0 && (
            <button
              type="button"
              onClick={() => setWeekOffset(0)}
              className="ml-1 px-2 py-1 text-xs font-medium rounded-lg bg-primary/10 text-primary hover:bg-primary/20 transition-colors"
            >
              Today
            </button>
          )}
        </div>
      </div>

      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div className="flex-1" />
        <div className="flex items-center gap-3 w-full sm:w-auto">
          <div className="relative flex-1 sm:flex-none">
            <Search
              className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400"
              size={18}
            />
            <input
              type="text"
              placeholder="Search course..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="w-full sm:w-64 pl-10 pr-4 py-2 rounded-xl glass-input text-sm text-gray-900 dark:text-gray-100 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-primary/50"
            />
          </div>
          <div className="relative">
            <select
              value={filterDate}
              onChange={(e) => setFilterDate(e.target.value)}
              className="appearance-none pl-10 pr-8 py-2 rounded-xl glass-input text-sm text-gray-700 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-primary/50 bg-transparent cursor-pointer"
              style={{
                backgroundImage: `url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%236B7280%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpolyline%20points%3D%226%209%2012%2015%2018%209%22%3E%3C%2Fpolyline%3E%3C%2Fsvg%3E")`,
                backgroundRepeat: "no-repeat",
                backgroundPosition: "right 0.5rem center",
                backgroundSize: "1em 1em",
              }}
            >
              <option value="All" className="bg-white dark:bg-dark-bg text-gray-900 dark:text-white">
                All (This Week)
              </option>
              <option value="Today" className="bg-white dark:bg-dark-bg text-gray-900 dark:text-white">
                Today
              </option>
              <option value="Tomorrow" className="bg-white dark:bg-dark-bg text-gray-900 dark:text-white">
                Tomorrow
              </option>
            </select>
            <Filter
              className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none"
              size={16}
            />
          </div>
        </div>
      </div>

      {isTeacherUnlinked && (
        <div className="rounded-2xl border border-amber-200 dark:border-amber-500/20 bg-amber-50 dark:bg-amber-500/10 p-4 text-sm text-amber-800 dark:text-amber-200">
          Your account is not yet linked to a teacher profile. Contact HR to link your login account before your schedule will appear here.
        </div>
      )}

      {(schedulesQuery.isError || coursesQuery.isError) && (
        <div className="rounded-2xl border border-rose-200 dark:border-rose-500/20 bg-rose-50 dark:bg-rose-500/10 p-4 text-sm text-rose-700 dark:text-rose-200">
          Failed to load schedule data. Please try refreshing.
        </div>
      )}

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="glass-card rounded-2xl overflow-hidden shadow-xl"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="bg-gray-50/80 dark:bg-white/5 text-xs uppercase text-gray-600 dark:text-gray-300 border-b border-gray-200 dark:border-white/10">
              <tr>
                <th className="px-6 py-4 font-semibold">Course Name</th>
                <th className="px-6 py-4 font-semibold">Day</th>
                <th className="px-6 py-4 font-semibold">Date</th>
                <th className="px-6 py-4 font-semibold">Start Time</th>
                <th className="px-6 py-4 font-semibold">End Time</th>
                <th className="px-6 py-4 font-semibold">Grace Period</th>
                <th className="px-6 py-4 font-semibold">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 dark:divide-white/10">
              {schedulesQuery.isLoading || coursesQuery.isLoading ? (
                Array.from({ length: 5 }).map((_, i) => (
                  <tr key={i} className="animate-pulse border-b border-gray-100 dark:border-white/5">
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-3">
                        <div className="h-8 w-8 rounded-lg bg-gray-200 dark:bg-white/10 shrink-0" />
                        <div className="h-4 bg-gray-200 dark:bg-white/10 rounded w-40" />
                      </div>
                    </td>
                    <td className="px-6 py-4"><div className="h-4 bg-gray-200 dark:bg-white/10 rounded w-16" /></td>
                    <td className="px-6 py-4"><div className="h-4 bg-gray-200 dark:bg-white/10 rounded w-20" /></td>
                    <td className="px-6 py-4"><div className="h-4 bg-gray-200 dark:bg-white/10 rounded w-16" /></td>
                    <td className="px-6 py-4"><div className="h-4 bg-gray-200 dark:bg-white/10 rounded w-16" /></td>
                    <td className="px-6 py-4"><div className="h-5 bg-gray-200 dark:bg-white/10 rounded-full w-14" /></td>
                    <td className="px-6 py-4"><div className="h-8 bg-gray-200 dark:bg-white/10 rounded-xl w-20" /></td>
                  </tr>
                ))
              ) : filteredSchedule.length === 0 ? (
                <tr>
                  <td
                    colSpan={7}
                    className="px-6 py-10 text-center text-sm text-gray-500 dark:text-gray-400"
                  >
                    {scheduleData.length === 0
                      ? "No schedule found. Make sure your courses have schedules assigned by faculty."
                      : "No results for the selected filter."}
                  </td>
                </tr>
              ) : (
                filteredSchedule.map((item) => (
                  <tr
                    key={item.id}
                    className={`hover:bg-gray-50 dark:hover:bg-white/5 transition-colors group
                      ${item.isCurrent ? "bg-primary/5 dark:bg-primary/10 relative" : ""}
                      ${item.isNext && !item.isCurrent ? "bg-blue-50/50 dark:bg-blue-500/5" : ""}
                    `}
                  >
                    <td className="px-6 py-4 relative">
                      {item.isCurrent && (
                        <div className="absolute left-0 top-0 bottom-0 w-1 bg-primary rounded-r-full" />
                      )}
                      <div className="flex items-center gap-3">
                        <div
                          className={`p-2 rounded-lg ${item.isCurrent ? "bg-primary/20 text-primary" : "bg-gray-100 dark:bg-white/10 text-gray-500 dark:text-gray-400 group-hover:text-primary transition-colors"}`}
                        >
                          <CalendarIcon size={16} />
                        </div>
                        <div>
                          <span
                            className={`font-medium block ${item.isCurrent ? "text-primary dark:text-primary-accent" : "text-gray-900 dark:text-white"}`}
                          >
                            {item.course}
                          </span>
                          {(item.isCurrent || item.isNext) && (
                            <span
                              className={`text-[10px] uppercase font-bold tracking-wider px-1.5 py-0.5 rounded-full mt-1 inline-block ${
                                item.isCurrent
                                  ? "bg-primary/20 text-primary dark:text-primary-accent"
                                  : "bg-blue-100 text-blue-600 dark:bg-blue-500/20 dark:text-blue-400"
                              }`}
                            >
                              {item.isCurrent ? "Ongoing Now" : "Up Next"}
                            </span>
                          )}
                        </div>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-gray-600 dark:text-gray-300">
                      {item.class_section}
                    </td>
                    <td className="px-6 py-4 text-gray-600 dark:text-gray-300">
                      {item.date}
                    </td>
                    <td className="px-6 py-4 font-medium text-gray-900 dark:text-gray-100">
                      {formatTime(item.start)}
                    </td>
                    <td className="px-6 py-4 text-gray-600 dark:text-gray-300">
                      {formatTime(item.end)}
                    </td>
                    <td className="px-6 py-4">
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800 dark:bg-white/10 dark:text-gray-300">
                        {item.grace} mins
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      {item.isCurrent ? (
                        <button
                          type="button"
                          onClick={() => navigate("/teacher/attendance")}
                          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold bg-primary/10 text-primary hover:bg-primary/20 dark:bg-primary/20 dark:text-primary-accent dark:hover:bg-primary/30 transition-colors"
                        >
                          <Play size={12} className="fill-current" />
                          Resume
                        </button>
                      ) : item.isToday ? (
                        <button
                          type="button"
                          onClick={() =>
                            navigate("/teacher/attendance", {
                              state: { course_id: item.course_id },
                            })
                          }
                          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-semibold bg-emerald-50 text-emerald-700 hover:bg-emerald-100 dark:bg-emerald-500/10 dark:text-emerald-400 dark:hover:bg-emerald-500/20 transition-colors"
                        >
                          <Play size={12} className="fill-current" />
                          Start
                        </button>
                      ) : null}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination info */}
        <div className="px-6 py-4 border-t border-gray-200 dark:border-white/10 flex items-center justify-between flex-wrap gap-2">
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Showing{" "}
            <span className="font-medium text-gray-900 dark:text-white">
              {filteredSchedule.length}
            </span>{" "}
            of{" "}
            <span className="font-medium text-gray-900 dark:text-white">
              {scheduleData.length}
            </span>{" "}
            classes
            {" · "}
            <span className="text-gray-400 dark:text-gray-500">{displayWeekLabel}</span>
          </p>
          <div className="flex items-center gap-1">
            <button
              type="button"
              onClick={() => setWeekOffset((o) => o - 1)}
              className="p-1 rounded-lg text-gray-400 hover:text-gray-700 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-white/10 transition-colors"
            >
              <ChevronLeft size={16} />
            </button>
            {weekOffset !== 0 && (
              <button
                type="button"
                onClick={() => setWeekOffset(0)}
                className="px-2 py-0.5 text-xs rounded-lg text-primary hover:bg-primary/10 transition-colors"
              >
                Current
              </button>
            )}
            <button
              type="button"
              onClick={() => setWeekOffset((o) => o + 1)}
              className="p-1 rounded-lg text-gray-400 hover:text-gray-700 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-white/10 transition-colors"
            >
              <ChevronRight size={16} />
            </button>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
