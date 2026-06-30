import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import {
  Search,
  Filter,
  Calendar as CalendarIcon,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { useLocation } from "react-router-dom";
import facultyService from "@/services/facultyService";
import courseService from "@/services/courseService";
import attendanceService from "@/services/attendanceService";
import { useAuthStore } from "@/store/useAuthStore";

// Weekday code → JS getDay() (0=Sun … 6=Sat)
const WD_TO_DAY: Record<string, number> = {
  sat: 6, sun: 0, mon: 1, tue: 2, wed: 3, thu: 4, fri: 5,
  saturday: 6, sunday: 0, monday: 1, tuesday: 2, wednesday: 3, thursday: 4, friday: 5,
};

function formatTime(timeStr: string): string {
  if (!timeStr || timeStr === "TBA") return "TBA";
  const [hh, mm] = timeStr.split(":").map(Number);
  if (isNaN(hh)) return timeStr;
  const period = hh >= 12 ? "PM" : "AM";
  const h = hh % 12 || 12;
  return `${h}:${String(mm ?? 0).padStart(2, "0")} ${period}`;
}

export default function TeacherSchedule() {
  const location = useLocation();
  const initialFilter = (location.state as any)?.filter ?? "All";
  const [searchTerm, setSearchTerm] = useState("");
  const [filterDate, setFilterDate] = useState<string>(initialFilter);

  const { user } = useAuthStore();
  const teacherId = Number(user?.teacherId ?? user?.id ?? 0);

  const schedulesQuery = useQuery({
    queryKey: ["teacherSchedules", teacherId],
    queryFn: () => facultyService.listSchedules(),
    enabled: !!teacherId,
  });

  // Poll active sessions every 10 s so "Ongoing Now" appears/disappears
  // as soon as a teacher or admin starts/ends a session.
  const activeSessionsQuery = useQuery({
    queryKey: ["teacherScheduleActiveSessions"],
    queryFn: () => attendanceService.listActiveSessions(),
    refetchInterval: 10_000,
    staleTime: 5_000,
  });

  const coursesQuery = useQuery({
    queryKey: ["teacherScheduleCourses", teacherId],
    queryFn: () =>
      courseService.listAssignments({ teacher_id: teacherId, skip: 0, limit: 200 }),
    enabled: !!teacherId,
    retry: false,
  });

  const scheduleData = useMemo(() => {
    const scheduleResponse: any[] =
      schedulesQuery.data?.items ?? schedulesQuery.data ?? [];
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

    // Teacher's assigned course IDs
    const teacherCourseIds = new Set(assignments.map((a: any) => String(a.course_id)));

    if (teacherCourseIds.size === 0) return [];

    // Filter schedules to only those belonging to teacher's courses
    const teacherSchedules = scheduleResponse.filter((s: any) =>
      teacherCourseIds.has(String(s.course_id)),
    );

    // Week bounds (Sat–Fri)
    const now = new Date();
    const day = now.getDay();
    const daysFromSat = day === 6 ? 0 : day + 1;
    const weekStart = new Date(now);
    weekStart.setDate(now.getDate() - daysFromSat);
    weekStart.setHours(0, 0, 0, 0);

    const rows: any[] = [];
    teacherSchedules.forEach((schedule: any) => {
      const weekdays: string[] = Array.isArray(schedule.weekday)
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
  }, [schedulesQuery.data, coursesQuery.data, activeSessionsQuery.data]);

  // Date filter helpers
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
      } else if (filterDate === "This Week") {
        // already showing this week's rows by default, so same as All in weekly view
        matchesDate = true;
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
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 dark:divide-white/10">
              {schedulesQuery.isLoading || coursesQuery.isLoading ? (
                <tr>
                  <td
                    colSpan={6}
                    className="px-6 py-10 text-center text-sm text-gray-500 dark:text-gray-400"
                  >
                    Loading schedule…
                  </td>
                </tr>
              ) : filteredSchedule.length === 0 ? (
                <tr>
                  <td
                    colSpan={6}
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
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination info */}
        <div className="px-6 py-4 border-t border-gray-200 dark:border-white/10 flex items-center justify-between">
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Showing{" "}
            <span className="font-medium text-gray-900 dark:text-white">
              {filteredSchedule.length}
            </span>{" "}
            of{" "}
            <span className="font-medium text-gray-900 dark:text-white">
              {scheduleData.length}
            </span>{" "}
            classes this week
          </p>
          <div className="flex items-center gap-2">
            <button
              className="p-2 rounded-lg border border-gray-200 dark:border-white/10 text-gray-500 hover:bg-gray-50 dark:hover:bg-white/5 disabled:opacity-50 transition-colors"
              disabled
            >
              <ChevronLeft size={16} />
            </button>
            <button className="p-2 rounded-lg border border-gray-200 dark:border-white/10 text-gray-500 hover:bg-gray-50 dark:hover:bg-white/5 disabled:opacity-50 transition-colors" disabled>
              <ChevronRight size={16} />
            </button>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
