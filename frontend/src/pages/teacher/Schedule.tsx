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
import attendanceService from "@/services/attendanceService";
import courseService from "@/services/courseService";

export default function TeacherSchedule() {
  const [searchTerm, setSearchTerm] = useState("");
  const [filterDate, setFilterDate] = useState("All");
  const schedulesQuery = useQuery({
    queryKey: ["teacherSchedules"],
    queryFn: () => attendanceService.listSessions({ skip: 0, limit: 200 }),
  });
  const coursesQuery = useQuery({
    queryKey: ["teacherScheduleCourses"],
    queryFn: () => courseService.listCourses({ skip: 0, limit: 200 }),
  });

  const scheduleData = useMemo(() => {
    const scheduleResponse =
      schedulesQuery.data?.items ?? schedulesQuery.data ?? [];
    const courses = coursesQuery.data?.items ?? coursesQuery.data ?? [];
    const courseNames = new Map(
      courses.map((course: any) => [
        String(course.id),
        course.title ?? course.name ?? `Course ${course.id}`,
      ]),
    );

    return scheduleResponse.map((schedule: any, index: number) => {
      const weekdays = Array.isArray(schedule.weekday)
        ? schedule.weekday.join(", ")
        : (schedule.weekday_summary ?? "Scheduled");
      return {
        id: schedule.id ?? index,
        course:
          courseNames.get(String(schedule.course_id)) ??
          `Course ${schedule.course_id}`,
        class_section: weekdays,
        start: schedule.start_time ?? "TBA",
        end: schedule.end_time ?? "TBA",
        grace: schedule.grace_period_minutes ?? 0,
        date: weekdays,
        isNext: schedule.weekday_count === 1,
        isCurrent: schedule.weekday_count > 1,
      };
    });
  }, [coursesQuery.data, schedulesQuery.data]);

  const filteredSchedule = scheduleData.filter((item) => {
    const matchesSearch =
      item.course.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.class_section.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesDate = filterDate === "All" || item.date === filterDate;
    return matchesSearch && matchesDate;
  });

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            My Schedule
          </h2>
          <p className="text-gray-500 dark:text-gray-400 text-sm mt-1">
            View and manage your upcoming classes and lab sessions.
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
              className="appearance-none pl-10 pr-8 py-2 rounded-xl glass-input text-sm text-gray-700 dark:text-gray-200 focus:outline-none focus:ring-2 focus:ring-primary/50 bg-transparent cursor-pointer bg-no-repeat bg-[right_0.5rem_center] bg-[length:1em_1em]"
              style={{
                backgroundImage: `url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%236B7280%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpolyline%20points%3D%226%209%2012%2015%2018%209%22%3E%3C%2Fpolyline%3E%3C%2Fsvg%3E")`,
              }}
            >
              <option
                value="All"
                className="bg-white dark:bg-dark-bg text-gray-900 dark:text-white"
              >
                All Dates
              </option>
              <option
                value="Today"
                className="bg-white dark:bg-dark-bg text-gray-900 dark:text-white"
              >
                Today
              </option>
              <option
                value="Tomorrow"
                className="bg-white dark:bg-dark-bg text-gray-900 dark:text-white"
              >
                Tomorrow
              </option>
              <option
                value="Next Week"
                className="bg-white dark:bg-dark-bg text-gray-900 dark:text-white"
              >
                Next Week
              </option>
            </select>
            <Filter
              className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none"
              size={16}
            />
          </div>
        </div>
      </div>

      {schedulesQuery.error || coursesQuery.error ? (
        <div className="rounded-2xl border border-rose-200 dark:border-rose-500/20 bg-rose-50 dark:bg-rose-500/10 p-4 text-sm text-rose-700 dark:text-rose-200">
          Failed to load live schedule data.
        </div>
      ) : null}

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
                <th className="px-6 py-4 font-semibold">Class / Section</th>
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
                    Loading schedule...
                  </td>
                </tr>
              ) : filteredSchedule.length === 0 ? (
                <tr>
                  <td
                    colSpan={6}
                    className="px-6 py-10 text-center text-sm text-gray-500 dark:text-gray-400"
                  >
                    No live schedule rows found in the database.
                  </td>
                </tr>
              ) : (
                filteredSchedule.map((item) => (
                  <tr
                    key={item.id}
                    className={`
                    hover:bg-gray-50 dark:hover:bg-white/5 transition-colors group
                    ${item.isCurrent ? "bg-primary/5 dark:bg-primary/10 relative" : ""}
                    ${item.isNext ? "bg-blue-50/50 dark:bg-blue-500/5" : ""}
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
                      {item.start}
                    </td>
                    <td className="px-6 py-4 text-gray-600 dark:text-gray-300">
                      {item.end}
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

        {/* Pagination */}
        <div className="px-6 py-4 border-t border-gray-200 dark:border-white/10 flex items-center justify-between">
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Showing{" "}
            <span className="font-medium text-gray-900 dark:text-white">1</span>{" "}
            to{" "}
            <span className="font-medium text-gray-900 dark:text-white">
              {filteredSchedule.length}
            </span>{" "}
            of{" "}
            <span className="font-medium text-gray-900 dark:text-white">
              {scheduleData.length}
            </span>{" "}
            results
          </p>
          <div className="flex items-center gap-2">
            <button
              className="p-2 rounded-lg border border-gray-200 dark:border-white/10 text-gray-500 hover:bg-gray-50 dark:hover:bg-white/5 disabled:opacity-50 transition-colors"
              disabled
            >
              <ChevronLeft size={16} />
            </button>
            <button className="p-2 rounded-lg border border-gray-200 dark:border-white/10 text-gray-500 hover:bg-gray-50 dark:hover:bg-white/5 transition-colors">
              <ChevronRight size={16} />
            </button>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
