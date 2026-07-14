import { useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import {
  PercentSquare,
  CheckCircle2,
  XCircle,
  Clock,
  BadgeCheck,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { cn } from "@/utils/cn";
import { useQuery } from "@tanstack/react-query";
import dashboardService, {
  type StudentAttendanceCourse,
  type StudentScheduleItem,
} from "@/services/dashboardService";

const buildStatusBadge = (status: string) => {
  switch (status) {
    case "Good":
      return <Badge variant="success">Good</Badge>;
    case "Warning":
      return <Badge variant="warning">Warning</Badge>;
    case "Low":
      return <Badge variant="danger">Low</Badge>;
    default:
      return <Badge variant="default">{status}</Badge>;
  }
};

const getProgressColor = (percent: number) => {
  if (percent >= 85) return "bg-emerald-500";
  if (percent >= 70) return "bg-yellow-500";
  return "bg-rose-500";
};

// Map display day labels to JS getDay() values (0=Sun…6=Sat)
const DAY_TO_NUM: Record<string, number> = {
  Sun: 0, Mon: 1, Tue: 2, Wed: 3, Thu: 4, Fri: 5, Sat: 6,
};

export default function StudentDashboard() {
  const navigate = useNavigate();

  // Use the same cache keys as Attendance.tsx and Schedule.tsx so all three
  // pages share cached data — no duplicate network requests when navigating.
  const {
    data: attendance = [],
    isLoading: attendanceLoading,
    isError: attendanceError,
    refetch: refetchAttendance,
  } = useQuery<StudentAttendanceCourse[]>({
    queryKey: ["studentAttendance"],
    queryFn: () => dashboardService.studentAttendanceData(),
    refetchInterval: 60_000,
    staleTime: 1000 * 60 * 2,
  });

  const {
    data: schedule = [],
    isLoading: scheduleLoading,
    isError: scheduleError,
    refetch: refetchSchedule,
  } = useQuery<StudentScheduleItem[]>({
    queryKey: ["studentSchedule"],
    queryFn: () => dashboardService.studentScheduleData(),
    refetchInterval: 60_000,
    staleTime: 1000 * 60 * 2,
  });

  const isLoading = attendanceLoading || scheduleLoading;
  const hasError = attendanceError || scheduleError;

  // Per-course cards from attendance aggregation (typed — no any)
  const courseCards = useMemo(
    () =>
      attendance.map((course) => ({
        id: String(course.id),
        name: course.course_name,
        code: course.course_code,
        attended: course.classes_attended,
        absent: course.classes_absent,
        excused: course.classes_excused,
        total: course.total_classes,
        percent: Math.round(course.attendance_percentage),
        status: course.status,
      })),
    [attendance],
  );

  // Today's display label e.g. "Mon", "Tue" — matches backend weekday display labels
  const todayLabel = useMemo(() => {
    const days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    return days[new Date().getDay()];
  }, []);

  // Today's scheduled classes (typed — no any)
  const todayCards = useMemo(
    () =>
      schedule
        .filter((item) => item.weekdays.includes(todayLabel))
        .sort((a, b) => a.start_time.localeCompare(b.start_time))
        .map((item, index) => ({
          id: String(item.id ?? index),
          time: `${item.start_time} - ${item.end_time}`,
          course: item.course_name || item.course_code,
          room: item.class_name ?? "TBA",
          days: item.weekdays.join(", "),
          hasActiveSession: item.has_active_session,
        })),
    [schedule, todayLabel],
  );

  // Single-pass stat computation — one iteration instead of four
  const { totalAttended, totalAbsent, totalExcused, attendanceRate } = useMemo(() => {
    let attended = 0, absent = 0, excused = 0, total = 0;
    for (const c of courseCards) {
      attended += c.attended;
      absent   += c.absent;
      excused  += c.excused;
      total    += c.total;
    }
    return {
      totalAttended: attended,
      totalAbsent: absent,
      totalExcused: excused,
      attendanceRate: total > 0 ? Math.round((attended / total) * 100) : 0,
    };
  }, [courseCards]);

  // Upcoming classes count = class slots remaining from today to end of current week.
  // A slot on today counts only if its start_time hasn't passed yet.
  const upcomingCount = useMemo(() => {
    const now = new Date();
    const todayNum = now.getDay(); // 0=Sun … 6=Sat
    const todayHHMM = `${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}`;

    let count = 0;
    for (const item of schedule) {
      for (const day of item.weekdays) {
        const dayNum = DAY_TO_NUM[day] ?? -1;
        if (dayNum < todayNum) continue; // already passed this week
        if (dayNum === todayNum) {
          // Only count today's slot if start_time hasn't passed yet
          const startHHMM = (item.start_time ?? "23:59") as string;
          if (startHHMM <= todayHHMM) continue;
        }
        count++;
      }
    }
    return count;
  }, [schedule]);

  const statCards = useMemo(() => [
    {
      icon: CheckCircle2,
      iconColor: "text-emerald-500",
      iconBg: "bg-emerald-500/10",
      value: isLoading ? "—" : String(totalAttended),
      label: "Classes Attended",
      onClick: () => navigate("/student/attendance"),
    },
    {
      icon: XCircle,
      iconColor: "text-rose-500",
      iconBg: "bg-rose-500/10",
      value: isLoading ? "—" : String(totalAbsent),
      label: "Missed Classes",
      onClick: () => navigate("/student/attendance"),
    },
    {
      icon: BadgeCheck,
      iconColor: "text-amber-500",
      iconBg: "bg-amber-500/10",
      value: isLoading ? "—" : String(totalExcused),
      label: "Classes Excused",
      onClick: () => navigate("/student/attendance"),
    },
    {
      icon: Clock,
      iconColor: "text-orange-500",
      iconBg: "bg-orange-500/10",
      value: isLoading ? "—" : String(upcomingCount),
      label: "Upcoming Classes",
      onClick: () => navigate("/student/schedule"),
    },
    {
      icon: PercentSquare,
      iconColor: "text-purple-500",
      iconBg: "bg-purple-500/10",
      value: isLoading ? "—" : `${attendanceRate}%`,
      label: "Attendance Rate",
      onClick: () => navigate("/student/attendance"),
    },
  ], [isLoading, totalAttended, totalAbsent, totalExcused, upcomingCount, attendanceRate, navigate]);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white tracking-tight">
          Overview
        </h1>
        <p className="text-gray-500 dark:text-gray-400 mt-1">
          Welcome back! Here's a summary of your academic progress.
        </p>
      </div>

      {/* Error banner */}
      {hasError && (
        <div className="flex items-center justify-between rounded-xl border border-rose-200 dark:border-rose-500/20 bg-rose-50 dark:bg-rose-500/10 px-4 py-3 text-sm text-rose-700 dark:text-rose-300">
          <span>Failed to load some data. Check your connection.</span>
          <button
            onClick={() => { void refetchAttendance(); void refetchSchedule(); }}
            className="ml-4 shrink-0 text-xs font-semibold underline hover:no-underline"
          >
            Retry
          </button>
        </div>
      )}

      {/* Stats Grid — 5 cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
        {isLoading
          ? Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="glass-card border border-gray-200 dark:border-white/5 rounded-2xl p-5 flex flex-col items-start gap-4 animate-pulse">
                <div className="w-12 h-12 rounded-xl bg-gray-100 dark:bg-white/5" />
                <div className="space-y-2">
                  <div className="h-8 w-16 rounded-lg bg-gray-100 dark:bg-white/5" />
                  <div className="h-3 w-24 rounded bg-gray-100 dark:bg-white/5" />
                </div>
              </div>
            ))
          : statCards.map((card, i) => (
              <motion.div
                key={card.label}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05 }}
              >
                <Card
                  className="glass-card border-gray-200 dark:border-white/5 hover:border-primary/50 dark:hover:border-primary/50 transition-colors group cursor-pointer"
                  onClick={card.onClick}
                >
                  <CardContent className="p-5 flex flex-col items-start gap-4">
                    <div
                      className={cn("p-3 rounded-xl transition-colors", card.iconBg)}
                    >
                      <card.icon className={cn("w-6 h-6", card.iconColor)} />
                    </div>
                    <div>
                      <h3 className="text-3xl font-bold text-gray-900 dark:text-white mb-1">
                        {card.value}
                      </h3>
                      <p className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                        {card.label}
                      </p>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Course Attendance Table */}
        <div className="xl:col-span-2 space-y-6">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.3 }}
          >
            <Card className="glass-card border-gray-200 dark:border-white/5">
              <CardHeader className="border-b border-gray-100 dark:border-white/5 pb-4">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-lg font-bold text-gray-900 dark:text-white">
                    Course Attendance
                  </CardTitle>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-primary hover:text-primary-accent"
                    onClick={() => navigate("/student/attendance")}
                  >
                    View All
                  </Button>
                </div>
              </CardHeader>
              <CardContent className="p-0">
                <div className="overflow-x-auto">
                  <table className="w-full text-sm text-left">
                    <thead className="bg-gray-50/50 dark:bg-white/5 text-gray-500 dark:text-gray-400 font-medium border-b border-gray-100 dark:border-white/5">
                      <tr>
                        <th className="px-6 py-4">Course</th>
                        <th className="px-6 py-4 text-center">Attended</th>
                        <th className="px-6 py-4">Percentage</th>
                        <th className="px-6 py-4 text-center">Status</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-100 dark:divide-white/5">
                      {isLoading ? (
                        Array.from({ length: 4 }).map((_, i) => (
                          <tr key={i} className="animate-pulse">
                            <td className="px-6 py-4">
                              <div className="h-3.5 w-36 rounded bg-gray-100 dark:bg-white/5 mb-1.5" />
                              <div className="h-2.5 w-16 rounded bg-gray-100 dark:bg-white/5" />
                            </td>
                            <td className="px-6 py-4 text-center">
                              <div className="h-3.5 w-14 rounded bg-gray-100 dark:bg-white/5 mx-auto" />
                            </td>
                            <td className="px-6 py-4">
                              <div className="h-1.5 w-full rounded-full bg-gray-100 dark:bg-white/5" />
                            </td>
                            <td className="px-6 py-4 text-center">
                              <div className="h-5 w-14 rounded-full bg-gray-100 dark:bg-white/5 mx-auto" />
                            </td>
                          </tr>
                        ))
                      ) : courseCards.length === 0 ? (
                        <tr>
                          <td
                            colSpan={4}
                            className="px-6 py-10 text-center text-gray-400 dark:text-gray-500"
                          >
                            No attendance records yet.
                          </td>
                        </tr>
                      ) : (
                        courseCards.map((course) => (
                          <tr
                            key={course.id}
                            className="hover:bg-gray-50/50 dark:hover:bg-white/5 transition-colors group"
                          >
                            <td className="px-6 py-4">
                              <p className="font-medium text-gray-900 dark:text-white">
                                {course.name}
                              </p>
                              <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
                                {course.code}
                              </p>
                            </td>
                            <td className="px-6 py-4 text-center">
                              <span className="font-medium text-gray-900 dark:text-white">
                                {course.attended}
                              </span>
                              <span className="text-gray-500 dark:text-gray-400 mx-1">
                                /
                              </span>
                              <span className="text-gray-500 dark:text-gray-400">
                                {course.total}
                              </span>
                            </td>
                            <td className="px-6 py-4">
                              <div className="flex items-center gap-3">
                                <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-1.5 overflow-hidden">
                                  <motion.div
                                    initial={{ width: 0 }}
                                    animate={{ width: `${course.percent}%` }}
                                    transition={{ duration: 1, ease: "easeOut" }}
                                    className={cn(
                                      "h-full rounded-full",
                                      getProgressColor(course.percent),
                                    )}
                                  />
                                </div>
                                <span className="text-xs font-medium text-gray-600 dark:text-gray-300 w-8">
                                  {course.percent}%
                                </span>
                              </div>
                            </td>
                            <td className="px-6 py-4 text-center">
                              {buildStatusBadge(course.status)}
                            </td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </div>

        {/* Right Panel — Today's Classes */}
        <div className="space-y-6">
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
          >
            <Card className="glass-card border-gray-200 dark:border-white/5">
              <CardHeader className="border-b border-gray-100 dark:border-white/5 pb-4">
                <div className="flex items-center justify-between">
                  <CardTitle className="text-lg font-bold text-gray-900 dark:text-white">
                    Today's Classes
                  </CardTitle>
                  <span className="text-xs font-medium text-gray-400 dark:text-gray-500 bg-gray-100 dark:bg-white/5 px-2.5 py-1 rounded-full">
                    {todayLabel}
                  </span>
                </div>
              </CardHeader>
              <CardContent className="p-6">
                {isLoading ? (
                  <div className="space-y-4 animate-pulse">
                    {Array.from({ length: 3 }).map((_, i) => (
                      <div key={i} className="flex items-center gap-3">
                        <div className="w-5 h-5 rounded-full bg-gray-100 dark:bg-white/5 shrink-0" />
                        <div className="flex-1 rounded-xl bg-gray-100 dark:bg-white/5 h-[72px]" />
                      </div>
                    ))}
                  </div>
                ) : todayCards.length === 0 ? (
                  <div className="flex flex-col items-center justify-center py-8 text-center">
                    <div className="w-12 h-12 rounded-full bg-gray-100 dark:bg-white/5 flex items-center justify-center mb-3">
                      <Clock className="w-6 h-6 text-gray-400 dark:text-gray-500" />
                    </div>
                    <p className="text-sm font-medium text-gray-600 dark:text-gray-300">
                      No classes today
                    </p>
                    <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
                      Enjoy your free day!
                    </p>
                  </div>
                ) : (
                  <div className="space-y-6">
                    {todayCards.map((cls, idx) => (
                      <div key={cls.id} className="relative pl-6">
                        {idx !== todayCards.length - 1 && (
                          <div className="absolute left-[9px] top-6 bottom-[-24px] w-[2px] bg-gray-100 dark:bg-white/10" />
                        )}
                        <div className={cn(
                          "absolute left-0 top-1.5 w-[20px] h-[20px] rounded-full border-4 border-white dark:border-dark-card flex items-center justify-center",
                          cls.hasActiveSession
                            ? "bg-emerald-500/20"
                            : "bg-primary/20",
                        )}>
                          <div className={cn(
                            "w-2 h-2 rounded-full",
                            cls.hasActiveSession ? "bg-emerald-500 animate-pulse" : "bg-primary",
                          )} />
                        </div>
                        <div className={cn(
                          "rounded-xl p-4 border transition-colors",
                          cls.hasActiveSession
                            ? "bg-emerald-50/60 dark:bg-emerald-500/10 border-emerald-200 dark:border-emerald-500/30"
                            : "bg-gray-50/50 dark:bg-white/5 border-gray-100 dark:border-white/5 hover:border-primary/30",
                        )}>
                          <div className="flex items-center justify-between mb-1">
                            <p className="text-xs font-semibold text-primary dark:text-primary-accent">
                              {cls.time}
                            </p>
                            {cls.hasActiveSession && (
                              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider bg-emerald-500 text-white">
                                <span className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />
                                Live Now
                              </span>
                            )}
                          </div>
                          <p className="font-medium text-gray-900 dark:text-white leading-tight mb-2">
                            {cls.course}
                          </p>
                          <div className="flex items-center gap-3 text-xs text-gray-500 dark:text-gray-400">
                            <span>{cls.room}</span>
                            <span className="w-1 h-1 rounded-full bg-gray-300 dark:bg-gray-600" />
                            <span>{cls.days}</span>
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
                <Button
                  className="w-full mt-6 bg-gray-100 hover:bg-gray-200 text-gray-900 dark:bg-white/5 dark:hover:bg-white/10 dark:text-white border-0 transition-colors"
                  onClick={() => navigate("/student/schedule")}
                >
                  View Full Schedule
                </Button>
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
