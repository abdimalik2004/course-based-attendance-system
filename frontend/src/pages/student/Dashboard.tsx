import { useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import {
  PercentSquare,
  CheckCircle2,
  XCircle,
  Clock,
  BadgeCheck,
  MoreVertical,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { cn } from "@/utils/cn";
import { useQuery } from "@tanstack/react-query";
import dashboardService from "@/services/dashboardService";

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

  const { data, isLoading } = useQuery({
    queryKey: ["studentOverview"],
    queryFn: () => dashboardService.studentOverview(),
    refetchInterval: 60_000, // auto-refresh every 60 s
  });

  const attendance: any[] = data?.attendance ?? [];
  const schedule: any[] = data?.schedule ?? [];

  // Per-course cards from attendance aggregation
  const courseCards = useMemo(
    () =>
      attendance.map((course: any) => ({
        id: String(course.id),
        name: course.course_name ?? `Course ${course.id}`,
        code: course.course_code ?? "",
        attended: course.classes_attended ?? 0,
        absent: course.classes_absent ?? 0,
        excused: course.classes_excused ?? 0,
        total: course.total_classes ?? 0,
        percent: Math.round(course.attendance_percentage ?? 0),
        status: course.status ?? "Low",
      })),
    [attendance],
  );

  // Today's display label e.g. "Mon", "Tue" — matches backend weekday display labels
  const todayLabel = useMemo(() => {
    const days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    return days[new Date().getDay()];
  }, []);

  // Today's scheduled classes
  const todayCards = useMemo(
    () =>
      schedule
        .filter(
          (item: any) =>
            Array.isArray(item.weekdays) && item.weekdays.includes(todayLabel),
        )
        .sort((a: any, b: any) =>
          (a.start_time ?? "").localeCompare(b.start_time ?? ""),
        )
        .map((item: any, index: number) => ({
          id: String(item.id ?? index),
          time: `${item.start_time ?? "TBA"} - ${item.end_time ?? "TBA"}`,
          course:
            item.course_name ?? item.course_code ?? `Course ${index + 1}`,
          room: item.room_name ?? "TBA",
          days: Array.isArray(item.weekdays) ? item.weekdays.join(", ") : "Scheduled",
        })),
    [schedule, todayLabel],
  );

  // Stat computations
  const attendanceRate = useMemo(() => {
    const total = courseCards.reduce((s, c) => s + c.total, 0);
    const attended = courseCards.reduce((s, c) => s + c.attended, 0);
    return total > 0 ? Math.round((attended / total) * 100) : 0;
  }, [courseCards]);

  const totalAttended = useMemo(
    () => courseCards.reduce((s, c) => s + c.attended, 0),
    [courseCards],
  );

  const totalAbsent = useMemo(
    () => courseCards.reduce((s, c) => s + c.absent, 0),
    [courseCards],
  );

  const totalExcused = useMemo(
    () => courseCards.reduce((s, c) => s + c.excused, 0),
    [courseCards],
  );

  // Upcoming classes count = all schedule slots in the current Sat-Fri week
  const upcomingCount = useMemo(() => {
    // Each recurring schedule entry contributes once per weekday per week
    return schedule.reduce(
      (sum: number, item: any) =>
        sum + (Array.isArray(item.weekdays) ? item.weekdays.length : 0),
      0,
    );
  }, [schedule]);

  const statCards = [
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
  ];

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

      {/* Stats Grid — 5 cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
        {statCards.map((card, i) => (
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
                        <th className="px-6 py-4 text-right"></th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-100 dark:divide-white/5">
                      {isLoading ? (
                        <tr>
                          <td
                            colSpan={5}
                            className="px-6 py-10 text-center text-gray-400 dark:text-gray-500"
                          >
                            Loading…
                          </td>
                        </tr>
                      ) : courseCards.length === 0 ? (
                        <tr>
                          <td
                            colSpan={5}
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
                            <td className="px-6 py-4 text-right">
                              <button className="text-gray-400 hover:text-gray-900 dark:hover:text-white transition-colors opacity-0 group-hover:opacity-100">
                                <MoreVertical size={18} />
                              </button>
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
                {todayCards.length === 0 ? (
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
                        <div className="absolute left-0 top-1.5 w-[20px] h-[20px] rounded-full bg-primary/20 border-4 border-white dark:border-dark-card flex items-center justify-center">
                          <div className="w-2 h-2 rounded-full bg-primary" />
                        </div>
                        <div className="bg-gray-50/50 dark:bg-white/5 rounded-xl p-4 border border-gray-100 dark:border-white/5 hover:border-primary/30 transition-colors">
                          <p className="text-xs font-semibold text-primary dark:text-primary-accent mb-1">
                            {cls.time}
                          </p>
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
