import { useMemo } from "react";
import { motion } from "framer-motion";
import {
  BookOpen,
  PercentSquare,
  CheckCircle2,
  XCircle,
  Clock,
  MoreVertical,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { cn } from "@/utils/cn";
import { useAuthStore } from "@/store/useAuthStore";
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

export default function StudentDashboard() {
  const { user } = useAuthStore();
  const studentId = user?.id as number | undefined;

  const { data, isLoading } = useQuery({
    queryKey: ["studentOverview", studentId],
    queryFn: () => dashboardService.studentOverview(studentId),
    enabled: Boolean(studentId),
  });

  const attendance = data?.attendance ?? [];
  const schedule = data?.schedule ?? [];
  const normalizedAttendance = useMemo(
    () => attendance.map((record: any) => ({ ...record, status: String(record.status ?? '').toUpperCase() })),
    [attendance],
  );

  const courseCards = useMemo(() => {
    const grouped = new Map<string, { id: string; name: string; code: string; attended: number; total: number }>();

    normalizedAttendance.forEach((record: any) => {
      const name = record.course_title || record.course_name || record.course || `Course ${record.course_id ?? record.id}`;
      const code = record.course_code || record.courseId || record.course_id || '';
      const key = String(record.course_id ?? name);
      const entry = grouped.get(key) ?? { id: key, name, code: String(code), attended: 0, total: 0 };
      entry.total += 1;
      if (record.status === 'PRESENT' || record.status === 'LATE') {
        entry.attended += 1;
      }
      grouped.set(key, entry);
    });

    return Array.from(grouped.values()).map((course) => {
      const percent = course.total > 0 ? Math.round((course.attended / course.total) * 100) : 0;
      return {
        ...course,
        percent,
        status: percent >= 85 ? 'Good' : percent >= 70 ? 'Warning' : 'Low',
      };
    });
  }, [normalizedAttendance]);

  const todayCards = useMemo(
    () =>
      schedule.slice(0, 5).map((item: any, index: number) => ({
        id: String(item.id ?? index),
        time: item.time_label || `${item.start_time ?? 'TBA'} - ${item.end_time ?? 'TBA'}`,
        course: item.course_title || item.course_name || item.course_code || `Course ${item.course_id ?? item.id}`,
        room: item.room_name || item.location || 'TBA',
        section: item.section_name || item.weekday?.join?.(', ') || 'Scheduled',
      })),
    [schedule],
  );

  const attendanceRate = useMemo(() => {
    if (normalizedAttendance.length === 0) return 0;
    const attended = normalizedAttendance.filter((record: any) => record.status === 'PRESENT' || record.status === 'LATE').length;
    return Math.round((attended / normalizedAttendance.length) * 100);
  }, [normalizedAttendance]);

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

      {/* Stats Grid (simple derived from attendance) */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <Card className="glass-card border-gray-200 dark:border-white/5 hover:border-primary/50 dark:hover:border-primary/50 transition-colors group">
            <CardContent className="p-5 flex flex-col items-start gap-4">
              <div
                className={cn(
                  "p-3 rounded-xl transition-colors",
                  "bg-blue-500/10",
                )}
              >
                <BookOpen className={cn("w-6 h-6", "text-blue-500")} />
              </div>
              <div>
                <h3 className="text-3xl font-bold text-gray-900 dark:text-white mb-1">
                  {isLoading ? "—" : courseCards.length}
                </h3>
                <p className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  My Courses
                </p>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <Card className="glass-card border-gray-200 dark:border-white/5 hover:border-primary/50 dark:hover:border-primary/50 transition-colors group">
            <CardContent className="p-5 flex flex-col items-start gap-4">
              <div
                className={cn(
                  "p-3 rounded-xl transition-colors",
                  "bg-purple-500/10",
                )}
              >
                <PercentSquare className={cn("w-6 h-6", "text-purple-500")} />
              </div>
              <div>
                <h3 className="text-3xl font-bold text-gray-900 dark:text-white mb-1">
                  {isLoading ? "—" : `${attendanceRate}%`}
                </h3>
                <p className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Attendance Rate
                </p>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <Card className="glass-card border-gray-200 dark:border-white/5 hover:border-primary/50 dark:hover:border-primary/50 transition-colors group">
            <CardContent className="p-5 flex flex-col items-start gap-4">
              <div
                className={cn(
                  "p-3 rounded-xl transition-colors",
                  "bg-emerald-500/10",
                )}
              >
                <CheckCircle2 className={cn("w-6 h-6", "text-emerald-500")} />
              </div>
              <div>
                <h3 className="text-3xl font-bold text-gray-900 dark:text-white mb-1">
                  {isLoading
                    ? "—"
                    : normalizedAttendance.filter((a: any) => a.status === "PRESENT" || a.status === "LATE")
                        .length}
                </h3>
                <p className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Classes Attended
                </p>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <Card className="glass-card border-gray-200 dark:border-white/5 hover:border-primary/50 dark:hover:border-primary/50 transition-colors group">
            <CardContent className="p-5 flex flex-col items-start gap-4">
              <div
                className={cn(
                  "p-3 rounded-xl transition-colors",
                  "bg-rose-500/10",
                )}
              >
                <XCircle className={cn("w-6 h-6", "text-rose-500")} />
              </div>
              <div>
                <h3 className="text-3xl font-bold text-gray-900 dark:text-white mb-1">
                  {isLoading
                    ? "—"
                    : normalizedAttendance.filter((a: any) => a.status === "ABSENT")
                        .length}
                </h3>
                <p className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Missed Classes
                </p>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <Card className="glass-card border-gray-200 dark:border-white/5 hover:border-primary/50 dark:hover:border-primary/50 transition-colors group">
            <CardContent className="p-5 flex flex-col items-start gap-4">
              <div
                className={cn(
                  "p-3 rounded-xl transition-colors",
                  "bg-orange-500/10",
                )}
              >
                <Clock className={cn("w-6 h-6", "text-orange-500")} />
              </div>
              <div>
                <h3 className="text-3xl font-bold text-gray-900 dark:text-white mb-1">
                  {isLoading ? "—" : todayCards.length}
                </h3>
                <p className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                  Upcoming Classes
                </p>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Main Content Area - Courses Table */}
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
                      {courseCards.map((course) => (
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
                      ))}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </div>

        {/* Right Panel */}
        <div className="space-y-6">
          {/* Today's Classes */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.4 }}
          >
            <Card className="glass-card border-gray-200 dark:border-white/5">
              <CardHeader className="border-b border-gray-100 dark:border-white/5 pb-4">
                <CardTitle className="text-lg font-bold text-gray-900 dark:text-white">
                  Today's Classes
                </CardTitle>
              </CardHeader>
              <CardContent className="p-6">
                <div className="space-y-6">
                  {todayCards.map((cls, idx) => (
                    <div key={cls.id} className="relative pl-6">
                      {/* Timeline Line */}
                      {idx !== todayCards.length - 1 && (
                        <div className="absolute left-[9px] top-6 bottom-[-24px] w-[2px] bg-gray-100 dark:bg-white/10" />
                      )}
                      {/* Timeline Dot */}
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
                          <span>{cls.section}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
                <Button className="w-full mt-6 bg-gray-100 hover:bg-gray-200 text-gray-900 dark:bg-white/5 dark:hover:bg-white/10 dark:text-white border-0 transition-colors">
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
