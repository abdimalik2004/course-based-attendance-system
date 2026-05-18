import { useState } from 'react';
import { motion } from 'framer-motion';
import { Search, Filter, Download } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { useAuthStore } from '@/store/useAuthStore';
import { Card, CardContent } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { cn } from '@/utils/cn';
import dashboardService from '@/services/dashboardService';

const getStatusBadge = (status: string) => {
  switch (status?.toString().toUpperCase()) {
    case 'PRESENT':
      return <Badge variant="success">Present</Badge>;
    case 'LATE':
      return <Badge variant="warning">Late</Badge>;
    case 'ABSENT':
      return <Badge variant="danger">Absent</Badge>;
    default:
      return <Badge variant="default">{status ?? 'Unknown'}</Badge>;
  }
};

const getProgressColor = (percent: number) => {
  if (percent >= 85) return 'bg-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)]';
  if (percent >= 70) return 'bg-yellow-500 shadow-[0_0_10px_rgba(234,179,8,0.5)]';
  return 'bg-rose-500 shadow-[0_0_10px_rgba(244,63,94,0.5)]';
};

export default function StudentAttendance() {
  const { user } = useAuthStore();
  const studentId = user?.id as number | undefined;
  const [searchTerm, setSearchTerm] = useState('');

  const { data, isLoading, error } = useQuery({
    queryKey: ['studentAttendance', studentId],
    queryFn: async () => {
      const overview = await dashboardService.studentOverview(studentId);
      return overview?.attendance ?? [];
    },
    enabled: Boolean(studentId),
    staleTime: 1000 * 60 * 2,
  });

  const attendanceData = data ?? [];

  const filteredData = attendanceData.filter((course: any) =>
    course.course_name?.toLowerCase().includes(searchTerm.toLowerCase()) ||
    course.course_code?.toLowerCase().includes(searchTerm.toLowerCase()),
  );

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white tracking-tight">Attendance Record</h1>
          <p className="text-gray-500 dark:text-gray-400 mt-1">Detailed view of your class attendance for this semester.</p>
        </div>

        <div className="flex items-center gap-3">
          <div className="relative w-full sm:w-64">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={18} />
            <Input
              placeholder="Search courses..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-10 glass-input"
            />
          </div>
          <Button variant="secondary" className="shrink-0">
            <Filter size={18} className="mr-2" />
            Filter
          </Button>
          <Button variant="secondary" className="shrink-0 hidden sm:flex">
            <Download size={18} className="mr-2" />
            Export
          </Button>
        </div>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
      >
        <Card className="glass-card border-gray-200 dark:border-white/5 overflow-hidden">
          <CardContent className="p-0">
            <div className="overflow-x-auto custom-scrollbar w-full">
              <table className="w-full min-w-[800px] text-sm text-left whitespace-nowrap">
                <thead className="bg-gray-50/80 dark:bg-white/5 text-gray-500 dark:text-gray-400 font-medium border-b border-gray-100 dark:border-white/5">
                  <tr>
                    <th className="px-6 py-4">Course</th>
                    <th className="px-6 py-4">Code</th>
                    <th className="px-6 py-4 text-center">Classes Attended</th>
                    <th className="px-6 py-4 min-w-[200px]">Attendance Progress</th>
                    <th className="px-6 py-4">Last Updated</th>
                    <th className="px-6 py-4 text-center">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100 dark:divide-white/5">
                  {isLoading ? (
                    <tr>
                      <td colSpan={6} className="px-6 py-16 text-center text-gray-500 dark:text-gray-400">
                        Loading attendance records...
                      </td>
                    </tr>
                  ) : filteredData.length === 0 ? (
                    <tr>
                      <td colSpan={6} className="px-6 py-12 text-center text-gray-500 dark:text-gray-400">
                        No attendance records found.
                      </td>
                    </tr>
                  ) : (
                    filteredData.map((course: any) => (
                      <tr key={course.id} className="hover:bg-gray-50/50 dark:hover:bg-white/5 transition-colors group">
                        <td className="px-6 py-4">
                          <p className="font-semibold text-gray-900 dark:text-white">{course.course_name}</p>
                        </td>
                        <td className="px-6 py-4">
                          <p className="text-xs font-medium text-primary dark:text-primary-accent bg-primary/5 dark:bg-primary/10 px-2.5 py-1 rounded-md inline-block border border-primary/10 dark:border-primary/20">{course.course_code}</p>
                        </td>
                        <td className="px-6 py-4 text-center">
                          <span className="text-lg font-bold text-gray-900 dark:text-white">{course.classes_attended}</span>
                          <span className="text-gray-400 dark:text-gray-500 mx-1">/</span>
                          <span className="text-gray-500 dark:text-gray-400 font-medium">{course.total_classes}</span>
                        </td>
                        <td className="px-6 py-4">
                          <div className="flex flex-col gap-2">
                            <div className="flex justify-between items-center text-xs">
                              <span className="text-gray-500 dark:text-gray-400">Attendance</span>
                              <span className="font-bold text-gray-900 dark:text-white">{Math.round(course.attendance_percentage ?? 0)}%</span>
                            </div>
                            <div className="w-full bg-gray-200 dark:bg-gray-800 rounded-full h-2 overflow-hidden shadow-inner">
                              <motion.div
                                initial={{ width: 0 }}
                                animate={{ width: `${Math.round(course.attendance_percentage ?? 0)}%` }}
                                transition={{ duration: 1, ease: 'easeOut' }}
                                className={cn('h-full rounded-full', getProgressColor(Math.round(course.attendance_percentage ?? 0)))}
                              />
                            </div>
                          </div>
                        </td>
                        <td className="px-6 py-4">
                          <span className="text-gray-600 dark:text-gray-300">
                            {course.created_at ? new Date(course.created_at).toLocaleDateString() : '-'}
                          </span>
                        </td>
                        <td className="px-6 py-4 text-center">
                          {getStatusBadge(course.status)}
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>

            <div className="border-t border-gray-100 dark:border-white/5 px-6 py-4 flex items-center justify-between bg-gray-50/30 dark:bg-white/[0.02]">
              <span className="text-sm text-gray-500 dark:text-gray-400">
                Showing <span className="font-medium text-gray-900 dark:text-white">{filteredData.length}</span> attendance records
              </span>
              <div className="flex items-center gap-2">
                <Button variant="secondary" size="sm" disabled>Previous</Button>
                <Button variant="secondary" size="sm">Next</Button>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}
