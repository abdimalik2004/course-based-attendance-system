import { useMemo, useState } from 'react';
import { motion } from 'framer-motion';
import { Search, Calendar as CalendarIcon, Clock, MapPin } from 'lucide-react';
import { useQuery } from '@tanstack/react-query';
import { Card, CardContent } from '@/components/ui/Card';
import { Input } from '@/components/ui/Input';
import { cn } from '@/utils/cn';
import dashboardService, { type StudentScheduleItem } from '@/services/dashboardService';

const daysOfWeek = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];

export default function StudentSchedule() {
  const [searchTerm, setSearchTerm] = useState('');
  const [filterDay, setFilterDay] = useState('All');

  const { data: rawSchedule = [], isLoading, isError, refetch } = useQuery<StudentScheduleItem[]>({
    queryKey: ['studentSchedule'],
    queryFn: () => dashboardService.studentScheduleData(),
    staleTime: 1000 * 60 * 2,
  });

  const scheduleData = useMemo(() => {
    return rawSchedule.map((item) => ({
      id: item.id,
      course: item.course_name,
      code: item.course_code,
      days: item.weekdays,
      start: item.start_time,
      end: item.end_time,
      grace: `${item.grace_period_minutes} Mins`,
      className: item.class_name,
      hasActiveSession: item.has_active_session,
    }));
  }, [rawSchedule]);

  const filteredSchedule = scheduleData.filter((item) => {
    const matchesSearch =
      item.course.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.code.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesDay = filterDay === 'All' || item.days.includes(filterDay);
    return matchesSearch && matchesDay;
  });

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white tracking-tight">Class Schedule</h1>
          <p className="text-gray-500 dark:text-gray-400 mt-1">Your weekly timetable for the current semester.</p>
        </div>
        <div className="relative w-full sm:w-64">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={18} />
          <Input
            placeholder="Search courses..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10 glass-input"
          />
        </div>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-wrap gap-2"
      >
        <button
          onClick={() => setFilterDay('All')}
          className={cn(
            'px-4 py-2 rounded-full text-sm font-medium transition-all duration-200',
            filterDay === 'All'
              ? 'bg-primary text-white shadow-[0_0_15px_rgba(37,99,235,0.4)]'
              : 'bg-white dark:bg-white/5 text-gray-600 dark:text-gray-400 border border-gray-200 dark:border-white/10 hover:border-primary/50',
          )}
        >
          All Days
        </button>
        {daysOfWeek.map((day) => (
          <button
            key={day}
            onClick={() => setFilterDay(day)}
            className={cn(
              'px-4 py-2 rounded-full text-sm font-medium transition-all duration-200',
              filterDay === day
                ? 'bg-primary text-white shadow-[0_0_15px_rgba(37,99,235,0.4)]'
                : 'bg-white dark:bg-white/5 text-gray-600 dark:text-gray-400 border border-gray-200 dark:border-white/10 hover:border-primary/50',
            )}
          >
            {day}
          </button>
        ))}
      </motion.div>

      {isError && (
        <div className="flex items-center justify-between rounded-xl border border-rose-200 dark:border-rose-500/20 bg-rose-50 dark:bg-rose-500/10 px-4 py-3 text-sm text-rose-700 dark:text-rose-300">
          <span>Failed to load your schedule. Check your connection.</span>
          <button
            onClick={() => void refetch()}
            className="ml-4 shrink-0 text-xs font-semibold underline hover:no-underline"
          >
            Retry
          </button>
        </div>
      )}

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
                    <th className="px-6 py-4 min-w-[200px]">Weekdays</th>
                    <th className="px-6 py-4">Timing</th>
                    <th className="px-6 py-4">Class</th>
                    <th className="px-6 py-4 text-center">Grace Period</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100 dark:divide-white/5">
                  {isLoading ? (
                    Array.from({ length: 4 }).map((_, i) => (
                      <tr key={i} className="animate-pulse">
                        <td className="px-6 py-4">
                          <div className="h-3.5 w-40 rounded bg-gray-100 dark:bg-white/5" />
                        </td>
                        <td className="px-6 py-4">
                          <div className="h-5 w-16 rounded-md bg-gray-100 dark:bg-white/5" />
                        </td>
                        <td className="px-6 py-4">
                          <div className="flex gap-2">
                            <div className="h-6 w-10 rounded-md bg-gray-100 dark:bg-white/5" />
                            <div className="h-6 w-10 rounded-md bg-gray-100 dark:bg-white/5" />
                          </div>
                        </td>
                        <td className="px-6 py-4">
                          <div className="h-3.5 w-28 rounded bg-gray-100 dark:bg-white/5" />
                        </td>
                        <td className="px-6 py-4">
                          <div className="h-3.5 w-20 rounded bg-gray-100 dark:bg-white/5" />
                        </td>
                        <td className="px-6 py-4 text-center">
                          <div className="h-6 w-16 rounded-lg bg-gray-100 dark:bg-white/5 mx-auto" />
                        </td>
                      </tr>
                    ))
                  ) : filteredSchedule.length === 0 ? (
                    <tr>
                      <td colSpan={6} className="px-6 py-16 text-center text-gray-500 dark:text-gray-400">
                        <CalendarIcon className="w-12 h-12 mx-auto mb-3 opacity-20" />
                        <p className="text-base">No classes scheduled for {filterDay === 'All' ? 'selected filters' : filterDay}.</p>
                      </td>
                    </tr>
                  ) : (
                    filteredSchedule.map((schedule) => (
                      <tr
                        key={schedule.id}
                        className={cn(
                          'transition-colors group',
                          schedule.hasActiveSession
                            ? 'bg-emerald-50/60 dark:bg-emerald-500/5 hover:bg-emerald-50 dark:hover:bg-emerald-500/10'
                            : 'hover:bg-gray-50/50 dark:hover:bg-white/5',
                        )}
                      >
                        <td className="px-6 py-4">
                          <div className="flex items-center gap-2">
                            <p className="font-semibold text-gray-900 dark:text-white">{schedule.course}</p>
                            {schedule.hasActiveSession && (
                              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider bg-emerald-500 text-white">
                                <span className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />
                                Live
                              </span>
                            )}
                          </div>
                        </td>
                        <td className="px-6 py-4">
                          <p className="text-xs font-medium text-primary dark:text-primary-accent bg-primary/5 dark:bg-primary/10 px-2.5 py-1 rounded-md inline-block border border-primary/10 dark:border-primary/20">
                            {schedule.code}
                          </p>
                        </td>
                        <td className="px-6 py-4">
                          <div className="flex gap-2 flex-wrap">
                            {schedule.days.map((day) => (
                              <span
                                key={day}
                                className={cn(
                                  'px-2.5 py-1 text-xs font-medium rounded-md',
                                  day === 'Sat' || day === 'Sun'
                                    ? 'bg-purple-100 text-purple-700 dark:bg-purple-500/20 dark:text-purple-300 border border-purple-200 dark:border-purple-500/30'
                                    : 'bg-blue-50 text-blue-700 dark:bg-blue-500/10 dark:text-blue-400 border border-blue-100 dark:border-blue-500/20',
                                )}
                              >
                                {day}
                              </span>
                            ))}
                          </div>
                        </td>
                        <td className="px-6 py-4">
                          <div className="flex items-center gap-2 text-gray-700 dark:text-gray-300">
                            <Clock size={16} className="text-gray-400" />
                            <span className="font-medium">{schedule.start}</span>
                            <span className="text-gray-400">-</span>
                            <span className="font-medium">{schedule.end}</span>
                          </div>
                        </td>
                        <td className="px-6 py-4">
                          <div className="flex items-center gap-2 text-gray-600 dark:text-gray-400">
                            <MapPin size={16} className="text-gray-400" />
                            <span>{schedule.className ?? <span className="text-gray-400 italic">N/A</span>}</span>
                          </div>
                        </td>
                        <td className="px-6 py-4 text-center">
                          <span className="inline-flex items-center justify-center px-2.5 py-1 text-xs font-medium bg-gray-100 text-gray-600 dark:bg-white/5 dark:text-gray-400 rounded-lg border border-gray-200 dark:border-white/10">
                            {schedule.grace}
                          </span>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>

            <div className="border-t border-gray-100 dark:border-white/5 px-6 py-4 bg-gray-50/30 dark:bg-white/[0.02]">
              <span className="text-sm text-gray-500 dark:text-gray-400">
                Showing <span className="font-medium text-gray-900 dark:text-white">{filteredSchedule.length}</span> classes
              </span>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}
