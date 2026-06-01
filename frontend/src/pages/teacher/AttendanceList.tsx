import { useState, useMemo } from 'react';
import { Search, Filter, Calendar as CalendarIcon } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/Card';
import { Input } from '@/components/ui/Input';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/Table';
import { Badge } from '@/components/ui/Badge';
import { useQuery } from '@tanstack/react-query';
import attendanceService from '@/services/attendanceService';
import courseService from '@/services/courseService';
import { useAuthStore } from '@/store/useAuthStore';

export default function AttendanceList() {
  const [searchTerm, setSearchTerm] = useState('');
  // courseFilter stores the course_id (number string) or 'All'
  const [courseFilter, setCourseFilter] = useState('All');
  const [statusFilter, setStatusFilter] = useState('All');
  const { user } = useAuthStore();
  const teacherId = Number(user?.teacherId ?? user?.id ?? 0);

  const {
    data: coursesData,
    isLoading: coursesLoading,
    isError: coursesError,
    error: coursesErrorObj,
    refetch: refetchCourses,
  } = useQuery({
    queryKey: ['teacherAttendanceCourses', teacherId],
    queryFn: () => courseService.listAssignments({ teacher_id: teacherId, skip: 0, limit: 200 }),
    enabled: !!teacherId,
    retry: false,
  });

  const {
    data: attendanceData,
    isLoading: attendanceLoading,
    isError: attendanceError,
    error: attendanceErrorObj,
    refetch: refetchAttendance,
  } = useQuery({
    queryKey: ['teacherAttendanceRecords', user?.id, courseFilter, statusFilter],
    // Load records once a course is selected; send course_id directly to backend
    enabled: courseFilter !== 'All' && !!user?.id,
    queryFn: async () =>
      attendanceService.getAttendanceList({
        course_id: courseFilter !== 'All' ? Number(courseFilter) : undefined,
        status: statusFilter !== 'All' ? statusFilter.toUpperCase() : undefined,
        limit: 200,
      }),
    retry: false,
    refetchInterval: 15_000,
  });

  // Build course list from assignments — now includes course_title from backend
  const uniqueCourses = useMemo(() => {
    const list: any[] = coursesData?.items ?? coursesData ?? [];
    return list
      .map((assignment: any) => ({
        id: String(assignment.course_id),
        title: assignment.course_title ?? `Course ${assignment.course_id}`,
        code: assignment.course_code ?? '',
      }))
      .filter((c) => c.id && c.id !== 'undefined');
  }, [coursesData]);

  const records: any[] = attendanceData?.data ?? [];

  const hasError = coursesError || attendanceError;
  const errorMessage =
    coursesError
      ? (coursesErrorObj as Error)?.message ?? 'Failed to load assigned courses.'
      : attendanceError
        ? (attendanceErrorObj as Error)?.message ?? 'Failed to load attendance records.'
        : null;
  const hasCourses = uniqueCourses.length > 0;

  // Client-side search filter on top of server-filtered results
  const filteredRecords = records.filter((record: any) => {
    if (!searchTerm) return true;
    const term = searchTerm.toLowerCase();
    return (
      (record.studentName ?? '').toLowerCase().includes(term) ||
      (record.course ?? '').toLowerCase().includes(term)
    );
  });

  const getStatusBadgeVariant = (status: string) => {
    switch (status) {
      case 'Present': return 'success';
      case 'Late': return 'warning';
      case 'Absent': return 'danger';
      case 'Excused': return 'neutral';
      default: return 'neutral';
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-gray-100">
            Attendance List
          </h1>
          <p className="text-gray-500 dark:text-gray-400 mt-1">
            View student attendance records for your assigned courses
          </p>
        </div>
      </div>

      {hasError && errorMessage ? (
        <div className="rounded-2xl border border-rose-200 dark:border-rose-500/20 bg-rose-50 dark:bg-rose-500/10 p-4 text-sm text-rose-700 dark:text-rose-200 flex flex-col gap-3">
          <div>{errorMessage}</div>
          <div className="flex gap-2">
            {coursesError && (
              <button
                type="button"
                onClick={() => refetchCourses()}
                className="inline-flex items-center justify-center rounded-full bg-rose-500 px-4 py-2 text-sm font-semibold text-white hover:bg-rose-600 transition"
              >
                Retry Courses
              </button>
            )}
            {attendanceError && (
              <button
                type="button"
                onClick={() => refetchAttendance()}
                className="inline-flex items-center justify-center rounded-full bg-rose-500 px-4 py-2 text-sm font-semibold text-white hover:bg-rose-600 transition"
              >
                Retry Records
              </button>
            )}
          </div>
        </div>
      ) : null}

      <Card className="glass-card shadow-2xl shadow-primary/5">
        <CardContent className="p-6">
          {/* Row 1: Search */}
          <div className="mb-4">
            <div className="relative w-full sm:max-w-md">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={18} />
              <Input
                placeholder="Search by student name or course..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-10 bg-white/50 dark:bg-white/5"
              />
            </div>
          </div>

          {/* Row 2: Filters */}
          <div className="flex flex-wrap items-center gap-2 mb-6">
            <Filter className="text-gray-400" size={18} />

            {/* Course filter */}
            <select
              value={courseFilter}
              onChange={(e) => setCourseFilter(e.target.value)}
              disabled={coursesLoading || !hasCourses}
              className="h-10 rounded-xl glass-input px-4 text-sm text-gray-900 dark:text-gray-100 bg-transparent appearance-none pr-8 cursor-pointer border border-gray-200 dark:border-white/10 focus:outline-none focus:ring-2 focus:ring-primary/50"
              style={{
                backgroundImage: `url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%236B7280%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpolyline%20points%3D%226%209%2012%2015%2018%209%22%3E%3C%2Fpolyline%3E%3C%2Fsvg%3E")`,
                backgroundRepeat: 'no-repeat',
                backgroundPosition: 'right 0.5rem center',
                backgroundSize: '1em 1em',
              }}
            >
              <option value="All" className="bg-white dark:bg-dark-bg">
                {coursesLoading
                  ? 'Loading courses…'
                  : hasCourses
                    ? 'Select a course'
                    : 'No assigned courses'}
              </option>
              {uniqueCourses.map((course) => (
                <option
                  key={course.id}
                  value={course.id}
                  className="bg-white dark:bg-dark-bg"
                >
                  {course.code ? `${course.code} — ${course.title}` : course.title}
                </option>
              ))}
            </select>

            {/* Status filter */}
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="h-10 rounded-xl glass-input px-4 text-sm text-gray-900 dark:text-gray-100 bg-transparent appearance-none pr-8 cursor-pointer border border-gray-200 dark:border-white/10 focus:outline-none focus:ring-2 focus:ring-primary/50"
              style={{
                backgroundImage: `url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%236B7280%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpolyline%20points%3D%226%209%2012%2015%2018%209%22%3E%3C%2Fpolyline%3E%3C%2Fsvg%3E")`,
                backgroundRepeat: 'no-repeat',
                backgroundPosition: 'right 0.5rem center',
                backgroundSize: '1em 1em',
              }}
            >
              <option value="All" className="bg-white dark:bg-dark-bg">All Statuses</option>
              <option value="Present" className="bg-white dark:bg-dark-bg">Present</option>
              <option value="Late" className="bg-white dark:bg-dark-bg">Late</option>
              <option value="Absent" className="bg-white dark:bg-dark-bg">Absent</option>
              <option value="Excused" className="bg-white dark:bg-dark-bg">Excused</option>
            </select>
          </div>

          <div className="overflow-x-auto custom-scrollbar rounded-xl border border-gray-100 dark:border-white/5">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Student Name</TableHead>
                  <TableHead>Course</TableHead>
                  <TableHead>Session ID</TableHead>
                  <TableHead>Attendance</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Confidence</TableHead>
                  <TableHead>Recognized At</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {hasError ? (
                  <TableRow>
                    <TableCell colSpan={7} className="h-32 text-center text-gray-500">
                      {errorMessage}
                    </TableCell>
                  </TableRow>
                ) : courseFilter === 'All' ? (
                  <TableRow>
                    <TableCell colSpan={7} className="h-32 text-center text-gray-500">
                      Select a course above to view attendance records.
                    </TableCell>
                  </TableRow>
                ) : attendanceLoading ? (
                  <TableRow>
                    <TableCell colSpan={7} className="h-32 text-center text-gray-500">
                      Loading attendance records…
                    </TableCell>
                  </TableRow>
                ) : filteredRecords.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={7} className="h-32 text-center text-gray-500">
                      No attendance records found for the selected filters.
                    </TableCell>
                  </TableRow>
                ) : (
                  filteredRecords.map((record) => (
                    <TableRow key={record.id} className="hover:bg-gray-50/50 dark:hover:bg-white/[0.02]">
                      <TableCell className="font-medium text-gray-900 dark:text-gray-100">
                        {record.studentName}
                      </TableCell>
                      <TableCell className="text-gray-600 dark:text-gray-400">
                        {record.course}
                      </TableCell>
                      <TableCell className="text-gray-500 dark:text-gray-400 font-mono text-sm">
                        {record.sessionId}
                      </TableCell>
                      <TableCell className="text-gray-700 dark:text-gray-300 font-medium">
                        {record.attendedSessions} / {record.totalSessions}
                      </TableCell>
                      <TableCell>
                        <Badge variant={getStatusBadgeVariant(record.status) as any}>
                          {record.status}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-sm font-medium text-gray-600 dark:text-gray-400">
                        {record.confidence != null ? (
                          `${record.confidence}%`
                        ) : (
                          <span className="text-gray-400">—</span>
                        )}
                      </TableCell>
                      <TableCell>
                        <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400">
                          {record.recognizedAt ? (
                            <>
                              <CalendarIcon size={14} className="text-gray-400" />
                              <span>
                                {new Date(record.recognizedAt).toLocaleString([], {
                                  dateStyle: 'medium',
                                  timeStyle: 'short',
                                })}
                              </span>
                            </>
                          ) : (
                            <span className="pl-4">—</span>
                          )}
                        </div>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </div>

          {/* Record count */}
          {courseFilter !== 'All' && !attendanceLoading && filteredRecords.length > 0 && (
            <p className="mt-3 text-xs text-gray-400 dark:text-gray-500">
              Showing {filteredRecords.length} record{filteredRecords.length !== 1 ? 's' : ''}
              {statusFilter !== 'All' ? ` · ${statusFilter} only` : ''}
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
