import { useState, useMemo, useEffect } from 'react';
import { Search, Filter, Calendar as CalendarIcon, Download, ShieldCheck } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/Card';
import { Input } from '@/components/ui/Input';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/Table';
import { Badge } from '@/components/ui/Badge';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import attendanceService from '@/services/attendanceService';
import teacherService from '@/services/teacherService';
import { useTeacherId } from '@/store/useTeacherStore';

export default function AttendanceList() {
  const [searchTerm, setSearchTerm] = useState('');
  // courseFilter stores the course_id (number string) or 'All'
  const [courseFilter, setCourseFilter] = useState('All');
  const [statusFilter, setStatusFilter] = useState('All');
  // dateFilter applies client-side on recognizedAt — 'All' | 'Today' | 'Week'
  const [dateFilter, setDateFilter] = useState('All');
  const [page, setPage] = useState(1);
  const PAGE_SIZE = 50;
  const { teacherId, isUnlinked: isTeacherUnlinked } = useTeacherId();
  const queryClient = useQueryClient();

  // Excuse mutation — TEACHER can only change ABSENT → EXCUSED (#33)
  const excuseMutation = useMutation({
    mutationFn: (recordId: number) => teacherService.excuseAttendanceRecord(recordId),
    onSuccess: () => {
      // Refetch records so the row updates immediately
      queryClient.invalidateQueries({ queryKey: ['teacherAttendanceRecords'] });
    },
  });

  const {
    data: coursesData,
    isLoading: coursesLoading,
    isError: coursesError,
    error: coursesErrorObj,
    refetch: refetchCourses,
  } = useQuery({
    queryKey: ['teacherCourses', teacherId],
    queryFn: () => teacherService.getAssignedCourses(teacherId),
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
    queryKey: ['teacherAttendanceRecords', teacherId, courseFilter, statusFilter, page],
    // Always load once the teacher is identified — pass course_id when a specific
    // course is selected, omit it for "All Courses" (backend scopes to teacher).
    enabled: !!teacherId,
    queryFn: async () =>
      attendanceService.getAttendanceList({
        course_id: courseFilter !== 'All' ? Number(courseFilter) : undefined,
        status: statusFilter !== 'All' ? statusFilter.toUpperCase() : undefined,
        page,
        limit: PAGE_SIZE,
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

  // Auto-select the first (and only) course so records load immediately
  // without requiring the teacher to manually pick from a one-item dropdown.
  useEffect(() => {
    if (uniqueCourses.length === 1 && courseFilter === 'All') {
      setCourseFilter(uniqueCourses[0].id);
    }
  }, [uniqueCourses, courseFilter]);

  // Reset to page 1 when any server-side filter changes
  useEffect(() => {
    setPage(1);
  }, [courseFilter, statusFilter]);

  const records: any[] = attendanceData?.data ?? [];

  const hasError = coursesError || attendanceError;
  const errorMessage =
    coursesError
      ? (coursesErrorObj as Error)?.message ?? 'Failed to load assigned courses.'
      : attendanceError
        ? (attendanceErrorObj as Error)?.message ?? 'Failed to load attendance records.'
        : null;
  const hasCourses = uniqueCourses.length > 0;

  // Week bounds (Sat–Fri) for the date filter — computed once per render,
  // cheap enough not to need useMemo here.
  const todayDateStr = new Date().toDateString();
  const { weekFilterStart, weekFilterEnd } = (() => {
    const now = new Date();
    const day = now.getDay();
    const daysFromSat = day === 6 ? 0 : day + 1;
    const ws = new Date(now);
    ws.setDate(now.getDate() - daysFromSat);
    ws.setHours(0, 0, 0, 0);
    const we = new Date(ws);
    we.setDate(ws.getDate() + 6);
    we.setHours(23, 59, 59, 999);
    return { weekFilterStart: ws, weekFilterEnd: we };
  })();

  // Client-side search + date filter on top of server-filtered results
  const filteredRecords = records.filter((record: any) => {
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      const matchesSearch =
        (record.studentName ?? '').toLowerCase().includes(term) ||
        (record.course ?? '').toLowerCase().includes(term);
      if (!matchesSearch) return false;
    }

    if (dateFilter !== 'All' && record.recognizedAt) {
      const d = new Date(record.recognizedAt);
      if (dateFilter === 'Today') {
        if (d.toDateString() !== todayDateStr) return false;
      } else if (dateFilter === 'Week') {
        if (d < weekFilterStart || d > weekFilterEnd) return false;
      }
    }

    return true;
  });

  // ── CSV Export ──────────────────────────────────────────────────────────────
  const exportCSV = () => {
    const headers = ['Student Name', 'Course', 'Session ID', 'Attended / Total', 'Status', 'Confidence', 'Recognized At'];
    const rows = filteredRecords.map((r: any) => [
      r.studentName ?? '',
      r.course ?? '',
      r.sessionId ?? '',
      `${r.attendedSessions ?? 0} / ${r.totalSessions ?? 1}`,
      r.status ?? '',
      r.confidence != null && Number(r.confidence) > 0
        ? `${(Number(r.confidence) * 100).toFixed(1)}%`
        : '',
      r.recognizedAt
        ? new Date(r.recognizedAt).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })
        : '',
    ]);
    const csvContent = [headers, ...rows]
      .map((row) => row.map((cell) => `"${String(cell).replace(/"/g, '""')}"`).join(','))
      .join('\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    const courseLabel = courseFilter !== 'All'
      ? uniqueCourses.find((c) => c.id === courseFilter)?.title ?? 'course'
      : 'all-courses';
    link.download = `attendance-${courseLabel}-${new Date().toISOString().slice(0, 10)}.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

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
        {filteredRecords.length > 0 && (
          <button
            type="button"
            onClick={exportCSV}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-medium bg-gray-100 hover:bg-gray-200 dark:bg-white/10 dark:hover:bg-white/20 text-gray-700 dark:text-gray-200 transition-colors shrink-0"
          >
            <Download size={15} />
            Export CSV
          </button>
        )}
      </div>

      {isTeacherUnlinked && (
        <div className="rounded-2xl border border-amber-200 dark:border-amber-500/20 bg-amber-50 dark:bg-amber-500/10 p-4 text-sm text-amber-800 dark:text-amber-200">
          Your account is not yet linked to a teacher profile. Contact HR to link your login account before attendance records will appear here.
        </div>
      )}

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
                {coursesLoading ? 'Loading courses…' : 'All Courses'}
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

            {/* Date filter — client-side on recognizedAt */}
            <select
              value={dateFilter}
              onChange={(e) => setDateFilter(e.target.value)}
              className="h-10 rounded-xl glass-input px-4 text-sm text-gray-900 dark:text-gray-100 bg-transparent appearance-none pr-8 cursor-pointer border border-gray-200 dark:border-white/10 focus:outline-none focus:ring-2 focus:ring-primary/50"
              style={{
                backgroundImage: `url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%236B7280%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpolyline%20points%3D%226%209%2012%2015%2018%209%22%3E%3C%2Fpolyline%3E%3C%2Fsvg%3E")`,
                backgroundRepeat: 'no-repeat',
                backgroundPosition: 'right 0.5rem center',
                backgroundSize: '1em 1em',
              }}
            >
              <option value="All" className="bg-white dark:bg-dark-bg">All Time</option>
              <option value="Today" className="bg-white dark:bg-dark-bg">Today</option>
              <option value="Week" className="bg-white dark:bg-dark-bg">This Week</option>
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
                  <TableHead>Action</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {hasError ? (
                  <TableRow>
                    <TableCell colSpan={8} className="h-32 text-center text-gray-500">
                      {errorMessage}
                    </TableCell>
                  </TableRow>
                ) : attendanceLoading ? (
                  <>
                    {Array.from({ length: 5 }).map((_, i) => (
                      <TableRow key={i} className="animate-pulse">
                        <TableCell><div className="h-4 bg-gray-200 dark:bg-white/10 rounded w-32" /></TableCell>
                        <TableCell><div className="h-4 bg-gray-200 dark:bg-white/10 rounded w-28" /></TableCell>
                        <TableCell><div className="h-4 bg-gray-200 dark:bg-white/10 rounded w-16 font-mono" /></TableCell>
                        <TableCell><div className="h-4 bg-gray-200 dark:bg-white/10 rounded w-12" /></TableCell>
                        <TableCell><div className="h-5 bg-gray-200 dark:bg-white/10 rounded-full w-16" /></TableCell>
                        <TableCell><div className="h-4 bg-gray-200 dark:bg-white/10 rounded w-12" /></TableCell>
                        <TableCell><div className="h-4 bg-gray-200 dark:bg-white/10 rounded w-28" /></TableCell>
                        <TableCell><div className="h-7 bg-gray-200 dark:bg-white/10 rounded-lg w-16" /></TableCell>
                      </TableRow>
                    ))}
                  </>
                ) : filteredRecords.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={8} className="h-32 text-center text-gray-500">
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
                        {record.confidence != null && Number(record.confidence) > 0 ? (
                          // confidence is a 0-1 float from the recognizer
                          `${(Number(record.confidence) * 100).toFixed(1)}%`
                        ) : (
                          // Absent records are auto-created with confidence=0 — show dash
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
                      {/* Action column — Excuse button for Absent records (#33) */}
                      <TableCell>
                        {record.status === 'Absent' ? (
                          <button
                            type="button"
                            disabled={excuseMutation.isPending}
                            onClick={() => excuseMutation.mutate(record.id)}
                            className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-semibold bg-amber-50 text-amber-700 hover:bg-amber-100 dark:bg-amber-500/10 dark:text-amber-400 dark:hover:bg-amber-500/20 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                          >
                            <ShieldCheck size={12} />
                            Excuse
                          </button>
                        ) : null}
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </div>

          {/* Pagination + record count */}
          {!attendanceLoading && (
            <div className="mt-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
              <p className="text-xs text-gray-400 dark:text-gray-500">
                {filteredRecords.length > 0 ? (
                  <>
                    Showing {filteredRecords.length} record{filteredRecords.length !== 1 ? 's' : ''}
                    {' '}(page {page})
                    {statusFilter !== 'All' ? ` · ${statusFilter} only` : ''}
                    {dateFilter === 'Today' ? ' · Today' : dateFilter === 'Week' ? ' · This Week' : ''}
                  </>
                ) : null}
              </p>
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={page === 1}
                  className="px-3 py-1.5 text-xs font-medium rounded-lg border border-gray-200 dark:border-white/10 text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-white/5 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                >
                  ← Prev
                </button>
                <span className="text-xs text-gray-500 dark:text-gray-400 px-1">
                  Page {page}
                </span>
                <button
                  type="button"
                  onClick={() => setPage((p) => p + 1)}
                  disabled={records.length < PAGE_SIZE}
                  className="px-3 py-1.5 text-xs font-medium rounded-lg border border-gray-200 dark:border-white/10 text-gray-600 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-white/5 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                >
                  Next →
                </button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
