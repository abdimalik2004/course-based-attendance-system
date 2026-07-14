import { useState, useMemo, useRef, useEffect } from 'react';
import { Search, Filter, Calendar as CalendarIcon, Download, ChevronLeft, ChevronRight } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/Card';
import { Input } from '@/components/ui/Input';
import { Button } from '@/components/ui/Button';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/Table';
import { Badge } from '@/components/ui/Badge';
import { useAttendanceList } from '@/hooks/queries/useAttendance';
import { useAcademiaStore } from '@/store/useAcademiaStore';

const PAGE_SIZE = 50;

const SELECT_STYLE: React.CSSProperties = {
  backgroundImage: `url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%236B7280%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpolyline%20points%3D%226%209%2012%2015%2018%209%22%3E%3C%2Fpolyline%3E%3C%2Fsvg%3E")`,
  backgroundRepeat: 'no-repeat',
  backgroundPosition: 'right 0.5rem center',
  backgroundSize: '1em 1em',
};

const SELECT_CLS = 'h-10 rounded-xl px-3 pr-8 text-sm text-gray-900 dark:text-gray-100 bg-white dark:bg-white/5 appearance-none cursor-pointer border border-gray-200 dark:border-white/10 focus:outline-none focus:ring-2 focus:ring-primary/50 shrink-0';

export default function AttendanceList() {
  const { faculties, departments, courses, fetchData, isLoading: academiaLoading } = useAcademiaStore();

  // Filter state
  const [searchTerm, setSearchTerm] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [facultyFilter, setFacultyFilter] = useState('All');
  const [departmentFilter, setDepartmentFilter] = useState('All');
  const [courseFilter, setCourseFilter] = useState('All');
  const [statusFilter, setStatusFilter] = useState('All');
  const [page, setPage] = useState(1);

  // Load academia data for filter dropdowns
  useEffect(() => {
    if (!academiaLoading && faculties.length === 0) fetchData();
  }, [fetchData, faculties.length, academiaLoading]);

  // Debounce search — reset to page 1 when search changes
  useEffect(() => {
    const t = setTimeout(() => {
      setDebouncedSearch(searchTerm);
      setPage(1);
    }, 300);
    return () => clearTimeout(t);
  }, [searchTerm]);

  // Reset page when any filter changes
  const handleFacultyChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setFacultyFilter(e.target.value);
    setDepartmentFilter('All');
    setCourseFilter('All');
    setPage(1);
  };
  const handleDepartmentChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setDepartmentFilter(e.target.value);
    setCourseFilter('All');
    setPage(1);
  };
  const handleCourseChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setCourseFilter(e.target.value);
    setPage(1);
  };
  const handleStatusChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setStatusFilter(e.target.value);
    setPage(1);
  };

  // Build server-side query params
  const queryParams = useMemo(() => ({
    page,
    limit: PAGE_SIZE,
    search: debouncedSearch || undefined,
    faculty: facultyFilter !== 'All' ? facultyFilter : undefined,
    department: departmentFilter !== 'All' ? departmentFilter : undefined,
    course: courseFilter !== 'All' ? courseFilter : undefined,
    status: statusFilter !== 'All' ? statusFilter.toUpperCase() : undefined,
  }), [page, debouncedSearch, facultyFilter, departmentFilter, courseFilter, statusFilter]);

  const { data, isLoading, error } = useAttendanceList(queryParams);

  const records = data?.data ?? [];
  const total = data?.total ?? 0;
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  // Cascaded dropdown options from academia store
  const facultiesList = useMemo(() => faculties, [faculties]);
  const departmentsList = useMemo(() => {
    if (facultyFilter === 'All') return departments;
    const fac = faculties.find(f => f.name === facultyFilter);
    return fac ? departments.filter(d => d.facultyId === fac.id) : departments;
  }, [facultyFilter, departments, faculties]);
  const coursesList = useMemo(() => {
    if (departmentFilter === 'All' && facultyFilter === 'All') return courses;
    let filtered = courses;
    if (departmentFilter !== 'All') {
      const dept = departments.find(d => d.name === departmentFilter);
      if (dept) filtered = filtered.filter(c => c.departmentId === dept.id);
    } else if (facultyFilter !== 'All') {
      const fac = faculties.find(f => f.name === facultyFilter);
      if (fac) filtered = filtered.filter(c => c.facultyId === fac.id);
    }
    return filtered;
  }, [facultyFilter, departmentFilter, courses, departments, faculties]);

  // CSV export — exports ALL displayed records on current page
  const handleExportCSV = () => {
    if (records.length === 0) return;
    const headers = ['Student Name', 'Course', 'Session ID', 'Attended/Total', 'Status', 'Confidence', 'Recognized At'];
    const rows = records.map((r: any) => [
      r.studentName,
      r.course,
      r.sessionId,
      `${r.attendedSessions}/${r.totalSessions}`,
      r.status,
      r.confidence ?? '',
      r.recognizedAt ? new Date(r.recognizedAt).toLocaleString() : '',
    ]);
    const csv = [headers, ...rows]
      .map(row => row.map(c => `"${String(c ?? '').replace(/"/g, '""')}"`).join(','))
      .join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `attendance_list_page${page}.csv`;
    a.click();
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
            View and manage student attendance records
          </p>
        </div>
        <Button
          variant="secondary"
          size="sm"
          onClick={handleExportCSV}
          disabled={records.length === 0}
          className="shrink-0 gap-2"
        >
          <Download size={15} />
          Export CSV
        </Button>
      </div>

      {error ? (
        <div className="rounded-2xl border border-rose-200 dark:border-rose-500/20 bg-rose-50 dark:bg-rose-500/10 p-4 text-sm text-rose-700 dark:text-rose-200">
          Failed to load attendance records.
        </div>
      ) : null}

      <Card className="glass-card shadow-2xl shadow-primary/5">
        <CardContent className="p-6">
          <div className="flex flex-col gap-3 mb-6">
            {/* ROW 1 — search + faculty filter */}
            <div className="flex items-center gap-2">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" size={16} />
                <Input
                  placeholder="Search by student or course..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-9 h-10 text-sm bg-white/50 dark:bg-white/5 w-full"
                />
              </div>
              <select value={facultyFilter} onChange={handleFacultyChange} className={SELECT_CLS} style={SELECT_STYLE}>
                <option value="All" className="bg-white dark:bg-dark-bg">All Faculties</option>
                {facultiesList.map(f => (
                  <option key={f.id} value={f.name} className="bg-white dark:bg-dark-bg">{f.name}</option>
                ))}
              </select>
            </div>

            {/* ROW 2 — department, course, status filters */}
            <div className="flex items-center gap-2">
              <Filter className="text-gray-400 shrink-0" size={16} />
              <select value={departmentFilter} onChange={handleDepartmentChange} className={SELECT_CLS} style={SELECT_STYLE}>
                <option value="All" className="bg-white dark:bg-dark-bg">All Departments</option>
                {departmentsList.map(d => (
                  <option key={d.id} value={d.name} className="bg-white dark:bg-dark-bg">{d.name}</option>
                ))}
              </select>
              <select value={courseFilter} onChange={handleCourseChange} className={SELECT_CLS} style={SELECT_STYLE}>
                <option value="All" className="bg-white dark:bg-dark-bg">All Courses</option>
                {coursesList.map(c => (
                  <option key={c.id} value={c.title} className="bg-white dark:bg-dark-bg">{c.title}</option>
                ))}
              </select>
              <select value={statusFilter} onChange={handleStatusChange} className={SELECT_CLS} style={SELECT_STYLE}>
                <option value="All" className="bg-white dark:bg-dark-bg">All Statuses</option>
                <option value="Present" className="bg-white dark:bg-dark-bg">Present</option>
                <option value="Late" className="bg-white dark:bg-dark-bg">Late</option>
                <option value="Absent" className="bg-white dark:bg-dark-bg">Absent</option>
                <option value="Excused" className="bg-white dark:bg-dark-bg">Excused</option>
              </select>
            </div>
          </div>

          <div className="overflow-auto custom-scrollbar rounded-xl border border-gray-100 dark:border-white/5">
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
                {isLoading ? (
                  Array.from({ length: 8 }).map((_, i) => (
                    <TableRow key={`skel-${i}`}>
                      {Array.from({ length: 7 }).map((__, j) => (
                        <TableCell key={j}>
                          <div className="h-4 w-full max-w-[120px] bg-gray-200 dark:bg-white/10 rounded animate-pulse" />
                        </TableCell>
                      ))}
                    </TableRow>
                  ))
                ) : records.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={7} className="h-32 text-center text-gray-500">
                      No attendance records found.
                    </TableCell>
                  </TableRow>
                ) : (
                  records.map((record: any) => (
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
                        {record.confidence != null ? record.confidence : <span className="text-gray-400">-</span>}
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
                            <span className="pl-4">-</span>
                          )}
                        </div>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </div>

          {/* Pagination */}
          <div className="mt-4 flex items-center justify-between text-sm text-gray-500 dark:text-gray-400">
            <span>
              {total > 0
                ? `Showing ${(page - 1) * PAGE_SIZE + 1}–${Math.min(page * PAGE_SIZE, total)} of ${total} records`
                : 'No records'}
            </span>
            <div className="flex items-center gap-1">
              <Button
                variant="ghost"
                size="sm"
                className="h-8 w-8 p-0"
                disabled={page <= 1 || isLoading}
                onClick={() => setPage(p => Math.max(1, p - 1))}
              >
                <ChevronLeft size={16} />
              </Button>
              {Array.from({ length: totalPages }, (_, i) => i + 1)
                .filter(p => p === 1 || p === totalPages || Math.abs(p - page) <= 1)
                .reduce<(number | '…')[]>((acc, p, idx, arr) => {
                  if (idx > 0 && p - (arr[idx - 1] as number) > 1) acc.push('…');
                  acc.push(p);
                  return acc;
                }, [])
                .map((p, idx) =>
                  p === '…' ? (
                    <span key={`ellipsis-${idx}`} className="px-1">…</span>
                  ) : (
                    <Button
                      key={p}
                      variant={p === page ? 'primary' : 'ghost'}
                      size="sm"
                      className="h-8 w-8 p-0"
                      disabled={isLoading}
                      onClick={() => setPage(p as number)}
                    >
                      {p}
                    </Button>
                  )
                )}
              <Button
                variant="ghost"
                size="sm"
                className="h-8 w-8 p-0"
                disabled={page >= totalPages || isLoading}
                onClick={() => setPage(p => p + 1)}
              >
                <ChevronRight size={16} />
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
