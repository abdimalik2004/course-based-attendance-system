import { useState, useMemo, useRef, useEffect } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import { Search, Filter, Calendar as CalendarIcon } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/Card';
import { Input } from '@/components/ui/Input';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/Table';
import { Badge } from '@/components/ui/Badge';
import { useAttendanceList } from '@/hooks/queries/useAttendance';
import { useAcademiaStore } from '@/store/useAcademiaStore';

export default function AttendanceList() {
  const { data, isLoading, error } = useAttendanceList({ page: 1, limit: 200 });
  const { faculties, departments, courses, fetchData, isLoading: academiaLoading } = useAcademiaStore();
  const [searchTerm, setSearchTerm] = useState('');
  const [facultyFilter, setFacultyFilter] = useState('All');
  const [departmentFilter, setDepartmentFilter] = useState('All');
  const [courseFilter, setCourseFilter] = useState('All');
  const [statusFilter, setStatusFilter] = useState('All');
  const [records, setRecords] = useState<any[]>([]);

  // Load academia data if not already loaded
  useEffect(() => {
    if (!academiaLoading && faculties.length === 0) fetchData();
  }, [fetchData, faculties.length, academiaLoading]);

  useEffect(() => {
    setRecords(data?.data ?? []);
  }, [data]);

  // Filter options come from the academia store — all faculties/depts/courses regardless of attendance data
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

  // Handle cascaded resets
  const handleFacultyChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setFacultyFilter(e.target.value);
    setDepartmentFilter('All');
    setCourseFilter('All');
  };

  const handleDepartmentChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setDepartmentFilter(e.target.value);
    setCourseFilter('All');
  };

  const filteredRecords = records.filter(record => {
    const matchesSearch = record.studentName.toLowerCase().includes(searchTerm.toLowerCase()) || 
                          record.course.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFaculty = facultyFilter === 'All' || record.faculty === facultyFilter;
    const matchesDepartment = departmentFilter === 'All' || record.department === departmentFilter;
    const matchesCourse = courseFilter === 'All' || record.course === courseFilter;
    const matchesStatus = statusFilter === 'All' || record.status === statusFilter;
    return matchesSearch && matchesFaculty && matchesDepartment && matchesCourse && matchesStatus;
  });

  const tableContainerRef = useRef<HTMLDivElement>(null);
  
  const rowVirtualizer = useVirtualizer({
    count: filteredRecords.length,
    getScrollElement: () => tableContainerRef.current,
    estimateSize: () => 64,
    overscan: 5,
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
            View and manage student attendance records
          </p>
        </div>
      </div>

      {error ? (
        <div className="rounded-2xl border border-rose-200 dark:border-rose-500/20 bg-rose-50 dark:bg-rose-500/10 p-4 text-sm text-rose-700 dark:text-rose-200">
          Failed to load live attendance sessions.
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
              <select
                value={facultyFilter}
                onChange={handleFacultyChange}
                className="h-10 rounded-xl px-3 pr-8 text-sm text-gray-900 dark:text-gray-100 bg-white dark:bg-white/5 appearance-none cursor-pointer border border-gray-200 dark:border-white/10 focus:outline-none focus:ring-2 focus:ring-primary/50 shrink-0"
                style={{ backgroundImage: `url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%236B7280%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpolyline%20points%3D%226%209%2012%2015%2018%209%22%3E%3C%2Fpolyline%3E%3C%2Fsvg%3E")`, backgroundRepeat: 'no-repeat', backgroundPosition: 'right 0.5rem center', backgroundSize: '1em 1em' }}
              >
                <option value="All" className="bg-white dark:bg-dark-bg">All Faculties</option>
                {facultiesList.map((f) => (
                  <option key={f.id} value={f.name} className="bg-white dark:bg-dark-bg">{f.name}</option>
                ))}
              </select>
            </div>

            {/* ROW 2 — department, course, status filters */}
            <div className="flex items-center gap-2">
              <Filter className="text-gray-400 shrink-0" size={16} />
              {/* Department filter */}
              <select
                value={departmentFilter}
                onChange={handleDepartmentChange}
                className="h-10 rounded-xl px-3 pr-8 text-sm text-gray-900 dark:text-gray-100 bg-white dark:bg-white/5 appearance-none cursor-pointer border border-gray-200 dark:border-white/10 focus:outline-none focus:ring-2 focus:ring-primary/50 shrink-0"
                style={{ backgroundImage: `url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%236B7280%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpolyline%20points%3D%226%209%2012%2015%2018%209%22%3E%3C%2Fpolyline%3E%3C%2Fsvg%3E")`, backgroundRepeat: 'no-repeat', backgroundPosition: 'right 0.5rem center', backgroundSize: '1em 1em' }}
              >
                <option value="All" className="bg-white dark:bg-dark-bg">All Departments</option>
                {departmentsList.map((d) => (
                  <option key={d.id} value={d.name} className="bg-white dark:bg-dark-bg">{d.name}</option>
                ))}
              </select>

              {/* Course filter */}
              <select
                value={courseFilter}
                onChange={(e) => setCourseFilter(e.target.value)}
                className="h-10 rounded-xl px-3 pr-8 text-sm text-gray-900 dark:text-gray-100 bg-white dark:bg-white/5 appearance-none cursor-pointer border border-gray-200 dark:border-white/10 focus:outline-none focus:ring-2 focus:ring-primary/50 shrink-0"
                style={{ backgroundImage: `url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%236B7280%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpolyline%20points%3D%226%209%2012%2015%2018%209%22%3E%3C%2Fpolyline%3E%3C%2Fsvg%3E")`, backgroundRepeat: 'no-repeat', backgroundPosition: 'right 0.5rem center', backgroundSize: '1em 1em' }}
              >
                <option value="All" className="bg-white dark:bg-dark-bg">All Courses</option>
                {coursesList.map((c) => (
                  <option key={c.id} value={c.title} className="bg-white dark:bg-dark-bg">{c.title}</option>
                ))}
              </select>
              <select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                className="h-10 rounded-xl px-3 pr-8 text-sm text-gray-900 dark:text-gray-100 bg-white dark:bg-white/5 appearance-none cursor-pointer border border-gray-200 dark:border-white/10 focus:outline-none focus:ring-2 focus:ring-primary/50 shrink-0"
                style={{ backgroundImage: `url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%236B7280%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpolyline%20points%3D%226%209%2012%2015%2018%209%22%3E%3C%2Fpolyline%3E%3C%2Fsvg%3E")`, backgroundRepeat: 'no-repeat', backgroundPosition: 'right 0.5rem center', backgroundSize: '1em 1em' }}
              >
                <option value="All" className="bg-white dark:bg-dark-bg">All Statuses</option>
                <option value="Present" className="bg-white dark:bg-dark-bg">Present</option>
                <option value="Late" className="bg-white dark:bg-dark-bg">Late</option>
                <option value="Absent" className="bg-white dark:bg-dark-bg">Absent</option>
                <option value="Excused" className="bg-white dark:bg-dark-bg">Excused</option>
              </select>
            </div>
          </div>

          <div ref={tableContainerRef} className="overflow-auto custom-scrollbar rounded-xl border border-gray-100 dark:border-white/5 h-[500px] relative w-full">
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
              <TableBody
                style={{
                  height: `${rowVirtualizer.getTotalSize()}px`,
                  position: 'relative'
                }}
              >
                {isLoading ? (
                  <TableRow>
                    <TableCell colSpan={7} className="h-32 text-center text-gray-500">
                      Loading live attendance data...
                    </TableCell>
                  </TableRow>
                ) : filteredRecords.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={7} className="h-32 text-center text-gray-500">
                      No live attendance records found in the database.
                    </TableCell>
                  </TableRow>
                ) : (
                  (() => {
                    const virtualItems = rowVirtualizer.getVirtualItems();
                    const paddingTop = virtualItems.length > 0 ? virtualItems[0]?.start || 0 : 0;
                    const paddingBottom = virtualItems.length > 0
                      ? rowVirtualizer.getTotalSize() - (virtualItems[virtualItems.length - 1]?.end || 0)
                      : 0;

                    return (
                      <>
                        {paddingTop > 0 && (
                          <TableRow>
                            <TableCell style={{ height: `${paddingTop}px` }} colSpan={7} />
                          </TableRow>
                        )}
                        {virtualItems.map((virtualRow) => {
                          const index = virtualRow.index;
                          const record = filteredRecords[index];
                          return (
                            <TableRow key={record.id} style={{ height: `${virtualRow.size}px` }} className="hover:bg-gray-50/50 dark:hover:bg-white/[0.02]">
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
                                          timeStyle: 'short' 
                                        })}
                                      </span>
                                    </>
                                  ) : (
                                    <span className="pl-4">-</span>
                                  )}
                                </div>
                              </TableCell>
                            </TableRow>
                          );
                        })}
                        {paddingBottom > 0 && (
                          <TableRow>
                            <TableCell style={{ height: `${paddingBottom}px` }} colSpan={7} />
                          </TableRow>
                        )}
                      </>
                    );
                  })()
                )}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
