import { useEffect, useState, useRef, useMemo } from 'react';
import { useVirtualizer } from '@tanstack/react-virtual';
import { useQuery } from '@tanstack/react-query';
import { useNavigate } from 'react-router-dom';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { Users, GraduationCap, Building2, Activity, Search, FileText, Download, Printer } from 'lucide-react';
import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';

import { Card, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Select } from '@/components/ui/Select';
import { Input } from '@/components/ui/Input';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/Table';
import { Badge } from '@/components/ui/Badge';
import { StatCard } from '@/components/ui/StatCard';
import { ExportButtons } from '@/components/ui/ExportButtons';
import { useReportSummary, useAbsenceRanking, useAttendanceChartData, useDistributionSummary } from '@/hooks/queries/useReports';
import { api } from '@/services/api';
import { useHrStore } from '@/store/useHrStore';
import courseService from '@/services/courseService';

export default function SystemReports() {
  const navigate = useNavigate();
  const { data: summary, isLoading: isLoadingSummary } = useReportSummary();
  const [page, setPage] = useState(1);
  const [limit] = useState(10);
  const [reportType, setReportType] = useState('');
  // facultyId / departmentId store the numeric ID as a string for the <select> value.
  // '' means "All".
  const [facultyId, setFacultyId] = useState('');
  const [departmentId, setDepartmentId] = useState('');
  const [course, setCourse] = useState('all');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [searchQuery, setSearchQuery] = useState('');

  // Teacher report state
  const [teacherReportGenerated, setTeacherReportGenerated] = useState(false);
  const [teacherAppliedFacultyId, setTeacherAppliedFacultyId] = useState('');
  const [teacherAppliedDepartmentId, setTeacherAppliedDepartmentId] = useState('');

  // HR store for teacher report data
  const { teachers, faculties: hrFaculties, departments: hrDepartments, fetchTeachers, fetchFaculties, fetchDepartments, isLoading: hrLoading } = useHrStore();

  useEffect(() => {
    if (reportType === 'teacher_attendance') {
      fetchTeachers();
      fetchFaculties();
      fetchDepartments();
    }
  }, [reportType, fetchTeachers, fetchFaculties, fetchDepartments]);

  // Teacher performance query (same logic as HR Reports)
  const teacherPerformanceQuery = useQuery({
    queryKey: ['admin', 'teacher-performance'],
    enabled: reportType === 'teacher_attendance',
    queryFn: async () => {
      const fetchAllAttendanceRecords = async () => {
        const lim = 200;
        let pg = 1;
        let total = Number.POSITIVE_INFINITY;
        const records: any[] = [];
        while (records.length < total) {
          const response = await api.get('/attendance/records', { params: { page: pg, limit: lim } });
          const pageRecords = response.data?.data ?? [];
          records.push(...pageRecords);
          total = Number(response.data?.total ?? records.length);
          if (pageRecords.length < lim) break;
          pg += 1;
        }
        return records;
      };
      const [assignmentsResponse, attendanceRecords, sessionsData] = await Promise.all([
        courseService.listAssignments(),
        fetchAllAttendanceRecords(),
        api.get('/sessions', { params: { limit: 500 } })
          .then(r => Array.isArray(r.data) ? r.data : (r.data?.items ?? []))
          .catch(() => [] as any[]),
      ]);
      const assignments = Array.isArray(assignmentsResponse)
        ? assignmentsResponse
        : (assignmentsResponse?.items ?? assignmentsResponse?.data ?? []);

      const courseTeachers = new Map<string, Set<string>>();
      for (const assignment of assignments) {
        const courseId = String(assignment.course_id ?? assignment.courseId ?? '');
        const teacherId = String(assignment.teacher_id ?? assignment.teacherId ?? '');
        if (!courseId || !teacherId) continue;
        if (!courseTeachers.has(courseId)) courseTeachers.set(courseId, new Set());
        courseTeachers.get(courseId)!.add(teacherId);
      }

      const attendedByTeacher = new Map<string, Set<string>>();
      for (const record of attendanceRecords) {
        const courseId = String(record.courseId ?? record.course_id ?? '');
        const sessionId = String(record.sessionId ?? record.session_id ?? '');
        if (!courseId || !sessionId) continue;
        const teachers = courseTeachers.get(courseId);
        if (!teachers) continue;
        teachers.forEach(tid => {
          if (!attendedByTeacher.has(tid)) attendedByTeacher.set(tid, new Set());
          attendedByTeacher.get(tid)!.add(sessionId);
        });
      }

      const totalByTeacher = new Map<string, Set<string>>();
      for (const session of sessionsData) {
        const courseId = String(session.course_id ?? session.courseId ?? '');
        const sessionId = String(session.id ?? '');
        if (!courseId || !sessionId) continue;
        const teachers = courseTeachers.get(courseId);
        if (!teachers) continue;
        teachers.forEach(tid => {
          if (!totalByTeacher.has(tid)) totalByTeacher.set(tid, new Set());
          totalByTeacher.get(tid)!.add(sessionId);
        });
      }

      const allIds = new Set([...attendedByTeacher.keys(), ...totalByTeacher.keys()]);
      return Object.fromEntries(
        Array.from(allIds).map(tid => [
          tid,
          {
            attended: attendedByTeacher.get(tid)?.size ?? 0,
            total: totalByTeacher.get(tid)?.size ?? 0,
          },
        ])
      ) as Record<string, { attended: number; total: number }>;
    },
  });

  const getHrFacultyName = (id: string) => hrFaculties.find(f => f.id === id)?.name ?? id;
  const getHrDepartmentName = (id: string) => hrDepartments.find(d => d.id === id)?.name ?? id;
  const getPerformance = (id: string) => {
    const data = teacherPerformanceQuery.data?.[id];
    const attended = data?.attended ?? 0;
    const total = data?.total ?? 0;
    return `${attended}/${total}`;
  };

  const filteredTeachersForReport = useMemo(() => {
    return teachers.filter(t => {
      if (teacherAppliedFacultyId && t.facultyId !== teacherAppliedFacultyId) return false;
      if (teacherAppliedDepartmentId && t.departmentId !== teacherAppliedDepartmentId) return false;
      return true;
    });
  }, [teachers, teacherAppliedFacultyId, teacherAppliedDepartmentId]);

  const [appliedFilters, setAppliedFilters] = useState({
    search: '', type: '', faculty: 'all', department: 'all', course: 'all',
    startDate: '', endDate: '',
  });

  // --- Dynamic filter data ---
  // Faculties: fetch all (no server-side filter needed)
  const { data: facultiesRaw } = useQuery({
    queryKey: ['filterFaculties'],
    queryFn: () => api.get('/faculties', { params: { limit: 200 } }).then(r => r.data?.items ?? []),
    staleTime: 1000 * 60 * 5,
  });
  const facultiesList: any[] = facultiesRaw ?? [];

  // Departments: filtered by selected faculty_id on the server
  const { data: departmentsRaw } = useQuery({
    queryKey: ['filterDepartments', facultyId],
    queryFn: () =>
      api.get('/departments', {
        params: { limit: 200, ...(facultyId ? { faculty_id: facultyId } : {}) },
      }).then(r => r.data?.items ?? []),
    staleTime: 1000 * 60 * 5,
  });
  const departmentsList: any[] = departmentsRaw ?? [];

  // Courses: filtered by selected faculty_id on the server, then client-filtered by department_id
  const { data: coursesRaw } = useQuery({
    queryKey: ['filterCourses', facultyId],
    queryFn: () =>
      api.get('/courses', {
        params: { limit: 200, ...(facultyId ? { faculty_id: facultyId } : {}) },
      }).then(r => r.data?.items ?? []),
    staleTime: 1000 * 60 * 5,
  });
  const coursesForDept: any[] = (coursesRaw ?? []).filter(
    (c: any) => !departmentId || String(c.department_id) === departmentId,
  );

  const facultyOptions = [
    { value: '', label: 'All Faculties' },
    ...facultiesList.map((f: any) => ({ value: String(f.id), label: f.name })),
  ];

  const departmentOptions = [
    { value: '', label: 'All Departments' },
    ...departmentsList.map((d: any) => ({ value: String(d.id), label: d.name })),
  ];

  const courseOptions = [
    { value: 'all', label: 'All Courses' },
    ...coursesForDept.map((c: any) => ({ value: c.title ?? c.name, label: c.title ?? c.name })),
  ];

  useEffect(() => {
    const handler = setTimeout(() => {
      setAppliedFilters(prev => {
        if (prev.search !== searchQuery) {
          setPage(1);
          return { ...prev, search: searchQuery };
        }
        return prev;
      });
    }, 300);
    return () => clearTimeout(handler);
  }, [searchQuery]);

  const { data: absenceData, isLoading: isLoadingAbsence } = useAbsenceRanking({
    page,
    limit,
    search: appliedFilters.search,
    type: appliedFilters.type,
    faculty: appliedFilters.faculty,
    department: appliedFilters.department,
    course: appliedFilters.course,
    startDate: appliedFilters.startDate,
    endDate: appliedFilters.endDate,
  });
  const displayedRecords = absenceData?.data || [];
  const totalRecords = absenceData?.total || 0;

  const { data: chartData = [], isLoading: isLoadingChart } = useAttendanceChartData();
  const { data: distribution, isLoading: isLoadingDistribution } = useDistributionSummary();
  const isLoading = isLoadingSummary || isLoadingAbsence || isLoadingChart || isLoadingDistribution;
  const tableContainerRef = useRef<HTMLDivElement>(null);


  const rowVirtualizer = useVirtualizer({
    count: displayedRecords.length,
    getScrollElement: () => tableContainerRef.current,
    estimateSize: () => 53,
    overscan: 5,
  });

  const handleGenerateReport = () => {
    if (reportType === 'teacher_attendance') {
      setTeacherAppliedFacultyId(facultyId);
      setTeacherAppliedDepartmentId(departmentId);
      setTeacherReportGenerated(true);
      return;
    }
    // Resolve names from IDs — the backend absence-ranking filters by name
    const selectedFaculty = facultiesList.find((f: any) => String(f.id) === facultyId);
    const selectedDept = departmentsList.find((d: any) => String(d.id) === departmentId);
    setTeacherReportGenerated(false);
    setPage(1);
    setAppliedFilters(prev => ({
      ...prev,
      type: reportType,
      faculty: selectedFaculty?.name ?? 'all',
      department: selectedDept?.name ?? 'all',
      course,
      startDate,
      endDate,
    }));
  };

  const handleReset = () => {
    setReportType('');
    setFacultyId('');
    setDepartmentId('');
    setCourse('all');
    setStartDate('');
    setEndDate('');
    setSearchQuery('');
    setPage(1);
    setTeacherReportGenerated(false);
    setTeacherAppliedFacultyId('');
    setTeacherAppliedDepartmentId('');
    setAppliedFilters({
      search: '', type: '', faculty: 'all', department: 'all', course: 'all',
      startDate: '', endDate: '',
    });
  };

  const handleExportPDF = () => {
    const doc = new jsPDF();
    doc.setFontSize(16);
    doc.text('Absence Ranking Report', 14, 18);
    doc.setFontSize(10);
    doc.setTextColor(100);
    doc.text(`Generated: ${new Date().toLocaleDateString()}`, 14, 26);
    autoTable(doc, {
      head: [['#', 'Student Name', 'Type', 'Course', 'Total Absences', 'Attendance %', 'Status']],
      body: displayedRecords.map((r: any, i: number) => [
        i + 1,
        r.studentName,
        r.type,
        r.facultyOrDepartment,
        `${r.totalAbsences} days`,
        `${r.attendancePercentage}%`,
        getAttendanceLabel(r.attendancePercentage),
      ]),
      startY: 32,
      headStyles: { fillColor: [37, 99, 235] },
      styles: { fontSize: 9 },
    });
    doc.save('absence_ranking_report.pdf');
  };

  const handleExportCSV = () => {
    const rows = [
      ['#', 'Student Name', 'Type', 'Course', 'Total Absences', 'Attendance %', 'Status'],
      ...displayedRecords.map((r: any, i: number) => [
        i + 1,
        r.studentName,
        r.type,
        r.facultyOrDepartment,
        `${r.totalAbsences} days`,
        `${r.attendancePercentage}%`,
        getAttendanceLabel(r.attendancePercentage),
      ]),
    ];
    const csv = rows.map(r => r.map((c: any) => `"${String(c).replace(/"/g, '""')}"`).join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'absence_ranking_report.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  const handlePrint = () => {
    window.print();
  };

  // --- Teacher report export handlers ---
  const handleTeacherExportPDF = () => {
    const doc = new jsPDF();
    doc.setFontSize(16);
    doc.text('Teacher Attendance Report', 14, 18);
    doc.setFontSize(10);
    doc.setTextColor(100);
    doc.text(`Generated: ${new Date().toLocaleDateString()}`, 14, 26);
    autoTable(doc, {
      head: [['Name', 'Role', 'Faculty', 'Department', 'Performance', 'Status']],
      body: filteredTeachersForReport.map(t => [
        t.fullName,
        t.role,
        getHrFacultyName(t.facultyId),
        getHrDepartmentName(t.departmentId),
        getPerformance(t.id),
        t.status,
      ]),
      startY: 32,
      headStyles: { fillColor: [37, 99, 235] },
      styles: { fontSize: 9 },
    });
    doc.save('teacher_attendance_report.pdf');
  };

  const handleTeacherExportCSV = () => {
    const rows = [
      ['Name', 'Role', 'Faculty', 'Department', 'Performance', 'Status'],
      ...filteredTeachersForReport.map(t => [
        t.fullName,
        t.role,
        getHrFacultyName(t.facultyId),
        getHrDepartmentName(t.departmentId),
        getPerformance(t.id),
        t.status,
      ]),
    ];
    const csv = rows.map(r => r.map(c => `"${String(c).replace(/"/g, '""')}"`).join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'teacher_attendance_report.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleTeacherPrint = () => {
    window.print();
  };

  const handleDownloadTeacherSingle = (teacher: any) => {
    const doc = new jsPDF();
    doc.setFontSize(16);
    doc.text(`Teacher Report — ${teacher.fullName}`, 14, 18);
    doc.setFontSize(10);
    doc.setTextColor(100);
    doc.text(`Generated: ${new Date().toLocaleDateString()}`, 14, 26);
    autoTable(doc, {
      head: [["Field", "Value"]],
      body: [
        ["T-NO", teacher.teacherNumber || teacher.id],
        ["Name", teacher.fullName],
        ["Role", teacher.role],
        ["Faculty", getHrFacultyName(teacher.facultyId)],
        ["Department", getHrDepartmentName(teacher.departmentId)],
        ["Performance", getPerformance(teacher.id)],
        ["Status", teacher.status],
      ],
      startY: 32,
      headStyles: { fillColor: [37, 99, 235] },
      styles: { fontSize: 10 },
    });
    doc.save(`${teacher.fullName.replace(/\s+/g, "_")}_report.pdf`);
  };

  const getAttendanceLabel = (percentage: number) =>
    percentage < 50 ? 'Low' : percentage < 75 ? 'Normal' : 'Good';

  const handleDownloadSingle = (record: any) => {
    const doc = new jsPDF();
    doc.text(`Report - ${record.studentName}`, 14, 15);

    autoTable(doc, {
      head: [['Field', 'Value']],
      body: [
        ['Name', record.studentName],
        ['Type', record.type],
        ['Course', record.facultyOrDepartment],
        ['Total Absences', `${record.totalAbsences} days`],
        ['Attendance', `${record.attendancePercentage}%`],
        ['Status', getAttendanceLabel(record.attendancePercentage)]
      ],
      startY: 20,
    });

    doc.save(`${record.studentName.replace(/\s+/g, '_')}_report.pdf`);
  };

  // Badge based on attendance percentage:
  // <50% → red "Low", 50–74% → yellow "Normal", ≥75% → green "Good"
  const getAttendanceBadge = (percentage: number) => {
    if (percentage < 50) {
      return (
        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-700 dark:bg-red-500/10 dark:text-red-400">
          Low
        </span>
      );
    }
    if (percentage < 75) {
      return (
        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-700 dark:bg-yellow-500/10 dark:text-yellow-400">
          Normal
        </span>
      );
    }
    return (
      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-700 dark:bg-green-500/10 dark:text-green-400">
        Good
      </span>
    );
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col gap-1">
        <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-white">
          System Reports
        </h1>
        <p className="text-gray-500 dark:text-gray-400">
          Generate and analyze reports across the system
        </p>
      </div>

      {/* Filter Bar */}
      <Card className="glass-card">
        <CardContent className="p-4 sm:p-6 flex flex-col gap-4">
          <div className={`grid grid-cols-1 gap-4 w-full ${reportType === 'teacher_attendance' ? 'md:grid-cols-3' : 'md:grid-cols-4'}`}>
            <div>
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1.5 ml-1">Report Type</label>
              <Select
                options={[
                  { value: '', label: 'Please Choose Report Type' },
                  { value: 'student_attendance', label: 'Student Attendance Report' },
                  { value: 'teacher_attendance', label: 'Teacher Attendance Report' },
                ]}
                value={reportType}
                onChange={(e) => { setReportType(e.target.value); setTeacherReportGenerated(false); }}
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1.5 ml-1">Faculty</label>
              <Select
                options={facultyOptions}
                value={facultyId}
                onChange={(e) => { setFacultyId(e.target.value); setDepartmentId(''); setCourse('all'); }}
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1.5 ml-1">Department</label>
              <Select
                options={departmentOptions}
                value={departmentId}
                onChange={(e) => { setDepartmentId(e.target.value); setCourse('all'); }}
              />
            </div>
            {reportType !== 'teacher_attendance' && (
              <div>
                <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1.5 ml-1">Course</label>
                <Select
                  options={courseOptions}
                  value={course}
                  onChange={(e) => setCourse(e.target.value)}
                />
              </div>
            )}
          </div>
          <div className="flex flex-col sm:flex-row items-end gap-4 w-full pt-4 border-t border-gray-100 dark:border-white/5">
            <div className="flex-1 w-full sm:w-auto">
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1.5 ml-1">Start Date</label>
              <Input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} className="text-gray-900 dark:text-white dark:[color-scheme:dark]" />
            </div>
            <div className="flex-1 w-full sm:w-auto">
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1.5 ml-1">End Date</label>
              <Input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} className="text-gray-900 dark:text-white dark:[color-scheme:dark]" />
            </div>
            <div className="flex items-center justify-end gap-3 w-full sm:w-auto">
              <Button variant="ghost" className="w-full sm:w-auto" onClick={handleReset}>Reset</Button>
              <Button className="w-full sm:w-auto whitespace-nowrap" onClick={handleGenerateReport}>Generate Report</Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Summary KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Total Students"
          value={isLoading ? '-' : `+${summary?.totalStudents?.toLocaleString()}`}
          icon={Users}
          iconColor="primary"
          onClick={() => navigate('/admin/students')}
        />
        <StatCard
          title="Total Teachers"
          value={isLoading ? '-' : summary?.totalTeachers || 0}
          icon={GraduationCap}
          iconColor="success"
          onClick={() => navigate('/admin/teachers')}
        />
        <StatCard
          title="Total Faculties"
          value={isLoading ? '-' : summary?.totalFaculties || 0}
          icon={Building2}
          iconColor="warning"
          onClick={() => navigate('/admin/faculties')}
        />
        <StatCard
          title="Attendance Rate"
          value={isLoading ? '-' : `${summary?.attendanceRate}%`}
          icon={Activity}
          iconColor="primary"
          onClick={() => navigate('/admin/attendance-list')}
        />
      </div>

      {/* Analytics Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="glass-card lg:col-span-2">
          <CardContent className="p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">Attendance Overview</h3>
            <div className="h-[300px] w-full mt-4">
              {isLoading ? (
                <div className="w-full h-full bg-gray-200 dark:bg-white/5 animate-pulse rounded-xl" />
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                    <XAxis dataKey="name" stroke="#6B7280" tick={{ fill: '#6B7280', fontSize: 12 }} axisLine={false} tickLine={false} />
                    <YAxis stroke="#6B7280" tick={{ fill: '#6B7280', fontSize: 12 }} axisLine={false} tickLine={false} tickFormatter={(val) => `${val}%`} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1F2937', borderColor: '#374151', borderRadius: '8px', color: '#fff' }}
                      itemStyle={{ color: '#60A5FA' }}
                    />
                    <Line type="monotone" dataKey="value" stroke="#3B82F6" strokeWidth={3} dot={{ r: 4, fill: '#3B82F6', strokeWidth: 2, stroke: '#0B0F19' }} activeDot={{ r: 6 }} />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </CardContent>
        </Card>

        <Card className="glass-card">
          <CardContent className="p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-6">Distribution Summary</h3>
            <div className="grid grid-cols-3 gap-2 mb-8 border-b border-gray-200 dark:border-white/5 pb-6">
              <div>
                <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">Students</p>
                <div className="flex items-center gap-2">
                  <div className="w-1.5 h-4 bg-primary rounded-full"></div>
                  <span className="text-xl font-bold text-gray-900 dark:text-white">{distribution?.students}%</span>
                </div>
              </div>
              <div>
                <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">Teachers</p>
                <div className="flex items-center gap-2">
                  <div className="w-1.5 h-4 bg-emerald-500 rounded-full"></div>
                  <span className="text-xl font-bold text-gray-900 dark:text-white">{distribution?.teachers}</span>
                </div>
              </div>
              <div>
                <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">Faculties</p>
                <div className="flex items-center gap-2">
                  <div className="w-1.5 h-4 bg-amber-500 rounded-full"></div>
                  <span className="text-xl font-bold text-gray-900 dark:text-white">{distribution?.faculties}</span>
                </div>
              </div>
            </div>

            <div className="h-[180px] w-full">
              {isLoading ? (
                <div className="w-full h-full bg-gray-200 dark:bg-white/5 animate-pulse rounded-xl" />
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={[
                    { name: 'Students', value: distribution?.students || 0, fill: '#3B82F6' },
                    { name: 'Teachers', value: distribution?.teachers || 0, fill: '#10B981' },
                    { name: 'Faculties', value: distribution?.faculties || 0, fill: '#F59E0B' }
                  ]} margin={{ top: 0, right: 0, bottom: 0, left: -20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                    <XAxis dataKey="name" stroke="#6B7280" tick={{ fill: '#6B7280', fontSize: 11 }} axisLine={false} tickLine={false} />
                    <YAxis stroke="#6B7280" tick={{ fill: '#6B7280', fontSize: 11 }} axisLine={false} tickLine={false} />
                    <Tooltip cursor={{ fill: 'rgba(255,255,255,0.05)' }} contentStyle={{ backgroundColor: '#1F2937', borderColor: '#374151', borderRadius: '8px', color: '#fff' }} />
                    <Bar dataKey="value" radius={[4, 4, 0, 0]} barSize={30} />
                  </BarChart>
                </ResponsiveContainer>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Teacher Attendance Report Table */}
      {teacherReportGenerated && (
        <Card className="glass-card">
          <div className="p-4 sm:p-6 border-b border-gray-100 dark:border-white/5 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
            <div className="flex items-center gap-2">
              <GraduationCap className="text-primary" size={24} />
              <h2 className="text-xl font-semibold text-gray-900 dark:text-white">Teacher Attendance Report</h2>
              <span className="text-sm text-gray-500 dark:text-gray-400 ml-2">({filteredTeachersForReport.length} teachers)</span>
            </div>
            <ExportButtons
              onExportPDF={handleTeacherExportPDF}
              onExportCSV={handleTeacherExportCSV}
              onPrint={handleTeacherPrint}
            />
          </div>
          <CardContent className="p-0">
            <div className="overflow-auto custom-scrollbar">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>No.</TableHead>
                    <TableHead>Name</TableHead>
                    <TableHead>Role</TableHead>
                    <TableHead>Faculty</TableHead>
                    <TableHead>Department</TableHead>
                    <TableHead>Performance</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead className="text-right">Download</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {(hrLoading || teacherPerformanceQuery.isLoading) ? (
                    Array.from({ length: 5 }).map((_, i) => (
                      <TableRow key={`tskel-${i}`}>
                        <TableCell><div className="h-4 w-6 bg-gray-200 dark:bg-white/10 rounded animate-pulse" /></TableCell>
                        <TableCell><div className="h-4 w-32 bg-gray-200 dark:bg-white/10 rounded animate-pulse" /></TableCell>
                        <TableCell><div className="h-4 w-24 bg-gray-200 dark:bg-white/10 rounded animate-pulse" /></TableCell>
                        <TableCell><div className="h-4 w-24 bg-gray-200 dark:bg-white/10 rounded animate-pulse" /></TableCell>
                        <TableCell><div className="h-4 w-32 bg-gray-200 dark:bg-white/10 rounded animate-pulse" /></TableCell>
                        <TableCell><div className="h-4 w-20 bg-gray-200 dark:bg-white/10 rounded animate-pulse" /></TableCell>
                        <TableCell><div className="h-6 w-16 bg-gray-200 dark:bg-white/10 rounded-full animate-pulse" /></TableCell>
                        <TableCell><div className="h-8 w-8 ml-auto bg-gray-200 dark:bg-white/10 rounded animate-pulse" /></TableCell>
                      </TableRow>
                    ))
                  ) : filteredTeachersForReport.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={8} className="h-24 text-center text-gray-500">
                        No teachers found matching the selected filters.
                      </TableCell>
                    </TableRow>
                  ) : (
                    filteredTeachersForReport.map((teacher, index) => (
                      <TableRow key={teacher.id}>
                        <TableCell className="text-gray-500">{index + 1}</TableCell>
                        <TableCell className="font-medium text-gray-900 dark:text-white">{teacher.fullName}</TableCell>
                        <TableCell className="text-gray-600 dark:text-gray-300">{teacher.role}</TableCell>
                        <TableCell className="text-gray-600 dark:text-gray-300">{getHrFacultyName(teacher.facultyId)}</TableCell>
                        <TableCell className="text-gray-600 dark:text-gray-300">{getHrDepartmentName(teacher.departmentId)}</TableCell>
                        <TableCell className="font-medium text-gray-900 dark:text-white">{getPerformance(teacher.id)}</TableCell>
                        <TableCell>
                          <Badge
                            variant={
                              teacher.status === 'Active' ? 'success'
                              : teacher.status === 'On Leave' ? 'warning'
                              : 'danger'
                            }
                          >
                            {teacher.status}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-right">
                          <Button
                            variant="ghost"
                            size="sm"
                            className="text-primary hover:text-primary-600 hover:bg-primary/10"
                            onClick={() => handleDownloadTeacherSingle(teacher)}
                            title="Download PDF"
                          >
                            <Download size={16} />
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Main Table: Absence Ranking — hidden for Teacher Attendance Report */}
      {reportType !== 'teacher_attendance' && <Card className="glass-card">
        <div className="p-4 sm:p-6 border-b border-gray-100 dark:border-white/5 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
          <div className="flex items-center gap-2">
            <FileText className="text-primary" size={24} />
            <h2 className="text-xl font-semibold text-gray-900 dark:text-white">Absence Ranking</h2>
          </div>
          <div className="flex items-center gap-4 w-full sm:w-auto">
            <div className="relative w-full sm:w-64">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" size={16} />
              <Input 
                placeholder="Search records..." 
                className="pl-9 h-10 border-gray-200 bg-white dark:bg-white/5 dark:border-white/10" 
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
            <ExportButtons 
              className="hidden md:flex" 
              onExportPDF={handleExportPDF} 
              onExportCSV={handleExportCSV} 
              onPrint={handlePrint} 
            />
          </div>
        </div>
        <CardContent className="p-0">
           <ExportButtons 
             className="flex md:hidden p-4 border-b border-gray-200 dark:border-white/5 justify-end" 
             onExportPDF={handleExportPDF} 
             onExportCSV={handleExportCSV} 
             onPrint={handlePrint} 
           />
          <div ref={tableContainerRef} className="overflow-auto custom-scrollbar h-[400px] relative w-full">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>No.</TableHead>
                  <TableHead>Student Name</TableHead>
                  <TableHead>Type</TableHead>
                  <TableHead>Course</TableHead>
                  <TableHead>Total Absences</TableHead>
                  <TableHead>Attendance %</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody
                style={{
                  height: `${rowVirtualizer.getTotalSize()}px`,
                  position: 'relative'
                }}
              >
                {isLoading ? (
                  Array.from({ length: 5 }).map((_, i) => (
                    <TableRow key={`skeleton-row-${i}`}>
                      <TableCell><div className="h-4 w-6 bg-gray-200 dark:bg-white/10 rounded animate-pulse" /></TableCell>
                      <TableCell><div className="h-4 w-24 bg-gray-200 dark:bg-white/10 rounded animate-pulse" /></TableCell>
                      <TableCell><div className="h-4 w-16 bg-gray-200 dark:bg-white/10 rounded animate-pulse" /></TableCell>
                      <TableCell><div className="h-4 w-32 bg-gray-200 dark:bg-white/10 rounded animate-pulse" /></TableCell>
                      <TableCell><div className="h-4 w-12 bg-gray-200 dark:bg-white/10 rounded animate-pulse" /></TableCell>
                      <TableCell><div className="h-4 w-12 bg-gray-200 dark:bg-white/10 rounded animate-pulse" /></TableCell>
                      <TableCell><div className="h-6 w-16 bg-gray-200 dark:bg-white/10 rounded-full animate-pulse" /></TableCell>
                      <TableCell className="text-right"><div className="h-8 w-8 ml-auto bg-gray-200 dark:bg-white/10 rounded animate-pulse" /></TableCell>
                    </TableRow>
                  ))
                ) : displayedRecords.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={8} className="h-24 text-center text-gray-500">
                      No records found.
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
                            <TableCell style={{ height: `${paddingTop}px` }} colSpan={8} />
                          </TableRow>
                        )}
                        {virtualItems.map((virtualRow) => {
                          const index = virtualRow.index;
                          const record = displayedRecords[index];
                          return (
                            <TableRow key={record.id} style={{ height: `${virtualRow.size}px` }}>
                              <TableCell className="text-gray-500">{index + 1}</TableCell>
                              <TableCell className="font-medium text-gray-900 dark:text-white">
                                {record.studentName}
                              </TableCell>
                              <TableCell className="text-gray-500 dark:text-gray-400">
                                {record.type}
                              </TableCell>
                              <TableCell className="text-gray-500 dark:text-gray-400">
                                {record.facultyOrDepartment}
                              </TableCell>
                              <TableCell>
                                <span className="font-bold text-gray-900 dark:text-white">{record.totalAbsences} days</span>
                              </TableCell>
                              <TableCell className="text-gray-500 dark:text-gray-400">
                                {record.attendancePercentage}%
                              </TableCell>
                              <TableCell>
                                {getAttendanceBadge(record.attendancePercentage)}
                              </TableCell>
                              <TableCell className="text-right">
                                <Button 
                                  variant="ghost" 
                                  size="sm" 
                                  className="text-primary hover:text-primary-600 hover:bg-primary-50"
                                  onClick={() => handleDownloadSingle(record)}
                                >
                                  <Download size={16} />
                                </Button>
                              </TableCell>
                            </TableRow>
                          );
                        })}
                        {paddingBottom > 0 && (
                          <TableRow>
                            <TableCell style={{ height: `${paddingBottom}px` }} colSpan={8} />
                          </TableRow>
                        )}
                      </>
                    );
                  })()
                )}
              </TableBody>
            </Table>
          </div>
          <div className="p-4 border-t border-gray-100 dark:border-white/5 flex items-center justify-between">
             <span className="text-sm text-gray-500">Showing {displayedRecords.length} of {totalRecords} records</span>
             <div className="flex gap-2">
               <Button variant="secondary" size="sm" onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page === 1}>Previous</Button>
               <Button variant="secondary" size="sm" onClick={() => setPage(p => p + 1)} disabled={displayedRecords.length < limit}>Next</Button>
             </div>
          </div>
        </CardContent>
      </Card>}
    </div>
  );
}
