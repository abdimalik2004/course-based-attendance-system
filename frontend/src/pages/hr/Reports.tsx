import { useState, useEffect } from "react";
import { Filter, FileText, Download } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { useHrStore } from "@/store/useHrStore";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Card, CardContent } from "@/components/ui/Card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/Table";
import { Badge } from "@/components/ui/Badge";
import { api } from "@/services/api";
import courseService from "@/services/courseService";
import jsPDF from "jspdf";
import autoTable from "jspdf-autotable";

export default function Reports() {
  const {
    teachers,
    faculties,
    departments,
    fetchAll,
    isLoading,
  } = useHrStore();

  const [filterFaculty, setFilterFaculty] = useState("All");
  const [filterDepartment, setFilterDepartment] = useState("All");
  const [filterRole, setFilterRole] = useState("All");
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [generateTriggered, setGenerateTriggered] = useState(false);

  useEffect(() => {
    fetchAll();
  }, [fetchAll]);

  const getFacultyName = (id: string) =>
    faculties.find((f) => f.id === id)?.name || id;
  const getDepartmentName = (id: string) =>
    departments.find((d) => d.id === id)?.name || id;

  const filteredTeachers = teachers.filter((t) => {
    if (filterFaculty !== "All" && t.facultyId !== filterFaculty) return false;
    if (filterDepartment !== "All" && t.departmentId !== filterDepartment)
      return false;
    if (filterRole !== "All" && t.role !== filterRole) return false;
    return true;
  });

  const availableDepartments =
    filterFaculty === "All"
      ? departments
      : departments.filter((d) => d.facultyId === filterFaculty);

  const roles = Array.from(new Set(teachers.map((teacher) => teacher.role)));

  const teacherPerformanceQuery = useQuery({
    queryKey: ["hr", "teacher-performance", startDate, endDate],
    enabled: generateTriggered,
    queryFn: async () => {
      const fetchAllSessions = async () => {
        const lim = 200;
        let sk = 0;
        const results: any[] = [];
        while (true) {
          const r = await api.get("/sessions", { params: { limit: lim, skip: sk } });
          const page: any[] = Array.isArray(r.data) ? r.data : (r.data?.items ?? r.data?.data ?? []);
          results.push(...page);
          if (page.length < lim) break;
          sk += lim;
        }
        return results;
      };

      const [assignmentsResponse, sessionsData] = await Promise.all([
        courseService.listAssignments(),
        fetchAllSessions().catch(() => [] as any[]),
      ]);

      const assignments = Array.isArray(assignmentsResponse)
        ? assignmentsResponse
        : (assignmentsResponse?.items ?? assignmentsResponse?.data ?? []);

      // Apply date range filter to sessions if dates are set.
      // Sessions have a session_date field (ISO date string, e.g. "2025-01-15").
      const filteredSessions = sessionsData.filter((session: any) => {
        const sessionDate: string | null = session.session_date ?? null;
        if (!sessionDate) return true; // keep sessions with no date
        if (startDate && sessionDate < startDate) return false;
        if (endDate && sessionDate > endDate) return false;
        return true;
      });

      // course_id → Set<teacher_id>
      const courseTeachers = new Map<string, Set<string>>();
      for (const a of assignments) {
        const courseId = String(a.course_id ?? a.courseId ?? "");
        const teacherId = String(a.teacher_id ?? a.teacherId ?? "");
        if (!courseId || !teacherId) continue;
        if (!courseTeachers.has(courseId)) courseTeachers.set(courseId, new Set());
        courseTeachers.get(courseId)!.add(teacherId);
      }

      // For each session:
      //   total     → every session ever started for any course the teacher is assigned to
      //   attended  → only sessions where session.teacher_id matches THIS teacher
      //               (i.e. the teacher personally started it; admin sessions are excluded)
      const statsByTeacher = new Map<string, { total: Set<string>; teacherStarted: Set<string> }>();

      for (const session of filteredSessions) {
        const courseId = String(session.course_id ?? session.courseId ?? "");
        const sessionId = String(session.id ?? "");
        const sessionTeacherId = session.teacher_id != null ? String(session.teacher_id) : null;
        if (!courseId || !sessionId) continue;

        const teachers = courseTeachers.get(courseId);
        if (!teachers) continue;

        teachers.forEach(tid => {
          if (!statsByTeacher.has(tid)) {
            statsByTeacher.set(tid, { total: new Set(), teacherStarted: new Set() });
          }
          const stats = statsByTeacher.get(tid)!;
          // Every session for this course counts toward the teacher's total
          stats.total.add(sessionId);
          // Only count as teacher's own session if they personally started it
          if (sessionTeacherId === tid) {
            stats.teacherStarted.add(sessionId);
          }
        });
      }

      return Object.fromEntries(
        Array.from(statsByTeacher.entries()).map(([tid, stats]) => [
          tid,
          {
            attended: stats.teacherStarted.size,
            total: stats.total.size,
          },
        ])
      ) as Record<string, { attended: number; total: number }>;
    },
  });

  const getPerformance = (id: string) => {
    if (!generateTriggered || !teacherPerformanceQuery.data) return "—";
    const data = teacherPerformanceQuery.data[id];
    const attended = data?.attended ?? 0;
    const total = data?.total ?? 0;
    return `${attended}/${total}`;
  };

  const handleGenerateReport = () => {
    setGenerateTriggered(true);
  };

  const handleReset = () => {
    setFilterFaculty("All");
    setFilterDepartment("All");
    setFilterRole("All");
    setStartDate("");
    setEndDate("");
    setGenerateTriggered(false);
  };

  const buildTableRows = () =>
    filteredTeachers.map((t) => [
      t.fullName,
      t.role,
      getFacultyName(t.facultyId),
      getDepartmentName(t.departmentId),
      getPerformance(t.id),
      t.status,
    ]);

  const handleExportPDF = () => {
    const doc = new jsPDF();
    doc.setFontSize(16);
    doc.text("HR Teacher Attendance Report", 14, 18);
    doc.setFontSize(10);
    doc.setTextColor(100);
    const dateRangeLabel =
      startDate && endDate
        ? `Period: ${startDate} — ${endDate}`
        : startDate
        ? `From: ${startDate}`
        : endDate
        ? `Until: ${endDate}`
        : "All time";
    doc.text(`Generated: ${new Date().toLocaleDateString()}  |  ${dateRangeLabel}`, 14, 26);
    autoTable(doc, {
      head: [["Name", "Role", "Faculty", "Department", "Performance", "Status"]],
      body: buildTableRows(),
      startY: 32,
      headStyles: { fillColor: [37, 99, 235] },
      styles: { fontSize: 9 },
    });
    doc.save("hr_teacher_report.pdf");
  };

  const handleDownloadSingle = (teacher: any) => {
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
        ["Faculty", getFacultyName(teacher.facultyId)],
        ["Department", getDepartmentName(teacher.departmentId)],
        ["Performance", getPerformance(teacher.id)],
        ["Status", teacher.status],
      ],
      startY: 32,
      headStyles: { fillColor: [37, 99, 235] },
      styles: { fontSize: 10 },
    });
    doc.save(`${teacher.fullName.replace(/\s+/g, "_")}_report.pdf`);
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            HR Reports
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Generate and export staff reports.
          </p>
        </div>
      </div>

      <Card className="glass-card shadow-2xl shadow-primary/5">
        {/* Filters Section */}
        <div className="p-4 border-b border-gray-200 dark:border-white/10 bg-gray-50/50 dark:bg-white/5 space-y-4">
          <div className="flex items-center gap-2 mb-2 text-sm font-medium text-gray-700 dark:text-gray-300">
            <Filter size={16} /> Filters
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 w-full">
            {/* Faculty Filter */}
            <div>
              <label className="block text-xs text-gray-500 mb-1">
                Faculty
              </label>
              <select
                value={filterFaculty}
                onChange={(e) => {
                  setFilterFaculty(e.target.value);
                  setFilterDepartment("All"); // Reset department on faculty change
                }}
                className="w-full rounded-xl border border-gray-300 dark:border-white/10 bg-white dark:bg-dark-card px-3 py-2 text-sm text-gray-900 dark:text-white focus:border-primary focus:ring-2 focus:ring-primary/20 outline-none"
              >
                <option value="All">All Faculties</option>
                {faculties.map((f) => (
                  <option key={f.id} value={f.id}>
                    {f.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Department Filter */}
            <div>
              <label className="block text-xs text-gray-500 mb-1">
                Department
              </label>
              <select
                value={filterDepartment}
                onChange={(e) => setFilterDepartment(e.target.value)}
                className="w-full rounded-xl border border-gray-300 dark:border-white/10 bg-white dark:bg-dark-card px-3 py-2 text-sm text-gray-900 dark:text-white focus:border-primary focus:ring-2 focus:ring-primary/20 outline-none"
                disabled={
                  filterFaculty !== "All" && availableDepartments.length === 0
                }
              >
                <option value="All">All Departments</option>
                {availableDepartments.map((d) => (
                  <option key={d.id} value={d.id}>
                    {d.name}
                  </option>
                ))}
              </select>
            </div>

            {/* Role Filter */}
            <div>
              <label className="block text-xs text-gray-500 mb-1">Role</label>
              <select
                value={filterRole}
                onChange={(e) => setFilterRole(e.target.value)}
                className="w-full rounded-xl border border-gray-300 dark:border-white/10 bg-white dark:bg-dark-card px-3 py-2 text-sm text-gray-900 dark:text-white focus:border-primary focus:ring-2 focus:ring-primary/20 outline-none"
              >
                <option value="All">All Roles</option>
                {roles.map((r) => (
                  <option key={r} value={r}>
                    {r}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="flex flex-col sm:flex-row items-end gap-4 w-full pt-4 border-t border-gray-100 dark:border-white/5">
            <div className="flex-1 w-full sm:w-auto">
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1.5 ml-1">
                Start Date
              </label>
              <Input
                type="date"
                value={startDate}
                onChange={(e) => {
                  setStartDate(e.target.value);
                  setGenerateTriggered(false); // require re-generate when dates change
                }}
                className="text-gray-900 dark:text-white dark:[color-scheme:dark]"
              />
            </div>
            <div className="flex-1 w-full sm:w-auto">
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1.5 ml-1">
                End Date
              </label>
              <Input
                type="date"
                value={endDate}
                min={startDate || undefined}
                onChange={(e) => {
                  setEndDate(e.target.value);
                  setGenerateTriggered(false); // require re-generate when dates change
                }}
                className="text-gray-900 dark:text-white dark:[color-scheme:dark]"
              />
            </div>
            <div className="flex items-center justify-end gap-3 w-full sm:w-auto">
              <Button variant="ghost" className="w-full sm:w-auto" onClick={handleReset}>
                Reset
              </Button>
              {generateTriggered && (
                <Button
                  variant="secondary"
                  className="w-full sm:w-auto whitespace-nowrap"
                  onClick={handleExportPDF}
                  isLoading={teacherPerformanceQuery.isLoading}
                >
                  <Download size={16} className="mr-2" /> Export PDF
                </Button>
              )}
              <Button
                className="w-full sm:w-auto whitespace-nowrap"
                onClick={handleGenerateReport}
              >
                <FileText size={16} className="mr-2" /> Generate Report
              </Button>
            </div>
          </div>
        </div>

        {/* Table */}
        <CardContent className="p-0">
          <div className="overflow-x-auto custom-scrollbar">
            <Table>
              <TableHeader>
                <TableRow>
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
                {isLoading ? (
                  Array.from({ length: 5 }).map((_, i) => (
                    <TableRow key={`skeleton-${i}`}>
                      <TableCell><div className="h-4 w-32 bg-gray-200 dark:bg-white/10 rounded animate-pulse" /></TableCell>
                      <TableCell><div className="h-4 w-24 bg-gray-200 dark:bg-white/10 rounded animate-pulse" /></TableCell>
                      <TableCell><div className="h-4 w-24 bg-gray-200 dark:bg-white/10 rounded animate-pulse" /></TableCell>
                      <TableCell><div className="h-4 w-32 bg-gray-200 dark:bg-white/10 rounded animate-pulse" /></TableCell>
                      <TableCell><div className="h-4 w-12 bg-gray-200 dark:bg-white/10 rounded animate-pulse" /></TableCell>
                      <TableCell><div className="h-6 w-20 bg-gray-200 dark:bg-white/10 rounded-md animate-pulse" /></TableCell>
                      <TableCell><div className="h-8 w-8 ml-auto bg-gray-200 dark:bg-white/10 rounded animate-pulse" /></TableCell>
                    </TableRow>
                  ))
                ) : filteredTeachers.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={7} className="h-24 text-center text-gray-500">
                      <FileText size={32} className="mx-auto mb-3 opacity-20" />
                      No data matching your filters.
                    </TableCell>
                  </TableRow>
                ) : (
                  filteredTeachers.map((teacher) => (
                    <TableRow key={teacher.id}>
                      <TableCell className="font-medium text-gray-900 dark:text-white">
                        {teacher.fullName}
                      </TableCell>
                      <TableCell className="text-gray-600 dark:text-gray-300">
                        {teacher.role}
                      </TableCell>
                      <TableCell className="text-gray-600 dark:text-gray-300">
                        {getFacultyName(teacher.facultyId)}
                      </TableCell>
                      <TableCell className="text-gray-600 dark:text-gray-300">
                        {getDepartmentName(teacher.departmentId)}
                      </TableCell>
                      <TableCell className="font-medium text-gray-900 dark:text-white">
                        {getPerformance(teacher.id)}
                      </TableCell>
                      <TableCell>
                        <Badge
                          variant={
                            teacher.status === "Active"
                              ? "success"
                              : teacher.status === "On Leave"
                                ? "warning"
                                : "danger"
                          }
                        >
                          {teacher.status}
                        </Badge>
                      </TableCell>
                      <TableCell className="text-right">
                        <button
                          onClick={() => handleDownloadSingle(teacher)}
                          title="Download PDF"
                          className="p-1.5 rounded-lg text-primary hover:text-primary-600 hover:bg-primary/10 transition-colors"
                        >
                          <Download size={16} />
                        </button>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
