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
    fetchTeachers,
    fetchFaculties,
    fetchDepartments,
    isLoading,
  } = useHrStore();

  const [filterFaculty, setFilterFaculty] = useState("All");
  const [filterDepartment, setFilterDepartment] = useState("All");
  const [filterRole, setFilterRole] = useState("All");
  const [generateTriggered, setGenerateTriggered] = useState(false);

  useEffect(() => {
    fetchTeachers();
    fetchFaculties();
    fetchDepartments();
  }, [fetchTeachers, fetchFaculties, fetchDepartments]);

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
    queryKey: ["hr", "teacher-performance"],
    enabled: generateTriggered,
    queryFn: async () => {
      const fetchAllAttendanceRecords = async () => {
        const lim = 200;
        let pg = 1;
        let total = Number.POSITIVE_INFINITY;
        const records: any[] = [];
        while (records.length < total) {
          const response = await api.get("/attendance/records", { params: { page: pg, limit: lim } });
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
        api.get("/sessions", { params: { limit: 500 } })
          .then(r => Array.isArray(r.data) ? r.data : (r.data?.items ?? []))
          .catch(() => [] as any[]),
      ]);

      const assignments = Array.isArray(assignmentsResponse)
        ? assignmentsResponse
        : (assignmentsResponse?.items ?? assignmentsResponse?.data ?? []);

      // course_id → Set<teacher_id>
      const courseTeachers = new Map<string, Set<string>>();
      for (const a of assignments) {
        const courseId = String(a.course_id ?? a.courseId ?? "");
        const teacherId = String(a.teacher_id ?? a.teacherId ?? "");
        if (!courseId || !teacherId) continue;
        if (!courseTeachers.has(courseId)) courseTeachers.set(courseId, new Set());
        courseTeachers.get(courseId)!.add(teacherId);
      }

      // Attended: sessions that have ≥1 attendance record, mapped per teacher
      const attendedByTeacher = new Map<string, Set<string>>();
      for (const record of attendanceRecords) {
        const courseId = String(record.courseId ?? record.course_id ?? "");
        const sessionId = String(record.sessionId ?? record.session_id ?? "");
        if (!courseId || !sessionId) continue;
        const teachers = courseTeachers.get(courseId);
        if (!teachers) continue;
        teachers.forEach(tid => {
          if (!attendedByTeacher.has(tid)) attendedByTeacher.set(tid, new Set());
          attendedByTeacher.get(tid)!.add(sessionId);
        });
      }

      // Total: all sessions (started/ended) for each teacher's courses
      const totalByTeacher = new Map<string, Set<string>>();
      for (const session of sessionsData) {
        const courseId = String(session.course_id ?? session.courseId ?? "");
        const sessionId = String(session.id ?? "");
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

  const getPerformance = (id: string) => {
    const data = teacherPerformanceQuery.data?.[id];
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
    doc.text(`Generated: ${new Date().toLocaleDateString()}`, 14, 26);
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
                className="text-gray-900 dark:text-white dark:[color-scheme:dark]"
              />
            </div>
            <div className="flex-1 w-full sm:w-auto">
              <label className="block text-xs font-medium text-gray-500 dark:text-gray-400 mb-1.5 ml-1">
                End Date
              </label>
              <Input
                type="date"
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
