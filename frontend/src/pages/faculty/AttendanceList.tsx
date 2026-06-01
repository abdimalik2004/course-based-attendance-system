import { useState, useMemo, useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import {
  Search,
  Filter,
  Calendar as CalendarIcon,
  ShieldCheck,
  AlertCircle,
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/Card";
import { Input } from "@/components/ui/Input";
import { Button } from "@/components/ui/Button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/Table";
import { Badge } from "@/components/ui/Badge";
import { Modal } from "@/components/ui/Modal";
import {
  useAttendanceList,
  attendanceKeys,
} from "@/hooks/queries/useAttendance";
import { attendanceService } from "@/services/attendanceService";

export default function AttendanceList() {
  const queryClient = useQueryClient();
  const attendanceParams = { page: 1, limit: 200 };
  const { data, isLoading, error } = useAttendanceList(attendanceParams);
  const [records, setRecords] = useState<any[]>([]);
  const [searchTerm, setSearchTerm] = useState("");
  const [departmentFilter, setDepartmentFilter] = useState("All");
  const [courseFilter, setCourseFilter] = useState("All");
  const [statusFilter, setStatusFilter] = useState("All");
  const [isSaving, setIsSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [selectedRecord, setSelectedRecord] = useState<any>(null);

  useEffect(() => {
    setRecords(data?.data ?? []);
  }, [data]);

  const handleEditClick = (record: any) => {
    setSelectedRecord(record);
    setSaveError(null);
    setIsEditModalOpen(true);
  };

  const handleStatusUpdate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedRecord) return;

    setIsSaving(true);
    setSaveError(null);
    try {
      await attendanceService.updateAttendanceStatus(
        selectedRecord.id,
        "EXCUSED",
      );
      setRecords(
        records.map((r) =>
          r.id === selectedRecord.id ? { ...r, status: "Excused" } : r,
        ),
      );
      queryClient.invalidateQueries(attendanceKeys.all);
      setIsEditModalOpen(false);
      setSelectedRecord(null);
    } catch (err: any) {
      const detail =
        err?.response?.data?.detail ||
        err?.message ||
        "Failed to update attendance status. Please try again.";
      setSaveError(typeof detail === "string" ? detail : JSON.stringify(detail));
      console.error("Failed to update attendance status", err);
    } finally {
      setIsSaving(false);
    }
  };

  const uniqueDepartments = useMemo(
    () => Array.from(new Set(records.map((r) => r.department))),
    [records],
  );

  const uniqueCourses = useMemo(() => {
    return Array.from(
      new Set(
        records
          .filter(
            (r) =>
              departmentFilter === "All" || r.department === departmentFilter,
          )
          .map((r) => r.course),
      ),
    );
  }, [departmentFilter, records]);

  const handleDepartmentChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setDepartmentFilter(e.target.value);
    setCourseFilter("All"); // Reset course filter when department changes
  };

  const filteredRecords = records.filter((record) => {
    const matchesSearch =
      record.studentName.toLowerCase().includes(searchTerm.toLowerCase()) ||
      record.course.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesDepartment =
      departmentFilter === "All" || record.department === departmentFilter;
    const matchesCourse =
      courseFilter === "All" || record.course === courseFilter;
    const matchesStatus =
      statusFilter === "All" || record.status === statusFilter;
    return matchesSearch && matchesDepartment && matchesCourse && matchesStatus;
  });

  const getStatusBadgeVariant = (status: string) => {
    switch (status) {
      case "Present":
        return "success";
      case "Late":
        return "warning";
      case "Absent":
        return "danger";
      case "Excused":
        return "neutral";
      default:
        return "neutral";
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
            View and manage student attendance records across departments
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
          <div className="flex flex-col gap-4 mb-6">
            <div className="flex flex-col lg:flex-row gap-4 lg:items-center">
              <div className="relative w-full lg:w-64 xl:w-80 shrink-0">
                <Search
                  className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400"
                  size={18}
                />
                <Input
                  placeholder="Search..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10 bg-white/50 dark:bg-white/5"
                />
              </div>
              <div className="flex items-center gap-2 overflow-x-auto pb-1 lg:pb-0 w-full">
                <Filter className="text-gray-400 mr-1 shrink-0" size={18} />
                <select
                  value={departmentFilter}
                  onChange={handleDepartmentChange}
                  className="h-10 rounded-xl glass-input px-4 text-sm text-gray-900 dark:text-gray-100 bg-transparent appearance-none pr-8 cursor-pointer border border-gray-200 dark:border-white/10 focus:outline-none focus:ring-2 focus:ring-primary/50 shrink-0"
                  style={{
                    backgroundImage: `url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%236B7280%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpolyline%20points%3D%226%209%2012%2015%2018%209%22%3E%3C%2Fpolyline%3E%3C%2Fsvg%3E")`,
                    backgroundRepeat: "no-repeat",
                    backgroundPosition: "right 0.5rem center",
                    backgroundSize: "1em 1em",
                  }}
                >
                  <option value="All" className="bg-white dark:bg-dark-bg">
                    All Departments
                  </option>
                  {uniqueDepartments.map((dep) => (
                    <option
                      key={dep}
                      value={dep}
                      className="bg-white dark:bg-dark-bg"
                    >
                      {dep}
                    </option>
                  ))}
                </select>
                <select
                  value={courseFilter}
                  onChange={(e) => setCourseFilter(e.target.value)}
                  className="h-10 rounded-xl glass-input px-4 text-sm text-gray-900 dark:text-gray-100 bg-transparent appearance-none pr-8 cursor-pointer border border-gray-200 dark:border-white/10 focus:outline-none focus:ring-2 focus:ring-primary/50 shrink-0"
                  style={{
                    backgroundImage: `url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%236B7280%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpolyline%20points%3D%226%209%2012%2015%2018%209%22%3E%3C%2Fpolyline%3E%3C%2Fsvg%3E")`,
                    backgroundRepeat: "no-repeat",
                    backgroundPosition: "right 0.5rem center",
                    backgroundSize: "1em 1em",
                  }}
                >
                  <option value="All" className="bg-white dark:bg-dark-bg">
                    All Courses
                  </option>
                  {uniqueCourses.map((course) => (
                    <option
                      key={course}
                      value={course}
                      className="bg-white dark:bg-dark-bg"
                    >
                      {course}
                    </option>
                  ))}
                </select>
                <select
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value)}
                  className="h-10 rounded-xl glass-input px-4 text-sm text-gray-900 dark:text-gray-100 bg-transparent appearance-none pr-8 cursor-pointer border border-gray-200 dark:border-white/10 focus:outline-none focus:ring-2 focus:ring-primary/50 shrink-0"
                  style={{
                    backgroundImage: `url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%236B7280%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpolyline%20points%3D%226%209%2012%2015%2018%209%22%3E%3C%2Fpolyline%3E%3C%2Fsvg%3E")`,
                    backgroundRepeat: "no-repeat",
                    backgroundPosition: "right 0.5rem center",
                    backgroundSize: "1em 1em",
                  }}
                >
                  <option value="All" className="bg-white dark:bg-dark-bg">
                    All Statuses
                  </option>
                  <option
                    value="Present"
                    className="bg-white dark:bg-dark-bg text-emerald-600 dark:text-emerald-400"
                  >
                    Present
                  </option>
                  <option
                    value="Late"
                    className="bg-white dark:bg-dark-bg text-amber-600 dark:text-amber-400"
                  >
                    Late
                  </option>
                  <option
                    value="Absent"
                    className="bg-white dark:bg-dark-bg text-rose-600 dark:text-rose-400"
                  >
                    Absent
                  </option>
                  <option
                    value="Excused"
                    className="bg-white dark:bg-dark-bg text-gray-600 dark:text-gray-400"
                  >
                    Excused
                  </option>
                </select>
              </div>
            </div>
          </div>

          <div className="overflow-x-auto custom-scrollbar rounded-xl border border-gray-100 dark:border-white/5">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Student Name</TableHead>
                  <TableHead>Course</TableHead>
                  <TableHead>Department</TableHead>
                  <TableHead>Session ID</TableHead>
                  <TableHead>Attendance</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Confidence</TableHead>
                  <TableHead>Recognized At</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {isLoading ? (
                  <TableRow>
                    <TableCell
                      colSpan={9}
                      className="h-32 text-center text-gray-500"
                    >
                      Loading live attendance data...
                    </TableCell>
                  </TableRow>
                ) : filteredRecords.length === 0 ? (
                  <TableRow>
                    <TableCell
                      colSpan={9}
                      className="h-32 text-center text-gray-500"
                    >
                      No live attendance records found in the database.
                    </TableCell>
                  </TableRow>
                ) : (
                  filteredRecords.map((record) => (
                    <TableRow
                      key={record.id}
                      className="hover:bg-gray-50/50 dark:hover:bg-white/[0.02]"
                    >
                      <TableCell className="font-medium text-gray-900 dark:text-gray-100">
                        {record.studentName}
                      </TableCell>
                      <TableCell className="text-gray-600 dark:text-gray-400">
                        {record.course}
                      </TableCell>
                      <TableCell className="text-gray-600 dark:text-gray-400">
                        {record.department}
                      </TableCell>
                      <TableCell className="text-gray-500 dark:text-gray-400 font-mono text-sm">
                        {record.sessionId}
                      </TableCell>
                      <TableCell className="text-gray-700 dark:text-gray-300 font-medium">
                        {record.attendedSessions} / {record.totalSessions}
                      </TableCell>
                      <TableCell>
                        <Badge
                          variant={getStatusBadgeVariant(record.status) as any}
                        >
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
                              <CalendarIcon
                                size={14}
                                className="text-gray-400"
                              />
                              <span>
                                {new Date(record.recognizedAt).toLocaleString(
                                  [],
                                  {
                                    dateStyle: "medium",
                                    timeStyle: "short",
                                  },
                                )}
                              </span>
                            </>
                          ) : (
                            <span className="pl-4">-</span>
                          )}
                        </div>
                      </TableCell>
                      <TableCell className="text-right">
                        {record.status === "Absent" ? (
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-8 px-3 gap-1.5 text-xs font-semibold text-amber-600 hover:text-amber-700 hover:bg-amber-50 dark:text-amber-400 dark:hover:text-amber-300 dark:hover:bg-amber-500/10 border border-amber-200 dark:border-amber-500/30 rounded-lg"
                            onClick={() => handleEditClick(record)}
                            title="Mark as Excused"
                          >
                            <ShieldCheck size={14} />
                            Excuse
                          </Button>
                        ) : (
                          <span className="text-xs text-gray-300 dark:text-gray-600 select-none px-2">
                            —
                          </span>
                        )}
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>

      {/* Edit Modal */}
      <Modal
        isOpen={isEditModalOpen}
        onClose={() => setIsEditModalOpen(false)}
        title="Edit Attendance Status"
      >
        <form onSubmit={handleStatusUpdate} className="space-y-5">
          <div className="flex items-center gap-3 p-3 bg-amber-50 dark:bg-amber-500/10 text-amber-800 dark:text-amber-200 rounded-lg text-sm mb-4 border border-amber-200 dark:border-amber-500/20">
            <AlertCircle size={18} className="shrink-0" />
            <p>
              You can only change the status from <strong>Absent</strong> to{" "}
              <strong>Excused</strong>. Other fields cannot be edited.
            </p>
          </div>

          <div className="space-y-4">
            <div>
              <label className="text-sm font-medium text-gray-700 dark:text-gray-300 block mb-1.5">
                Student Name
              </label>
              <Input
                value={selectedRecord?.studentName || ""}
                disabled
                className="bg-gray-100 dark:bg-gray-800/50 text-gray-500 dark:text-gray-400 cursor-not-allowed"
              />
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300 block mb-1.5">
                  Course
                </label>
                <Input
                  value={selectedRecord?.course || ""}
                  disabled
                  className="bg-gray-100 dark:bg-gray-800/50 text-gray-500 dark:text-gray-400 cursor-not-allowed"
                />
              </div>
              <div>
                <label className="text-sm font-medium text-gray-700 dark:text-gray-300 block mb-1.5">
                  Session ID
                </label>
                <Input
                  value={selectedRecord?.sessionId || ""}
                  disabled
                  className="bg-gray-100 dark:bg-gray-800/50 text-gray-500 dark:text-gray-400 cursor-not-allowed"
                />
              </div>
            </div>

            <div>
              <label className="text-sm font-medium text-gray-700 dark:text-gray-300 block mb-1.5">
                Status
              </label>
              <select className="w-full h-10 rounded-xl glass-input px-4 text-sm text-gray-900 dark:text-gray-100 bg-transparent border border-gray-200 dark:border-white/10 focus:outline-none focus:ring-2 focus:ring-primary/50">
                <option
                  value="Excused"
                  className="bg-white dark:bg-dark-bg text-gray-900 dark:text-gray-100"
                >
                  Excused
                </option>
              </select>
            </div>
          </div>

          {saveError && (
            <div className="flex items-start gap-2 p-3 bg-rose-50 dark:bg-rose-500/10 border border-rose-200 dark:border-rose-500/20 rounded-lg text-sm text-rose-700 dark:text-rose-300">
              <AlertCircle size={16} className="shrink-0 mt-0.5" />
              <span>{saveError}</span>
            </div>
          )}

          <div className="pt-4 flex justify-end gap-3 border-t border-gray-100 dark:border-white/10 mt-6">
            <Button
              type="button"
              variant="secondary"
              onClick={() => setIsEditModalOpen(false)}
            >
              Cancel
            </Button>
            <Button type="submit" disabled={isSaving}>
              {isSaving ? "Saving..." : "Save Changes"}
            </Button>
          </div>
        </form>
      </Modal>
    </div>
  );
}
