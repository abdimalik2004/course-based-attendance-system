import { useEffect, useMemo, useState } from "react";
import { Search, Users, Camera, Mail, Phone } from "lucide-react";
import { Card, CardContent } from "@/components/ui/Card";
import { Input } from "@/components/ui/Input";
import { Badge } from "@/components/ui/Badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/Table";
import { ViewButton } from "@/components/ui/ViewButton";
import { ViewModal } from "@/components/ui/ViewModal";
import admissionService, { type AdmissionStudentDto } from "@/services/admissionService";
import { hrService, type Department } from "@/services/hrService";
import { useAuthStore } from "@/store/useAuthStore";

export default function FacultyStudents() {
  const { user } = useAuthStore();
  const [students, setStudents] = useState<AdmissionStudentDto[]>([]);
  const [departments, setDepartments] = useState<Department[]>([]);
  const [search, setSearch] = useState("");
  const [deptFilter, setDeptFilter] = useState("all");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [viewStudent, setViewStudent] = useState<AdmissionStudentDto | null>(null);

  useEffect(() => {
    const facultyId = user?.facultyId;
    if (!facultyId) {
      setError("No faculty associated with your account.");
      setIsLoading(false);
      return;
    }

    Promise.all([
      admissionService.listStudents({
        limit: 500,
        faculty_id: Number(facultyId),
        status: "approved",
      }),
      hrService.getDepartmentsByFaculty(String(facultyId)).catch(() => [] as import("@/services/hrService").Department[]),
    ])
      .then(([studentsRes, depts]) => {
        setStudents(studentsRes.items);
        setDepartments(depts);
      })
      .catch(() => setError("Failed to load students."))
      .finally(() => setIsLoading(false));
  }, [user?.facultyId]);

  // Build dept id → name map
  // hrService returns id as string; student.department_id is a number — key on Number so lookups match
  const deptMap = useMemo(
    () => new Map(departments.map((d) => [Number(d.id), d.name])),
    [departments],
  );

  const filtered = useMemo(() => {
    return students.filter((s) => {
      const matchSearch =
        s.full_name?.toLowerCase().includes(search.toLowerCase()) ||
        s.student_number?.toLowerCase().includes(search.toLowerCase()) ||
        s.email?.toLowerCase().includes(search.toLowerCase());
      const matchDept =
        deptFilter === "all" || String(s.department_id) === deptFilter;
      return matchSearch && matchDept;
    });
  }, [students, search, deptFilter]);

  const formatDate = (d: string | null | undefined) => {
    if (!d) return "—";
    try {
      return new Date(d).toLocaleDateString([], {
        year: "numeric",
        month: "short",
        day: "numeric",
      });
    } catch {
      return d;
    }
  };

  const viewData = useMemo(() => {
    if (!viewStudent) return null;
    return [
      { label: "Student Number", value: viewStudent.student_number },
      { label: "Full Name", value: viewStudent.full_name },
      {
        label: "Department",
        value: deptMap.get(viewStudent.department_id) ?? `Dept #${viewStudent.department_id}`,
      },
      { label: "Email", value: viewStudent.email ?? "—" },
      { label: "Phone", value: viewStudent.phone ?? "—" },
      { label: "Date of Birth", value: formatDate(viewStudent.date_of_birth) },
      { label: "Face Images", value: `${viewStudent.face_images_count} captured` },
      { label: "Status", value: viewStudent.status },
      { label: "Enrolled On", value: formatDate(viewStudent.created_at) },
    ];
  }, [viewStudent, deptMap]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-gray-100 flex items-center gap-3">
          <Users className="text-primary" size={28} />
          Students
        </h1>
        <p className="text-gray-500 dark:text-gray-400 mt-1">
          Approved students registered under your faculty.
        </p>
      </div>

      {error && (
        <div className="rounded-2xl border border-rose-200 dark:border-rose-500/20 bg-rose-50 dark:bg-rose-500/10 p-4 text-sm text-rose-700 dark:text-rose-200">
          {error}
        </div>
      )}

      <Card className="glass-card shadow-2xl shadow-primary/5">
        <CardContent className="p-6">
          <div className="flex flex-col sm:flex-row gap-3 mb-4">
            <div className="relative w-full max-w-sm">
              <Search
                className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400"
                size={16}
              />
              <Input
                placeholder="Search by name, number, or email..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="pl-9"
              />
            </div>

            {departments.length > 0 && (
              <select
                value={deptFilter}
                onChange={(e) => setDeptFilter(e.target.value)}
                className="h-10 rounded-xl glass-input px-4 text-sm text-gray-900 dark:text-gray-100 bg-transparent border border-gray-200 dark:border-white/10 focus:outline-none focus:ring-2 focus:ring-primary/50 appearance-none pr-8"
                style={{
                  backgroundImage: `url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%2224%22%20height%3D%2224%22%20viewBox%3D%220%200%2024%2024%22%20fill%3D%22none%22%20stroke%3D%22%236B7280%22%20stroke-width%3D%222%22%20stroke-linecap%3D%22round%22%20stroke-linejoin%3D%22round%22%3E%3Cpolyline%20points%3D%226%209%2012%2015%2018%209%22%3E%3C%2Fpolyline%3E%3C%2Fsvg%3E")`,
                  backgroundRepeat: "no-repeat",
                  backgroundPosition: "right 0.5rem center",
                  backgroundSize: "1em 1em",
                }}
              >
                <option value="all">All Departments</option>
                {departments.map((d) => (
                  <option key={d.id} value={String(d.id)}>
                    {d.name}
                  </option>
                ))}
              </select>
            )}
          </div>

          <div className="overflow-x-auto custom-scrollbar rounded-xl border border-gray-100 dark:border-white/5">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>#</TableHead>
                  <TableHead>Student No.</TableHead>
                  <TableHead>Full Name</TableHead>
                  <TableHead>Department</TableHead>
                  <TableHead>Contact</TableHead>
                  <TableHead>Faces</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {isLoading ? (
                  Array.from({ length: 5 }).map((_, i) => (
                    <TableRow key={`sk-${i}`}>
                      {Array.from({ length: 7 }).map((_, j) => (
                        <TableCell key={j}>
                          <div className="h-4 rounded bg-gray-200 dark:bg-white/10 animate-pulse" />
                        </TableCell>
                      ))}
                    </TableRow>
                  ))
                ) : filtered.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={7} className="h-32 text-center text-gray-500">
                      No approved students found.
                    </TableCell>
                  </TableRow>
                ) : (
                  filtered.map((s, i) => (
                    <TableRow
                      key={s.id}
                      className="hover:bg-gray-50/50 dark:hover:bg-white/[0.02]"
                    >
                      <TableCell className="text-gray-400 text-sm">{i + 1}</TableCell>
                      <TableCell className="font-mono font-medium text-gray-700 dark:text-gray-300">
                        {s.student_number}
                      </TableCell>
                      <TableCell className="font-medium text-gray-900 dark:text-gray-100">
                        {s.full_name}
                      </TableCell>
                      <TableCell className="text-gray-600 dark:text-gray-400 text-sm">
                        {deptMap.get(s.department_id) ?? `#${s.department_id}`}
                      </TableCell>
                      <TableCell>
                        <div className="flex flex-col gap-0.5">
                          {s.email && (
                            <span className="flex items-center gap-1 text-xs text-gray-500">
                              <Mail size={11} />
                              {s.email}
                            </span>
                          )}
                          {s.phone && (
                            <span className="flex items-center gap-1 text-xs text-gray-500">
                              <Phone size={11} />
                              {s.phone}
                            </span>
                          )}
                          {!s.email && !s.phone && (
                            <span className="text-gray-300 dark:text-gray-600 text-xs">—</span>
                          )}
                        </div>
                      </TableCell>
                      <TableCell>
                        <span
                          className={`flex items-center gap-1 text-xs font-medium ${
                            s.face_images_count > 0
                              ? "text-emerald-500"
                              : "text-gray-400"
                          }`}
                        >
                          <Camera size={13} />
                          {s.face_images_count}
                        </span>
                      </TableCell>
                      <TableCell className="text-right">
                        <ViewButton
                          onClick={() => setViewStudent(s)}
                          tooltip="View Details"
                        />
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </div>

          {!isLoading && (
            <p className="text-sm text-gray-400 mt-3">
              {filtered.length} student{filtered.length !== 1 ? "s" : ""}
              {deptFilter !== "all" && (
                <span className="ml-1 text-gray-300 dark:text-gray-600">
                  in {deptMap.get(Number(deptFilter))}
                </span>
              )}
            </p>
          )}
        </CardContent>
      </Card>

      <ViewModal
        isOpen={!!viewStudent}
        onClose={() => setViewStudent(null)}
        title="Student Details"
        data={viewData}
      />
    </div>
  );
}
