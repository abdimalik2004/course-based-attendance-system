import { useState, useEffect } from "react";
import { Search, Users } from "lucide-react";
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
import admissionService from "@/services/admissionService";
import { useAuthStore } from "@/store/useAuthStore";

export default function FacultyStudents() {
  const { user } = useAuthStore();
  const [students, setStudents] = useState<any[]>([]);
  const [search, setSearch] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const facultyId = user?.facultyId;
    if (!facultyId) {
      setError("No faculty associated with your account.");
      setIsLoading(false);
      return;
    }
    admissionService
      .listStudents({
        limit: 200,
        faculty_id: Number(facultyId),
        status: "approved",
      })
      .then((res) => setStudents(res.items))
      .catch(() => setError("Failed to load students."))
      .finally(() => setIsLoading(false));
  }, [user?.facultyId]);

  const filtered = students.filter(
    (s) =>
      s.full_name?.toLowerCase().includes(search.toLowerCase()) ||
      s.student_number?.toLowerCase().includes(search.toLowerCase()),
  );

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
          <div className="mb-4 relative w-full max-w-sm">
            <Search
              className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400"
              size={16}
            />
            <Input
              placeholder="Search by name or student number..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="pl-9"
            />
          </div>

          <div className="overflow-x-auto custom-scrollbar rounded-xl border border-gray-100 dark:border-white/5">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>#</TableHead>
                  <TableHead>Student No.</TableHead>
                  <TableHead>Full Name</TableHead>
                  <TableHead>Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {isLoading ? (
                  <TableRow>
                    <TableCell colSpan={4} className="h-32 text-center text-gray-500">
                      Loading students...
                    </TableCell>
                  </TableRow>
                ) : filtered.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={4} className="h-32 text-center text-gray-500">
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
                      <TableCell>
                        <Badge variant="success" className="capitalize">
                          {s.status}
                        </Badge>
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
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
