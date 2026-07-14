import { useState, useEffect } from "react";
import { Search, Building2 } from "lucide-react";
import { Card, CardContent } from "@/components/ui/Card";
import { Input } from "@/components/ui/Input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/Table";
import { hrService, type Department } from "@/services/hrService";
import { useAuthStore } from "@/store/useAuthStore";

export default function FacultyDepartments() {
  const { user } = useAuthStore();
  const [departments, setDepartments] = useState<Department[]>([]);
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
    hrService
      .getDepartmentsByFaculty(String(facultyId))
      .then(setDepartments)
      .catch(() => setError("Failed to load departments."))
      .finally(() => setIsLoading(false));
  }, [user?.facultyId]);

  const filtered = departments.filter((d) =>
    d.name?.toLowerCase().includes(search.toLowerCase()),
  );

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-gray-100 flex items-center gap-3">
          <Building2 className="text-amber-500" size={28} />
          Departments
        </h1>
        <p className="text-gray-500 dark:text-gray-400 mt-1">
          Academic departments within your faculty.
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
              placeholder="Search departments..."
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
                  <TableHead>Department Name</TableHead>
                  <TableHead>Code</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {isLoading ? (
                  Array.from({ length: 4 }).map((_, i) => (
                    <TableRow key={`sk-${i}`}>
                      <TableCell><div className="h-4 w-6 rounded bg-gray-200 dark:bg-white/10 animate-pulse" /></TableCell>
                      <TableCell><div className="h-4 rounded bg-gray-200 dark:bg-white/10 animate-pulse" /></TableCell>
                      <TableCell><div className="h-4 w-16 rounded bg-gray-200 dark:bg-white/10 animate-pulse" /></TableCell>
                    </TableRow>
                  ))
                ) : filtered.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={3} className="h-32 text-center text-gray-500">
                      No departments found.
                    </TableCell>
                  </TableRow>
                ) : (
                  filtered.map((d, i) => (
                    <TableRow
                      key={d.id}
                      className="hover:bg-gray-50/50 dark:hover:bg-white/[0.02]"
                    >
                      <TableCell className="text-gray-400 text-sm">{i + 1}</TableCell>
                      <TableCell className="font-medium text-gray-900 dark:text-gray-100">
                        {d.name}
                      </TableCell>
                      <TableCell className="font-mono text-gray-600 dark:text-gray-400">
                        {d.code || "—"}
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </div>

          {!isLoading && (
            <p className="text-sm text-gray-400 mt-3">
              {filtered.length} department{filtered.length !== 1 ? "s" : ""}
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
