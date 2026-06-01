import { useState, useEffect } from "react";
import { Search, GraduationCap } from "lucide-react";
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
import { hrService, type Teacher } from "@/services/hrService";

export default function FacultyTeachers() {
  const [teachers, setTeachers] = useState<Teacher[]>([]);
  const [search, setSearch] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    hrService
      .getTeachers()
      .then(setTeachers)
      .catch(() => setError("Failed to load teachers."))
      .finally(() => setIsLoading(false));
  }, []);

  const filtered = teachers.filter(
    (t) =>
      t.fullName?.toLowerCase().includes(search.toLowerCase()) ||
      t.teacherNumber?.toLowerCase().includes(search.toLowerCase()),
  );

  const statusVariant = (status: string) => {
    if (status === "Active") return "success";
    if (status === "Inactive") return "danger";
    return "warning"; // On Leave
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-gray-100 flex items-center gap-3">
          <GraduationCap className="text-emerald-500" size={28} />
          Teachers
        </h1>
        <p className="text-gray-500 dark:text-gray-400 mt-1">
          Teachers assigned to your faculty.
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
              placeholder="Search by name or teacher number..."
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
                  <TableHead>T-No.</TableHead>
                  <TableHead>Full Name</TableHead>
                  <TableHead>Role</TableHead>
                  <TableHead>Status</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {isLoading ? (
                  <TableRow>
                    <TableCell colSpan={5} className="h-32 text-center text-gray-500">
                      Loading teachers...
                    </TableCell>
                  </TableRow>
                ) : filtered.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={5} className="h-32 text-center text-gray-500">
                      No teachers found.
                    </TableCell>
                  </TableRow>
                ) : (
                  filtered.map((t, i) => (
                    <TableRow
                      key={t.id}
                      className="hover:bg-gray-50/50 dark:hover:bg-white/[0.02]"
                    >
                      <TableCell className="text-gray-400 text-sm">{i + 1}</TableCell>
                      <TableCell className="font-mono font-medium text-gray-700 dark:text-gray-300">
                        {t.teacherNumber || "—"}
                      </TableCell>
                      <TableCell className="font-medium text-gray-900 dark:text-gray-100">
                        {t.fullName}
                      </TableCell>
                      <TableCell className="text-gray-600 dark:text-gray-400">
                        {t.role}
                      </TableCell>
                      <TableCell>
                        <Badge variant={statusVariant(t.status) as any}>
                          {t.status}
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
              {filtered.length} teacher{filtered.length !== 1 ? "s" : ""}
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
