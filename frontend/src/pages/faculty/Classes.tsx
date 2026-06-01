import { useState, useEffect } from "react";
import { Search, Calendar } from "lucide-react";
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
import { facultyService } from "@/services/facultyService";

export default function FacultyClasses() {
  const [classes, setClasses] = useState<any[]>([]);
  const [search, setSearch] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    facultyService
      .getClasses()
      .then((res) => setClasses(res?.items ?? res ?? []))
      .catch(() => setError("Failed to load classes."))
      .finally(() => setIsLoading(false));
  }, []);

  const filtered = classes.filter(
    (c) =>
      c.name?.toLowerCase().includes(search.toLowerCase()) ||
      c.code?.toLowerCase().includes(search.toLowerCase()),
  );

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-gray-100 flex items-center gap-3">
          <Calendar className="text-emerald-500" size={28} />
          Classes
        </h1>
        <p className="text-gray-500 dark:text-gray-400 mt-1">
          Class batches running under your faculty.
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
              placeholder="Search classes..."
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
                  <TableHead>Class Name</TableHead>
                  <TableHead>Code</TableHead>
                  <TableHead>Year</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {isLoading ? (
                  <TableRow>
                    <TableCell colSpan={4} className="h-32 text-center text-gray-500">
                      Loading classes...
                    </TableCell>
                  </TableRow>
                ) : filtered.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={4} className="h-32 text-center text-gray-500">
                      No classes found.
                    </TableCell>
                  </TableRow>
                ) : (
                  filtered.map((c, i) => (
                    <TableRow
                      key={c.id}
                      className="hover:bg-gray-50/50 dark:hover:bg-white/[0.02]"
                    >
                      <TableCell className="text-gray-400 text-sm">{i + 1}</TableCell>
                      <TableCell className="font-medium text-gray-900 dark:text-gray-100">
                        {c.name}
                      </TableCell>
                      <TableCell className="font-mono text-gray-600 dark:text-gray-400">
                        {c.code ?? "—"}
                      </TableCell>
                      <TableCell className="text-gray-600 dark:text-gray-400">
                        {c.year ?? c.academic_year ?? "—"}
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </div>

          {!isLoading && (
            <p className="text-sm text-gray-400 mt-3">
              {filtered.length} class{filtered.length !== 1 ? "es" : ""}
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
