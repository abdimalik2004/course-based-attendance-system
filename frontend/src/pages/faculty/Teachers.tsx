import { useEffect, useMemo, useState } from "react";
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
import { ViewButton } from "@/components/ui/ViewButton";
import { ViewModal } from "@/components/ui/ViewModal";
import { useFacultyStore } from "@/store/useFacultyStore";

export default function FacultyTeachers() {
  const { teachers, assignments, courses, isLoading, error, fetchData } =
    useFacultyStore();

  const [search, setSearch] = useState("");
  const [viewTeacherId, setViewTeacherId] = useState<string | null>(null);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Build lookup: teacherId → array of assigned course objects
  const coursesByTeacher = useMemo(() => {
    const map = new Map<string, Array<{ code: string; title: string; isPrimary: boolean }>>();
    assignments.forEach((a) => {
      const course = courses.find((c) => c.id === a.courseId);
      if (!course) return;
      if (!map.has(a.teacherId)) map.set(a.teacherId, []);
      map.get(a.teacherId)!.push({
        code: course.code,
        title: course.title,
        isPrimary: a.isPrimary,
      });
    });
    return map;
  }, [assignments, courses]);

  const filtered = useMemo(
    () =>
      teachers.filter((t) =>
        t.fullName?.toLowerCase().includes(search.toLowerCase()),
      ),
    [teachers, search],
  );

  const viewTeacher = viewTeacherId
    ? teachers.find((t) => t.id === viewTeacherId)
    : null;

  const viewData = useMemo(() => {
    if (!viewTeacher) return null;
    const teacherCourses = coursesByTeacher.get(viewTeacher.id) ?? [];
    const primaryCourses = teacherCourses.filter((c) => c.isPrimary);
    const secondaryCourses = teacherCourses.filter((c) => !c.isPrimary);

    return [
      { label: "Full Name", value: viewTeacher.fullName },
      {
        label: "Primary Courses",
        value:
          primaryCourses.length > 0
            ? primaryCourses.map((c) => `${c.code} — ${c.title}`).join("\n")
            : "—",
      },
      {
        label: "Secondary Courses",
        value:
          secondaryCourses.length > 0
            ? secondaryCourses.map((c) => `${c.code} — ${c.title}`).join("\n")
            : "—",
      },
    ];
  }, [viewTeacher, coursesByTeacher]);

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
              placeholder="Search by name..."
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
                  <TableHead>Full Name</TableHead>
                  <TableHead>Assigned Courses</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {isLoading ? (
                  Array.from({ length: 4 }).map((_, i) => (
                    <TableRow key={`sk-${i}`}>
                      {Array.from({ length: 4 }).map((_, j) => (
                        <TableCell key={j}>
                          <div className="h-4 rounded bg-gray-200 dark:bg-white/10 animate-pulse" />
                        </TableCell>
                      ))}
                    </TableRow>
                  ))
                ) : filtered.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={4} className="h-32 text-center text-gray-500">
                      No teachers found.
                    </TableCell>
                  </TableRow>
                ) : (
                  filtered.map((t, i) => {
                    const teacherCourses = coursesByTeacher.get(t.id) ?? [];
                    return (
                      <TableRow
                        key={t.id}
                        className="hover:bg-gray-50/50 dark:hover:bg-white/[0.02]"
                      >
                        <TableCell className="text-gray-400 text-sm">{i + 1}</TableCell>
                        <TableCell className="font-medium text-gray-900 dark:text-gray-100">
                          {t.fullName}
                        </TableCell>
                        <TableCell>
                          {teacherCourses.length === 0 ? (
                            <span className="text-gray-300 dark:text-gray-600 text-sm">
                              None
                            </span>
                          ) : (
                            <div className="flex flex-wrap gap-1">
                              {teacherCourses.map((c) => (
                                <span
                                  key={c.code}
                                  title={`${c.isPrimary ? "Primary" : "Secondary"}: ${c.title}`}
                                  className={`px-2 py-0.5 rounded text-xs font-medium border ${
                                    c.isPrimary
                                      ? "bg-emerald-500/10 text-emerald-400 border-emerald-500/20"
                                      : "bg-blue-500/10 text-blue-400 border-blue-500/20"
                                  }`}
                                >
                                  {c.code}
                                </span>
                              ))}
                            </div>
                          )}
                        </TableCell>
                        <TableCell className="text-right">
                          <ViewButton
                            onClick={() => setViewTeacherId(t.id)}
                            tooltip="View Details"
                          />
                        </TableCell>
                      </TableRow>
                    );
                  })
                )}
              </TableBody>
            </Table>
          </div>

          {!isLoading && (
            <div className="flex items-center gap-4 mt-3">
              <p className="text-sm text-gray-400">
                {filtered.length} teacher{filtered.length !== 1 ? "s" : ""}
              </p>
              <div className="flex items-center gap-3 text-xs text-gray-400">
                <span className="flex items-center gap-1">
                  <span className="w-2 h-2 rounded-full bg-emerald-400 inline-block" />
                  Primary
                </span>
                <span className="flex items-center gap-1">
                  <span className="w-2 h-2 rounded-full bg-blue-400 inline-block" />
                  Secondary
                </span>
              </div>
            </div>
          )}
        </CardContent>
      </Card>

      <ViewModal
        isOpen={!!viewTeacher}
        onClose={() => setViewTeacherId(null)}
        title="Teacher Details"
        data={viewData}
      />
    </div>
  );
}
