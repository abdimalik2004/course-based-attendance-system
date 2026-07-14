import { useEffect, useMemo, useState } from "react";
import { Search, BookOpen } from "lucide-react";
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

export default function FacultyCourses() {
  const { courses, assignments, schedules, teachers, isLoading, error, fetchData } =
    useFacultyStore();

  const [search, setSearch] = useState("");
  const [viewCourseId, setViewCourseId] = useState<string | null>(null);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  // Build a lookup: courseId → primary teacher name
  const primaryTeacherByCourse = useMemo(() => {
    const map = new Map<string, string>();
    assignments.forEach((a) => {
      if (a.isPrimary) {
        const t = teachers.find((t) => t.id === a.teacherId);
        if (t) map.set(a.courseId, t.fullName);
      }
    });
    return map;
  }, [assignments, teachers]);

  // Build a lookup: courseId → schedule summary
  const scheduleByCourse = useMemo(() => {
    const map = new Map<string, { days: string; time: string }>();
    schedules.forEach((s) => {
      const days = s.weekdays.map((d) => d.charAt(0).toUpperCase() + d.slice(1)).join(", ");
      const time = `${s.startTime} – ${s.endTime}`;
      map.set(s.courseId, { days, time });
    });
    return map;
  }, [schedules]);

  const filtered = useMemo(
    () =>
      courses.filter(
        (c) =>
          c.title?.toLowerCase().includes(search.toLowerCase()) ||
          c.code?.toLowerCase().includes(search.toLowerCase()),
      ),
    [courses, search],
  );

  const viewCourse = viewCourseId ? courses.find((c) => c.id === viewCourseId) : null;

  const viewData = useMemo(() => {
    if (!viewCourse) return null;
    const teacher = primaryTeacherByCourse.get(viewCourse.id);
    const sched = scheduleByCourse.get(viewCourse.id);
    const allAssignments = assignments.filter((a) => a.courseId === viewCourse.id);
    const secondaryTeachers = allAssignments
      .filter((a) => !a.isPrimary)
      .map((a) => teachers.find((t) => t.id === a.teacherId)?.fullName)
      .filter(Boolean)
      .join(", ");

    return [
      { label: "Course Code", value: viewCourse.code },
      { label: "Course Title", value: viewCourse.title },
      { label: "Primary Teacher", value: teacher ?? "—" },
      { label: "Secondary Teacher(s)", value: secondaryTeachers || "—" },
      { label: "Schedule Days", value: sched?.days ?? "—" },
      { label: "Schedule Time", value: sched?.time ?? "—" },
    ];
  }, [viewCourse, primaryTeacherByCourse, scheduleByCourse, assignments, teachers]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-gray-100 flex items-center gap-3">
          <BookOpen className="text-rose-500" size={28} />
          Courses
        </h1>
        <p className="text-gray-500 dark:text-gray-400 mt-1">
          All courses offered by your faculty.
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
              placeholder="Search by title or code..."
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
                  <TableHead>Code</TableHead>
                  <TableHead>Course Title</TableHead>
                  <TableHead>Primary Teacher</TableHead>
                  <TableHead>Schedule</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {isLoading ? (
                  Array.from({ length: 4 }).map((_, i) => (
                    <TableRow key={`sk-${i}`}>
                      {Array.from({ length: 6 }).map((_, j) => (
                        <TableCell key={j}>
                          <div className="h-4 rounded bg-gray-200 dark:bg-white/10 animate-pulse" />
                        </TableCell>
                      ))}
                    </TableRow>
                  ))
                ) : filtered.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={6} className="h-32 text-center text-gray-500">
                      No courses found.
                    </TableCell>
                  </TableRow>
                ) : (
                  filtered.map((c, i) => {
                    const teacher = primaryTeacherByCourse.get(c.id);
                    const sched = scheduleByCourse.get(c.id);
                    return (
                      <TableRow
                        key={c.id}
                        className="hover:bg-gray-50/50 dark:hover:bg-white/[0.02]"
                      >
                        <TableCell className="text-gray-400 text-sm">{i + 1}</TableCell>
                        <TableCell className="font-mono font-medium text-gray-700 dark:text-gray-300">
                          {c.code}
                        </TableCell>
                        <TableCell className="font-medium text-gray-900 dark:text-gray-100">
                          {c.title}
                        </TableCell>
                        <TableCell className="text-gray-600 dark:text-gray-400">
                          {teacher ?? (
                            <span className="text-gray-300 dark:text-gray-600">Unassigned</span>
                          )}
                        </TableCell>
                        <TableCell>
                          {sched ? (
                            <div className="flex flex-col gap-0.5">
                              <span className="text-xs font-medium text-purple-500">
                                {sched.days}
                              </span>
                              <span className="text-xs text-gray-500">{sched.time}</span>
                            </div>
                          ) : (
                            <Badge variant="neutral" className="text-xs">No schedule</Badge>
                          )}
                        </TableCell>
                        <TableCell className="text-right">
                          <ViewButton
                            onClick={() => setViewCourseId(c.id)}
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
            <p className="text-sm text-gray-400 mt-3">
              {filtered.length} course{filtered.length !== 1 ? "s" : ""}
            </p>
          )}
        </CardContent>
      </Card>

      <ViewModal
        isOpen={!!viewCourse}
        onClose={() => setViewCourseId(null)}
        title="Course Details"
        data={viewData}
      />
    </div>
  );
}
