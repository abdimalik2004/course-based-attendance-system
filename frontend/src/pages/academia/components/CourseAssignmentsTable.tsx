import { useState, useMemo } from "react";
import { Edit2, Trash2 } from "lucide-react";
import { Card, CardContent } from "@/components/ui/Card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/academia/Table";
import { Input } from "@/components/ui/Input";
import { useAcademiaStore } from "@/store/useAcademiaStore";
import { ConfirmDeleteModal } from "@/components/academia/ConfirmDeleteModal";
import { ViewButton } from "@/components/ui/ViewButton";
import { ViewModal } from "@/components/ui/ViewModal";
import type { CourseAssignment } from "@/types/academia.types";

export function CourseAssignmentsTable() {
  const {
    courseAssignments,
    courses,
    faculties,
    departments,
    isLoading,
    openModal,
    deleteCourseAssignment,
  } = useAcademiaStore();
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const [viewAssignment, setViewAssignment] = useState<CourseAssignment | null>(
    null,
  );
  const [searchTerm, setSearchTerm] = useState("");

  const getCourse = (courseId: string) =>
    courses.find((c) => c.id === courseId) ?? null;
  const getFacultyName = (facultyId: string) =>
    faculties.find((f) => f.id === facultyId)?.name || "Unknown";
  const getDeptName = (deptId: string) =>
    departments.find((d) => d.id === deptId)?.name || "Unknown";

  const filteredAssignments = useMemo(() => {
    const q = searchTerm.toLowerCase();
    return courseAssignments.filter((a) => {
      const course = getCourse(a.courseId);
      return (
        (course?.code ?? "").toLowerCase().includes(q) ||
        (course?.title ?? "").toLowerCase().includes(q) ||
        getFacultyName(a.facultyId).toLowerCase().includes(q) ||
        getDeptName(a.departmentId).toLowerCase().includes(q)
      );
    });
  }, [courseAssignments, searchTerm, courses, faculties, departments]);

  return (
    <>
      <Card className="glass-card">
        <CardContent className="p-0">
          <div className="p-4 border-b border-gray-200 dark:border-gray-700">
            <Input
              placeholder="Search assignments..."
              className="max-w-sm"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
          <div className="overflow-x-auto custom-scrollbar">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Course</TableHead>
                  <TableHead>Faculty Name</TableHead>
                  <TableHead>Department</TableHead>
                  <TableHead>Semester</TableHead>
                  <TableHead>Created At</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {isLoading ? (
                  Array.from({ length: 3 }).map((_, i) => (
                    <TableRow key={`skeleton-${i}`}>
                      <TableCell colSpan={6}>
                        <div className="h-4 bg-gray-200 dark:bg-white/10 rounded animate-pulse" />
                      </TableCell>
                    </TableRow>
                  ))
                ) : filteredAssignments.length === 0 ? (
                  <TableRow>
                    <TableCell
                      colSpan={6}
                      className="h-24 text-center text-gray-500"
                    >
                      No course assignments found matching your search.
                    </TableCell>
                  </TableRow>
                ) : (
                  filteredAssignments.map((assignment) => {
                    const course = getCourse(assignment.courseId);
                    return (
                      <TableRow key={assignment.id}>
                        <TableCell>
                          <div className="font-medium text-gray-900 dark:text-gray-100">
                            {course?.code ?? "Unknown"}
                          </div>
                          {course?.title && (
                            <div className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
                              {course.title}
                            </div>
                          )}
                        </TableCell>
                        <TableCell className="text-gray-600 dark:text-gray-300">
                          {getFacultyName(assignment.facultyId)}
                        </TableCell>
                        <TableCell className="text-gray-600 dark:text-gray-300">
                          {getDeptName(assignment.departmentId)}
                        </TableCell>
                        <TableCell>Semester {assignment.semester}</TableCell>
                        <TableCell className="text-gray-500 dark:text-gray-400">
                          {new Date(assignment.createdAt).toLocaleDateString()}
                        </TableCell>
                        <TableCell className="text-right">
                          <div className="flex items-center justify-end gap-2">
                            <ViewButton
                              onClick={() => setViewAssignment(assignment)}
                              tooltip="View"
                            />
                            <button
                              onClick={() =>
                                openModal("courseAssign", "edit", assignment)
                              }
                              className="p-1.5 rounded-lg text-amber-500 hover:text-amber-600 hover:bg-amber-50 dark:hover:bg-amber-500/10 transition-colors"
                              title="Edit"
                            >
                              <Edit2 size={16} />
                            </button>
                            <button
                              onClick={() => setDeleteId(assignment.id)}
                              className="p-1.5 rounded-lg text-red-500 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-500/10 transition-colors"
                              title="Delete"
                            >
                              <Trash2 size={16} />
                            </button>
                          </div>
                        </TableCell>
                      </TableRow>
                    );
                  })
                )}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>

      <ConfirmDeleteModal
        isOpen={!!deleteId}
        onClose={() => setDeleteId(null)}
        onConfirm={async () => {
          if (deleteId) await deleteCourseAssignment(deleteId);
        }}
      />

      <ViewModal
        isOpen={!!viewAssignment}
        onClose={() => setViewAssignment(null)}
        title="Course Assignment Details"
        data={
          viewAssignment
            ? (() => {
                const course = getCourse(viewAssignment.courseId);
                return [
                  { label: "Course Code", value: course?.code ?? "Unknown" },
                  { label: "Course Title", value: course?.title ?? "Unknown" },
                  {
                    label: "Faculty Name",
                    value: getFacultyName(viewAssignment.facultyId),
                  },
                  {
                    label: "Department",
                    value: getDeptName(viewAssignment.departmentId),
                  },
                  { label: "Semester", value: viewAssignment.semester },
                  {
                    label: "Created At",
                    value: new Date(
                      viewAssignment.createdAt,
                    ).toLocaleDateString(),
                  },
                ];
              })()
            : null
        }
      />
    </>
  );
}
