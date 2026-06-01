import { useEffect, useState, useMemo } from "react";
import { Plus, Edit2, Trash2, Users } from "lucide-react";
import { Card, CardContent } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/Table";
import { Input } from "@/components/ui/Input";
import { Badge } from "@/components/ui/Badge";
import { ViewButton } from "@/components/ui/ViewButton";
import { ViewModal } from "@/components/ui/ViewModal";
import { useFacultyStore } from "@/store/useFacultyStore";
import { useHrStore } from "@/store/useHrStore";
import { AssignTeacherModal } from "@/components/faculty/AssignTeacherModal";
import { ConfirmDeleteModal } from "@/components/academia/ConfirmDeleteModal";
import type { TeacherAssignment } from "@/store/useFacultyStore";

export default function AssignTeacher() {
  const {
    assignments,
    courses,
    isLoading,
    error,
    fetchData,
    openModal,
    deleteAssignment,
  } = useFacultyStore();
  const { teachers, fetchTeachers } = useHrStore();
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const [viewAssignment, setViewAssignment] =
    useState<TeacherAssignment | null>(null);
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    fetchData();
    fetchTeachers();
  }, [fetchData, fetchTeachers]);

  const getCourse = (id: string) => courses.find((c) => c.id === id);
  const getTeacher = (id: string) => teachers.find((t) => t.id === id);

  const filteredAssignments = useMemo(() => {
    return assignments.filter((assignment) => {
      const course = getCourse(assignment.courseId);
      const teacher = getTeacher(assignment.teacherId);
      const searchMatch =
        (course?.title || "")
          .toLowerCase()
          .includes(searchTerm.toLowerCase()) ||
        (course?.code || "").toLowerCase().includes(searchTerm.toLowerCase()) ||
        (teacher?.fullName || "")
          .toLowerCase()
          .includes(searchTerm.toLowerCase());
      return searchMatch;
    });
  }, [assignments, searchTerm, courses, teachers]);

  return (
    <div className="space-y-6 max-w-7xl mx-auto">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-gray-100 flex items-center gap-3">
            <Users className="text-primary" size={28} />
            Course Assignments
          </h1>
          <p className="text-gray-500 dark:text-gray-400 mt-1">
            Assign teachers to faculty courses.
          </p>
        </div>
        <Button onClick={() => openModal("assign", "create")} className="gap-2">
          <Plus size={20} />
          Assign Teacher
        </Button>
      </div>

      {error ? (
        <div className="rounded-2xl border border-rose-200 dark:border-rose-500/20 bg-rose-50 dark:bg-rose-500/10 p-4 text-sm text-rose-700 dark:text-rose-200">
          {error}
        </div>
      ) : null}

      <Card>
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
                  <TableHead>Teacher Name</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {isLoading ? (
                  Array.from({ length: 3 }).map((_, i) => (
                    <TableRow key={`skeleton-${i}`}>
                      <TableCell colSpan={4}>
                        <div className="h-4 bg-gray-200 dark:bg-white/10 rounded animate-pulse" />
                      </TableCell>
                    </TableRow>
                  ))
                ) : filteredAssignments.length === 0 ? (
                  <TableRow>
                    <TableCell
                      colSpan={4}
                      className="text-center text-gray-500 py-8"
                    >
                      No assignments found matching your search.
                    </TableCell>
                  </TableRow>
                ) : (
                  filteredAssignments.map((assignment) => {
                    const course = getCourse(assignment.courseId);
                    const teacher = getTeacher(assignment.teacherId);

                    return (
                      <TableRow
                        key={assignment.id}
                        className="group hover:bg-white/5 transition-colors"
                      >
                        <TableCell>
                          <div className="font-medium text-gray-900 dark:text-white">
                            {course?.title || "Unknown Course"}
                          </div>
                          <div className="text-sm text-gray-500">
                            {course?.code}
                          </div>
                        </TableCell>
                        <TableCell className="font-medium text-gray-900 dark:text-white">
                          {teacher ? teacher.fullName : "Unknown Teacher"}
                        </TableCell>
                        <TableCell>
                          <Badge
                            variant={
                              assignment.isPrimary ? "success" : "danger"
                            }
                          >
                            {assignment.isPrimary ? "Active" : "Inactive"}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-right">
                          <div className="flex items-center justify-end gap-2">
                            <ViewButton
                              onClick={() => setViewAssignment(assignment)}
                              tooltip="View"
                            />
                            <button
                              onClick={() =>
                                openModal("assign", "edit", assignment)
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

      <AssignTeacherModal />

      <ConfirmDeleteModal
        isOpen={deleteId !== null}
        onClose={() => setDeleteId(null)}
        onConfirm={async () => {
          if (deleteId) {
            await deleteAssignment(deleteId);
            setDeleteId(null);
          }
        }}
        title="Delete Assignment"
        message="Are you sure you want to remove this teacher from the course? This action cannot be undone."
      />

      <ViewModal
        isOpen={!!viewAssignment}
        onClose={() => setViewAssignment(null)}
        title="Teacher Assignment Details"
        data={
          viewAssignment
            ? [
                {
                  label: "Course",
                  value:
                    getCourse(viewAssignment.courseId)?.title ||
                    "Unknown Course",
                },
                {
                  label: "Course Code",
                  value: getCourse(viewAssignment.courseId)?.code,
                },
                {
                  label: "Teacher Name",
                  value:
                    getTeacher(viewAssignment.teacherId)?.fullName ||
                    "Unknown Teacher",
                },
                {
                  label: "Status",
                  value: (
                    <Badge
                      variant={viewAssignment.isPrimary ? "success" : "danger"}
                    >
                      {viewAssignment.isPrimary ? "Active" : "Inactive"}
                    </Badge>
                  ),
                },
              ]
            : null
        }
      />
    </div>
  );
}
