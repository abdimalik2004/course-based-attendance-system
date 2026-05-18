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
import type { ClassAssignment } from "@/types/academia.types";

export function ClassAssignmentsTable() {
  const {
    classAssignments,
    classes,
    courses,
    faculties,
    departments,
    openModal,
    deleteClassAssignment,
  } = useAcademiaStore();
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const [viewAssignment, setViewAssignment] = useState<ClassAssignment | null>(
    null,
  );
  const [searchTerm, setSearchTerm] = useState("");

  const getClassName = (classId: string) =>
    classes.find((c) => c.id === classId)?.name || "Unknown";
  const getCourseTitle = (courseId: string) =>
    courses.find((c) => c.id === courseId)?.title || "Unknown";
  const getFacultyName = (facultyId: string) =>
    faculties.find((f) => f.id === facultyId)?.name || "Unknown";
  const getDeptName = (deptId: string) =>
    departments.find((d) => d.id === deptId)?.name || "Unknown";

  const filteredAssignments = useMemo(() => {
    return classAssignments.filter((assignment) => {
      const searchMatch =
        getClassName(assignment.classId)
          .toLowerCase()
          .includes(searchTerm.toLowerCase()) ||
        getCourseTitle(assignment.courseId)
          .toLowerCase()
          .includes(searchTerm.toLowerCase()) ||
        getFacultyName(assignment.facultyId)
          .toLowerCase()
          .includes(searchTerm.toLowerCase()) ||
        getDeptName(assignment.departmentId)
          .toLowerCase()
          .includes(searchTerm.toLowerCase());
      return searchMatch;
    });
  }, [classAssignments, searchTerm, classes, courses, faculties, departments]);

  return (
    <>
      <Card className="glass-card">
        <CardContent className="p-0">
          <div className="p-4 border-b border-gray-200 dark:border-gray-700">
            <Input
              placeholder="Search class assignments..."
              className="max-w-sm"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
          <div className="overflow-x-auto custom-scrollbar">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Class Name</TableHead>
                  <TableHead>Course Title</TableHead>
                  <TableHead>Faculty</TableHead>
                  <TableHead>Department</TableHead>
                  <TableHead>Created At</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredAssignments.length === 0 ? (
                  <TableRow>
                    <TableCell
                      colSpan={6}
                      className="h-24 text-center text-gray-500"
                    >
                      No class assignments found matching your search.
                    </TableCell>
                  </TableRow>
                ) : (
                  filteredAssignments.map((assignment) => (
                    <TableRow key={assignment.id}>
                      <TableCell className="font-medium text-gray-900 dark:text-gray-100">
                        {getClassName(assignment.classId)}
                      </TableCell>
                      <TableCell className="text-gray-900 dark:text-gray-100">
                        {getCourseTitle(assignment.courseId)}
                      </TableCell>
                      <TableCell className="text-gray-600 dark:text-gray-300">
                        {getFacultyName(assignment.facultyId)}
                      </TableCell>
                      <TableCell className="text-gray-600 dark:text-gray-300">
                        {getDeptName(assignment.departmentId)}
                      </TableCell>
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
                              openModal("classAssign", "edit", assignment)
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
                  ))
                )}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>

      <ConfirmDeleteModal
        isOpen={!!deleteId}
        onClose={() => setDeleteId(null)}
        onConfirm={() => {
          if (deleteId) {
            deleteClassAssignment(deleteId);
            setDeleteId(null);
          }
        }}
      />

      <ViewModal
        isOpen={!!viewAssignment}
        onClose={() => setViewAssignment(null)}
        title="Class Assignment Details"
        data={
          viewAssignment
            ? [
                {
                  label: "Class Name",
                  value: getClassName(viewAssignment.classId),
                },
                {
                  label: "Course Title",
                  value: getCourseTitle(viewAssignment.courseId),
                },
                {
                  label: "Faculty Name",
                  value: getFacultyName(viewAssignment.facultyId),
                },
                {
                  label: "Department",
                  value: getDeptName(viewAssignment.departmentId),
                },
                {
                  label: "Created At",
                  value: new Date(
                    viewAssignment.createdAt,
                  ).toLocaleDateString(),
                },
              ]
            : null
        }
      />
    </>
  );
}
