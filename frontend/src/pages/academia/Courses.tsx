import { useEffect, useState, useMemo } from "react";
import { Plus, Edit2, Trash2, BookOpen } from "lucide-react";
import { Pagination } from "@/components/academia/Pagination";
import { Card, CardContent } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
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
import { CourseModal } from "./components/CourseModal";
import { ConfirmDeleteModal } from "@/components/academia/ConfirmDeleteModal";
import { ViewButton } from "@/components/ui/ViewButton";
import { ViewModal } from "@/components/ui/ViewModal";
import type { Course } from "@/types/academia.types";

export default function Courses() {
  const {
    courses,
    faculties,
    departments,
    isLoading,
    error,
    fetchData,
    openModal,
    deleteCourse,
  } = useAcademiaStore();
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const [viewCourse, setViewCourse] = useState<Course | null>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [page, setPage] = useState(1);
  const PAGE_SIZE = 10;

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const getFacultyName = (facultyId: string) => {
    return faculties.find((f) => f.id === facultyId)?.name || "Unknown";
  };

  const getDepartmentName = (departmentId: string) => {
    return departments.find((d) => d.id === departmentId)?.name || "Unknown";
  };

  const filteredCourses = useMemo(() => {
    setPage(1);
    return courses.filter((c) => {
      const searchMatch =
        c.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
        c.code.toLowerCase().includes(searchTerm.toLowerCase());
      const facultyMatch = getFacultyName(c.facultyId)
        .toLowerCase()
        .includes(searchTerm.toLowerCase());
      const deptMatch = getDepartmentName(c.departmentId)
        .toLowerCase()
        .includes(searchTerm.toLowerCase());
      return searchMatch || facultyMatch || deptMatch;
    });
  }, [courses, searchTerm, faculties, departments]);

  const totalPages = Math.max(1, Math.ceil(filteredCourses.length / PAGE_SIZE));
  const paginatedCourses = filteredCourses.slice(
    (page - 1) * PAGE_SIZE,
    page * PAGE_SIZE,
  );

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-gray-100 flex items-center gap-3">
            <BookOpen className="text-primary" size={28} />
            Courses
          </h1>
          <p className="text-gray-500 dark:text-gray-400 mt-1">
            Manage course curriculum securely linked to faculties.
          </p>
          {error && (
            <div className="mt-3 rounded-2xl border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700 dark:border-rose-500/20 dark:bg-rose-500/10 dark:text-rose-200">
              {error}
            </div>
          )}
        </div>
        <Button onClick={() => openModal("course", "create")} className="gap-2">
          <Plus size={20} />
          Create Course
        </Button>
      </div>

      <Card className="glass-card">
        <CardContent className="p-0">
          <div className="p-4 border-b border-gray-200 dark:border-gray-700">
            <Input
              placeholder="Search courses..."
              className="max-w-sm"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
          <div className="overflow-x-auto custom-scrollbar">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Code</TableHead>
                  <TableHead>Title</TableHead>
                  <TableHead>Faculty Name</TableHead>
                  <TableHead>Department</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {isLoading ? (
                  Array.from({ length: 3 }).map((_, i) => (
                    <TableRow key={`skeleton-${i}`}>
                      <TableCell colSpan={5}>
                        <div className="h-4 bg-gray-200 dark:bg-white/10 rounded animate-pulse" />
                      </TableCell>
                    </TableRow>
                  ))
                ) : filteredCourses.length === 0 ? (
                  <TableRow>
                    <TableCell
                      colSpan={5}
                      className="h-24 text-center text-gray-500"
                    >
                      No courses found matching your search.
                    </TableCell>
                  </TableRow>
                ) : (
                  paginatedCourses.map((course) => (
                    <TableRow key={course.id}>
                      <TableCell className="font-medium text-gray-900 dark:text-white">
                        {course.code}
                      </TableCell>
                      <TableCell>{course.title}</TableCell>
                      <TableCell className="text-gray-600 dark:text-gray-300">
                        {getFacultyName(course.facultyId)}
                      </TableCell>
                      <TableCell className="text-gray-600 dark:text-gray-300">
                        {getDepartmentName(course.departmentId)}
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex items-center justify-end gap-2">
                          <ViewButton
                            onClick={() => setViewCourse(course)}
                            tooltip="View"
                          />
                          <button
                            onClick={() => openModal("course", "edit", course)}
                            className="p-1.5 rounded-lg text-amber-500 hover:text-amber-600 hover:bg-amber-50 dark:hover:bg-amber-500/10 transition-colors"
                            title="Edit"
                          >
                            <Edit2 size={16} />
                          </button>
                          <button
                            onClick={() => setDeleteId(course.id)}
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
          <Pagination
            page={page}
            totalPages={totalPages}
            totalItems={filteredCourses.length}
            pageSize={PAGE_SIZE}
            onPageChange={setPage}
          />
        </CardContent>
      </Card>

      <CourseModal />
      <ConfirmDeleteModal
        isOpen={!!deleteId}
        onClose={() => setDeleteId(null)}
        onConfirm={async () => { if (deleteId) await deleteCourse(deleteId); }}
      />

      <ViewModal
        isOpen={!!viewCourse}
        onClose={() => setViewCourse(null)}
        title="Course Details"
        data={
          viewCourse
            ? [
                { label: "Code", value: viewCourse.code },
                { label: "Title", value: viewCourse.title },
                {
                  label: "Faculty",
                  value: getFacultyName(viewCourse.facultyId),
                },
                {
                  label: "Department",
                  value: getDepartmentName(viewCourse.departmentId),
                },
              ]
            : null
        }
      />
    </div>
  );
}
