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
} from "@/components/academia/Table";
import { Input } from "@/components/ui/Input";
import { useAcademiaStore } from "@/store/useAcademiaStore";
import { ClassModal } from "./components/ClassModal";
import { ConfirmDeleteModal } from "@/components/academia/ConfirmDeleteModal";
import { format } from "date-fns";
import { ViewButton } from "@/components/ui/ViewButton";
import { ViewModal } from "@/components/ui/ViewModal";
import type { Class } from "@/types/academia.types";

export default function Classes() {
  const {
    classes,
    faculties,
    departments,
    isLoading,
    error,
    fetchData,
    openModal,
    deleteClass,
  } = useAcademiaStore();
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const [viewClass, setViewClass] = useState<Class | null>(null);
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const getFacultyName = (facultyId: string) => {
    return faculties.find((f) => f.id === facultyId)?.name || "Unknown";
  };

  const getDepartmentName = (deptId: string) => {
    return departments.find((d) => d.id === deptId)?.name || "Unknown";
  };

  const filteredClasses = useMemo(() => {
    return classes.filter((cls) => {
      const searchMatch = cls.name
        .toLowerCase()
        .includes(searchTerm.toLowerCase());
      const facultyMatch = getFacultyName(cls.facultyId)
        .toLowerCase()
        .includes(searchTerm.toLowerCase());
      const deptMatch = getDepartmentName(cls.departmentId)
        .toLowerCase()
        .includes(searchTerm.toLowerCase());
      return searchMatch || facultyMatch || deptMatch;
    });
  }, [classes, searchTerm, faculties, departments]);

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-gray-100 flex items-center gap-3">
            <Users className="text-primary" size={28} />
            Classes
          </h1>
          <p className="text-gray-500 dark:text-gray-400 mt-1">
            Manage scheduled classes across distinct faculties.
          </p>
          {error && (
            <div className="mt-3 rounded-2xl border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700 dark:border-rose-500/20 dark:bg-rose-500/10 dark:text-rose-200">
              {error}
            </div>
          )}
        </div>
        <Button onClick={() => openModal("class", "create")} className="gap-2">
          <Plus size={20} />
          Create Class
        </Button>
      </div>

      <Card className="glass-card">
        <CardContent className="p-0">
          <div className="p-4 border-b border-gray-200 dark:border-gray-700">
            <Input
              placeholder="Search classes..."
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
                  <TableHead>Faculty Name</TableHead>
                  <TableHead>Department</TableHead>
                  <TableHead>Year</TableHead>
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
                ) : filteredClasses.length === 0 ? (
                  <TableRow>
                    <TableCell
                      colSpan={6}
                      className="h-24 text-center text-gray-500"
                    >
                      No classes found matching your search.
                    </TableCell>
                  </TableRow>
                ) : (
                  filteredClasses.map((cls) => (
                    <TableRow key={cls.id}>
                      <TableCell className="font-medium text-gray-900 dark:text-white">
                        {cls.name}
                      </TableCell>
                      <TableCell className="text-gray-600 dark:text-gray-300">
                        {getFacultyName(cls.facultyId)}
                      </TableCell>
                      <TableCell className="text-gray-600 dark:text-gray-300">
                        {getDepartmentName(cls.departmentId)}
                      </TableCell>
                      <TableCell className="font-medium text-purple-600 dark:text-purple-400">
                        {cls.year}
                      </TableCell>
                      <TableCell className="text-gray-500 dark:text-gray-400">
                        {format(new Date(cls.createdAt), "MMM dd, yyyy")}
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex items-center justify-end gap-2">
                          <ViewButton
                            onClick={() => setViewClass(cls)}
                            tooltip="View"
                          />
                          <button
                            onClick={() => openModal("class", "edit", cls)}
                            className="p-1.5 rounded-lg text-amber-500 hover:text-amber-600 hover:bg-amber-50 dark:hover:bg-amber-500/10 transition-colors"
                            title="Edit"
                          >
                            <Edit2 size={16} />
                          </button>
                          <button
                            onClick={() => setDeleteId(cls.id)}
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

      <ClassModal />
      <ConfirmDeleteModal
        isOpen={!!deleteId}
        onClose={() => setDeleteId(null)}
        onConfirm={() => deleteId && deleteClass(deleteId)}
      />

      <ViewModal
        isOpen={!!viewClass}
        onClose={() => setViewClass(null)}
        title="Class Details"
        data={
          viewClass
            ? [
                { label: "Class Name", value: viewClass.name },
                {
                  label: "Faculty",
                  value: getFacultyName(viewClass.facultyId),
                },
                {
                  label: "Department",
                  value: getDepartmentName(viewClass.departmentId),
                },
                { label: "Year", value: viewClass.year },
                {
                  label: "Created At",
                  value: format(new Date(viewClass.createdAt), "MMM dd, yyyy"),
                },
              ]
            : null
        }
      />
    </div>
  );
}
