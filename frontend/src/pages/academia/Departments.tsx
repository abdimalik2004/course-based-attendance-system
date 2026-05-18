import { useEffect, useState, useMemo } from "react";
import { Plus, Edit2, Trash2, Network } from "lucide-react";
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
import { DepartmentModal } from "./components/DepartmentModal";
import { ConfirmDeleteModal } from "@/components/academia/ConfirmDeleteModal";
import { format } from "date-fns";
import { ViewButton } from "@/components/ui/ViewButton";
import { ViewModal } from "@/components/ui/ViewModal";
import type { Department } from "@/types/academia.types";

export default function Departments() {
  const {
    departments,
    faculties,
    isLoading,
    error,
    fetchData,
    openModal,
    deleteDepartment,
  } = useAcademiaStore();
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const [viewDepartment, setViewDepartment] = useState<Department | null>(null);
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const getFacultyName = (facultyId: string) => {
    return faculties.find((f) => f.id === facultyId)?.name || "Unknown";
  };

  const filteredDepartments = useMemo(() => {
    return departments.filter((d) => {
      const searchMatch =
        d.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        d.code.toLowerCase().includes(searchTerm.toLowerCase());
      const facultyMatch = getFacultyName(d.facultyId)
        .toLowerCase()
        .includes(searchTerm.toLowerCase());
      return searchMatch || facultyMatch;
    });
  }, [departments, searchTerm, faculties]);

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-gray-100 flex items-center gap-3">
            <Network className="text-primary" size={28} />
            Departments
          </h1>
          <p className="text-gray-500 dark:text-gray-400 mt-1">
            Manage academic departments strictly tracked by Faculty.
          </p>
          {error && (
            <div className="mt-3 rounded-2xl border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700 dark:border-rose-500/20 dark:bg-rose-500/10 dark:text-rose-200">
              {error}
            </div>
          )}
        </div>
        <Button
          onClick={() => openModal("department", "create")}
          className="gap-2"
        >
          <Plus size={20} />
          Create Department
        </Button>
      </div>

      <Card className="glass-card">
        <CardContent className="p-0">
          <div className="p-4 border-b border-gray-200 dark:border-gray-700">
            <Input
              placeholder="Search departments..."
              className="max-w-sm"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
          <div className="overflow-x-auto custom-scrollbar">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Faculty Name</TableHead>
                  <TableHead>Department Name</TableHead>
                  <TableHead>Code</TableHead>
                  <TableHead>Created At</TableHead>
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
                ) : filteredDepartments.length === 0 ? (
                  <TableRow>
                    <TableCell
                      colSpan={5}
                      className="h-24 text-center text-gray-500"
                    >
                      No departments found matching your search.
                    </TableCell>
                  </TableRow>
                ) : (
                  filteredDepartments.map((dept) => (
                    <TableRow key={dept.id}>
                      <TableCell className="text-gray-600 dark:text-gray-300">
                        {getFacultyName(dept.facultyId)}
                      </TableCell>
                      <TableCell className="font-medium text-gray-900 dark:text-white">
                        {dept.name}
                      </TableCell>
                      <TableCell>{dept.code}</TableCell>
                      <TableCell className="text-gray-500 dark:text-gray-400">
                        {format(new Date(dept.createdAt), "MMM dd, yyyy")}
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex items-center justify-end gap-2">
                          <ViewButton
                            onClick={() => setViewDepartment(dept)}
                            tooltip="View"
                          />
                          <button
                            onClick={() =>
                              openModal("department", "edit", dept)
                            }
                            className="p-1.5 rounded-lg text-amber-500 hover:text-amber-600 hover:bg-amber-50 dark:hover:bg-amber-500/10 transition-colors"
                            title="Edit"
                          >
                            <Edit2 size={16} />
                          </button>
                          <button
                            onClick={() => setDeleteId(dept.id)}
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

      <DepartmentModal />
      <ConfirmDeleteModal
        isOpen={!!deleteId}
        onClose={() => setDeleteId(null)}
        onConfirm={() => deleteId && deleteDepartment(deleteId)}
      />

      <ViewModal
        isOpen={!!viewDepartment}
        onClose={() => setViewDepartment(null)}
        title="Department Details"
        data={
          viewDepartment
            ? [
                {
                  label: "Faculty",
                  value: getFacultyName(viewDepartment.facultyId),
                },
                { label: "Department Name", value: viewDepartment.name },
                { label: "Code", value: viewDepartment.code },
                {
                  label: "Created At",
                  value: format(
                    new Date(viewDepartment.createdAt),
                    "MMM dd, yyyy",
                  ),
                },
              ]
            : null
        }
      />
    </div>
  );
}
