import { useState, useEffect, useMemo } from "react";
import {
  Plus,
  Edit2,
  Trash2,
  GraduationCap,
  Calendar,
  BookOpen,
  Users,
} from "lucide-react";
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
import { Badge } from "@/components/ui/Badge";
import { Input } from "@/components/ui/Input";
import { useAcademiaStore } from "@/store/useAcademiaStore";
import { cn } from "@/utils/cn";
import { AddStructureModal } from "./components/AddStructureModal";
import { CourseAssignmentsTable } from "./components/CourseAssignmentsTable";
import { CourseAssignModal } from "./components/CourseAssignModal";
import { ClassAssignmentsTable } from "./components/ClassAssignmentsTable";
import { ClassAssignModal } from "./components/ClassAssignModal";
import { ViewButton } from "@/components/ui/ViewButton";
import { ViewModal } from "@/components/ui/ViewModal";
import type { AcademicStructure as AcademicStructureType } from "@/types/academia.types";

export default function AcademicStructure() {
  const [activeTab, setActiveTab] = useState<"terms" | "courses" | "classes">(
    "terms",
  );
  const {
    structures,
    isLoading,
    error,
    fetchData,
    openModal,
    deleteStructure,
  } = useAcademiaStore();
  const [viewStructure, setViewStructure] =
    useState<AcademicStructureType | null>(null);
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const filteredStructures = useMemo(() => {
    return structures.filter((struct) => {
      const searchMatch =
        struct.academicYear.toLowerCase().includes(searchTerm.toLowerCase()) ||
        struct.term.toLowerCase().includes(searchTerm.toLowerCase());
      return searchMatch;
    });
  }, [structures, searchTerm]);

  const handleDelete = (id: string) => {
    if (
      window.confirm("Are you sure you want to delete this term structure?")
    ) {
      deleteStructure(id);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-gray-100 flex items-center gap-3">
            <GraduationCap className="text-primary-accent" size={28} />
            Academic Structure
          </h1>
          <p className="text-gray-500 dark:text-gray-400 mt-1">
            Configure academic years and assignment schedules.
          </p>
          {error && (
            <div className="mt-3 rounded-2xl border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700 dark:border-rose-500/20 dark:bg-rose-500/10 dark:text-rose-200">
              {error}
            </div>
          )}
        </div>
        {activeTab === "terms" && (
          <Button
            onClick={() => openModal("structure", "create")}
            className="gap-2"
          >
            <Plus size={20} />
            Create Term
          </Button>
        )}
        {activeTab === "courses" && (
          <Button
            onClick={() => openModal("courseAssign", "create")}
            className="gap-2"
          >
            <Plus size={20} />
            Assign Course
          </Button>
        )}
        {activeTab === "classes" && (
          <Button
            onClick={() => openModal("classAssign", "create")}
            className="gap-2"
          >
            <Plus size={20} />
            Assign Class
          </Button>
        )}
      </div>

      <div className="flex space-x-1 p-1 bg-gray-100/50 dark:bg-white/5 rounded-xl border border-gray-200 dark:border-white/10 max-w-fit">
        <button
          onClick={() => setActiveTab("terms")}
          className={cn(
            "flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg transition-all",
            activeTab === "terms"
              ? "bg-white dark:bg-primary/20 text-primary shadow-sm"
              : "text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200",
          )}
        >
          <Calendar size={18} />
          Define Academic Year
        </button>
        <button
          onClick={() => setActiveTab("courses")}
          className={cn(
            "flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg transition-all",
            activeTab === "courses"
              ? "bg-white dark:bg-blue-500/20 text-blue-600 dark:text-blue-400 shadow-sm"
              : "text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200",
          )}
        >
          <BookOpen size={18} />
          Assign Courses to Semesters
        </button>
        <button
          onClick={() => setActiveTab("classes")}
          className={cn(
            "flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg transition-all",
            activeTab === "classes"
              ? "bg-white dark:bg-emerald-500/20 text-emerald-600 dark:text-emerald-400 shadow-sm"
              : "text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200",
          )}
        >
          <Users size={18} />
          Assign Classes to Courses
        </button>
      </div>

      {activeTab === "terms" ? (
        <Card className="glass-card">
          <CardContent className="p-0">
            <div className="p-4 border-b border-gray-200 dark:border-gray-700">
              <Input
                placeholder="Search terms..."
                className="max-w-sm"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
            <div className="overflow-x-auto custom-scrollbar">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Academic Year</TableHead>
                    <TableHead>Term</TableHead>
                    <TableHead>Start Date</TableHead>
                    <TableHead>End Date</TableHead>
                    <TableHead>Status</TableHead>
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
                  ) : filteredStructures.length === 0 ? (
                    <TableRow>
                      <TableCell
                        colSpan={6}
                        className="h-24 text-center text-gray-500"
                      >
                        No structures found matching your search.
                      </TableCell>
                    </TableRow>
                  ) : (
                    filteredStructures.map((struct) => (
                      <TableRow key={struct.id}>
                        <TableCell className="font-medium text-gray-900 dark:text-gray-100">
                          {struct.academicYear}
                        </TableCell>
                        <TableCell className="text-gray-900 dark:text-gray-100">
                          {struct.term}
                        </TableCell>
                        <TableCell className="text-gray-600 dark:text-gray-300">
                          {new Date(struct.startDate).toLocaleDateString()}
                        </TableCell>
                        <TableCell className="text-gray-600 dark:text-gray-300">
                          {new Date(struct.endDate).toLocaleDateString()}
                        </TableCell>
                        <TableCell>
                          <Badge
                            variant={
                              struct.status === "Active"
                                ? "success"
                                : struct.status === "Draft"
                                  ? "warning"
                                  : "danger"
                            }
                          >
                            {struct.status}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-right">
                          <div className="flex items-center justify-end gap-2">
                            <ViewButton
                              onClick={() => setViewStructure(struct)}
                              tooltip="View"
                            />
                            <button
                              onClick={() => openModal("structure", "edit", struct)}
                              className="p-1.5 rounded-lg text-amber-500 hover:text-amber-600 hover:bg-amber-50 dark:hover:bg-amber-500/10 transition-colors"
                              title="Edit"
                            >
                              <Edit2 size={16} />
                            </button>
                            <button
                              onClick={() => handleDelete(struct.id)}
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
      ) : activeTab === "courses" ? (
        <CourseAssignmentsTable />
      ) : (
        <ClassAssignmentsTable />
      )}

      <AddStructureModal />
      <CourseAssignModal />
      <ClassAssignModal />

      <ViewModal
        isOpen={!!viewStructure}
        onClose={() => setViewStructure(null)}
        title="Term Details"
        data={
          viewStructure
            ? [
                { label: "Academic Year", value: viewStructure.academicYear },
                { label: "Term", value: viewStructure.term },
                {
                  label: "Start Date",
                  value: new Date(viewStructure.startDate).toLocaleDateString(),
                },
                {
                  label: "End Date",
                  value: new Date(viewStructure.endDate).toLocaleDateString(),
                },
                {
                  label: "Status",
                  value: (
                    <Badge
                      variant={
                        viewStructure.status === "Active"
                          ? "success"
                          : viewStructure.status === "Draft"
                            ? "warning"
                            : "danger"
                      }
                    >
                      {viewStructure.status}
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
