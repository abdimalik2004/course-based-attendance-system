import { useEffect, useState, useMemo } from "react";
import { Plus, Edit2, Trash2, Building2 } from "lucide-react";
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
import { FacultyModal } from "./components/FacultyModal";
import { ConfirmDeleteModal } from "@/components/academia/ConfirmDeleteModal";
import { format } from "date-fns";
import { ViewButton } from "@/components/ui/ViewButton";
import { ViewModal } from "@/components/ui/ViewModal";
import type { Faculty } from "@/types/academia.types";

export default function Faculties() {
  const { faculties, isLoading, error, fetchData, openModal, deleteFaculty } =
    useAcademiaStore();
  const [deleteId, setDeleteId] = useState<string | null>(null);
  const [viewFaculty, setViewFaculty] = useState<Faculty | null>(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [page, setPage] = useState(1);
  const PAGE_SIZE = 10;

  const filteredFaculties = useMemo(() => {
    setPage(1);
    return faculties.filter(
      (f) =>
        f.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
        f.code.toLowerCase().includes(searchTerm.toLowerCase()),
    );
  }, [faculties, searchTerm]);

  const totalPages = Math.max(1, Math.ceil(filteredFaculties.length / PAGE_SIZE));
  const paginatedFaculties = filteredFaculties.slice(
    (page - 1) * PAGE_SIZE,
    page * PAGE_SIZE,
  );

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-gray-100 flex items-center gap-3">
            <Building2 className="text-primary" size={28} />
            Faculties
          </h1>
          <p className="text-gray-500 dark:text-gray-400 mt-1">
            Manage Institutions faculties securely.
          </p>
          {error && (
            <div className="mt-3 rounded-2xl border border-rose-200 bg-rose-50 p-3 text-sm text-rose-700 dark:border-rose-500/20 dark:bg-rose-500/10 dark:text-rose-200">
              {error}
            </div>
          )}
        </div>
        <Button
          onClick={() => openModal("faculty", "create")}
          className="gap-2"
        >
          <Plus size={20} />
          Create Faculty
        </Button>
      </div>

      <Card className="glass-card">
        <CardContent className="p-0">
          <div className="p-4 border-b border-gray-200 dark:border-gray-700">
            <Input
              placeholder="Search faculties..."
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
                  <TableHead>Code</TableHead>
                  <TableHead>Created At</TableHead>
                  <TableHead>Years</TableHead>
                  <TableHead>Semesters</TableHead>
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
                ) : filteredFaculties.length === 0 ? (
                  <TableRow>
                    <TableCell
                      colSpan={6}
                      className="h-24 text-center text-gray-500"
                    >
                      No faculties found matching your search.
                    </TableCell>
                  </TableRow>
                ) : (
                  paginatedFaculties.map((faculty) => (
                    <TableRow key={faculty.id}>
                      <TableCell className="font-medium text-gray-900 dark:text-white">
                        {faculty.name}
                      </TableCell>
                      <TableCell>{faculty.code}</TableCell>
                      <TableCell className="text-gray-500 dark:text-gray-400">
                        {format(new Date(faculty.createdAt), "MMM dd, yyyy")}
                      </TableCell>
                      <TableCell>{faculty.years}</TableCell>
                      <TableCell className="font-medium text-emerald-600 dark:text-emerald-400">
                        {faculty.years * 2}
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex items-center justify-end gap-2">
                          <ViewButton
                            onClick={() => setViewFaculty(faculty)}
                            tooltip="View"
                          />
                          <button
                            onClick={() =>
                              openModal("faculty", "edit", faculty)
                            }
                            className="p-1.5 rounded-lg text-amber-500 hover:text-amber-600 hover:bg-amber-50 dark:hover:bg-amber-500/10 transition-colors"
                            title="Edit"
                          >
                            <Edit2 size={16} />
                          </button>
                          <button
                            onClick={() => setDeleteId(faculty.id)}
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
            totalItems={filteredFaculties.length}
            pageSize={PAGE_SIZE}
            onPageChange={setPage}
          />
        </CardContent>
      </Card>

      <FacultyModal />
      <ConfirmDeleteModal
        isOpen={!!deleteId}
        onClose={() => setDeleteId(null)}
        onConfirm={async () => { if (deleteId) await deleteFaculty(deleteId); }}
      />

      <ViewModal
        isOpen={!!viewFaculty}
        onClose={() => setViewFaculty(null)}
        title="Faculty Details"
        data={
          viewFaculty
            ? [
                { label: "Faculty Name", value: viewFaculty.name },
                { label: "Code", value: viewFaculty.code },
                {
                  label: "Created At",
                  value: format(
                    new Date(viewFaculty.createdAt),
                    "MMM dd, yyyy",
                  ),
                },
                { label: "Years", value: viewFaculty.years },
                { label: "Semesters", value: viewFaculty.years * 2 },
              ]
            : null
        }
      />
    </div>
  );
}
