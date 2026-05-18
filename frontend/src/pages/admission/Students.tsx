import { useEffect, useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useForm, Controller } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import {
  Search,
  Plus,
  Edit,
  Trash2,
  Filter,
  AlertTriangle,
} from "lucide-react";
import { Card, CardContent } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Input } from "@/components/ui/Input";
import { Modal } from "@/components/ui/Modal";
import { Badge } from "@/components/ui/Badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/Table";
import { Select } from "@/components/ui/Select";
import { useAdmissionStore } from "@/store/useAdmissionStore";
import type { Student } from "@/store/useAdmissionStore";
import { ViewButton } from "@/components/ui/ViewButton";
import { ViewModal } from "@/components/ui/ViewModal";

const studentSchema = z.object({
  fullName: z.string().min(2, "Full name must be at least 2 characters"),
  faculty: z.string().min(1, "Faculty is required"),
  department: z.string().min(1, "Department is required"),
});

type StudentForm = z.infer<typeof studentSchema>;

export default function Students() {
  const {
    students,
    faculties,
    departments,
    isLoading,
    isSaving,
    error,
    fetchAdmissionData,
    addStudent,
    updateStudent,
    deleteStudent,
  } = useAdmissionStore();
  const [searchParams] = useSearchParams();
  const [searchTerm, setSearchTerm] = useState("");
  const [statusFilter, setStatusFilter] = useState(
    searchParams.get("status") || "All",
  );

  // Modal states
  const [isAddModalOpen, setIsAddModalOpen] = useState(false);
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [isViewModalOpen, setIsViewModalOpen] = useState(false);
  const [isDeleteModalOpen, setIsDeleteModalOpen] = useState(false);
  const [selectedStudent, setSelectedStudent] = useState<Student | null>(null);

  const {
    register,
    handleSubmit,
    control,
    watch,
    reset,
    setValue,
    formState: { errors, isSubmitting },
  } = useForm<StudentForm>({
    resolver: zodResolver(studentSchema),
    defaultValues: {
      fullName: "",
      faculty: "",
      department: "",
    },
  });

  useEffect(() => {
    void fetchAdmissionData();
  }, [fetchAdmissionData]);

  const selectedFaculty = watch("faculty");

  const filteredStudents = useMemo(() => {
    return students.filter((student) => {
      const matchesSearch =
        student.fullName.toLowerCase().includes(searchTerm.toLowerCase()) ||
        student.studentNumber.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesStatus =
        statusFilter === "All" ||
        student.status.toLowerCase() === statusFilter.toLowerCase();
      return matchesSearch && matchesStatus;
    });
  }, [students, searchTerm, statusFilter]);

  const availableDepartments = useMemo(() => {
    return selectedFaculty ? departments[selectedFaculty] || [] : [];
  }, [selectedFaculty, departments]);

  // Handle Add Submit
  const onAddSubmit = async (data: StudentForm) => {
    await addStudent({
      fullName: data.fullName,
      faculty: data.faculty,
      department: data.department,
    });
    setIsAddModalOpen(false);
    reset();
  };

  // Handle Edit Submit
  const onEditSubmit = async (data: StudentForm) => {
    if (!selectedStudent) return;
    await updateStudent(selectedStudent.id, {
      fullName: data.fullName,
      faculty: data.faculty,
      department: data.department,
    });
    setIsEditModalOpen(false);
    setSelectedStudent(null);
    reset();
  };

  const openAddModal = () => {
    reset({ fullName: "", faculty: "", department: "" });
    setIsAddModalOpen(true);
  };

  const openEditModal = (student: Student) => {
    setSelectedStudent(student);
    reset({
      fullName: student.fullName,
      faculty: student.faculty,
      department: student.department,
    });
    setIsEditModalOpen(true);
  };

  const openViewModal = (student: Student) => {
    setSelectedStudent(student);
    setIsViewModalOpen(true);
  };

  const openDeleteModal = (student: Student) => {
    setSelectedStudent(student);
    setIsDeleteModalOpen(true);
  };

  const handleDeleteConfirm = async () => {
    if (selectedStudent) {
      await deleteStudent(selectedStudent.id);
      setIsDeleteModalOpen(false);
      setSelectedStudent(null);
    }
  };

  const closeModals = () => {
    setIsAddModalOpen(false);
    setIsEditModalOpen(false);
    setIsViewModalOpen(false);
    setIsDeleteModalOpen(false);
    setSelectedStudent(null);
    reset();
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white tracking-tight">
            Students
          </h1>
          <p className="text-gray-500 dark:text-gray-400 mt-1">
            Manage student applications and admissions.
          </p>
        </div>
        <Button
          onClick={openAddModal}
          className="shrink-0"
          disabled={isLoading}
        >
          <Plus size={20} className="mr-2" />
          Add New Student
        </Button>
      </div>

      {error ? (
        <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-500/20 dark:bg-red-500/10 dark:text-red-300">
          {error}
        </div>
      ) : null}

      <Card>
        <CardContent className="p-6">
          <div className="flex flex-col sm:flex-row gap-4 mb-6">
            <div className="relative flex-1">
              <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-gray-500">
                <Search size={18} />
              </div>
              <Input
                type="text"
                placeholder="Search by name or student number..."
                className="pl-10"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
            <div className="flex items-center gap-2 shrink-0">
              <Filter className="text-gray-400" size={18} />
              <Select
                value={statusFilter}
                onChange={(e) => setStatusFilter(e.target.value)}
                options={[
                  { value: "All", label: "All Statuses" },
                  { value: "Approved", label: "Approved" },
                  { value: "Pending", label: "Pending" },
                  { value: "Rejected", label: "Rejected" },
                ]}
              />
            </div>
          </div>

          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>ST-NO</TableHead>
                <TableHead>ST-Name</TableHead>
                <TableHead>Faculty</TableHead>
                <TableHead>Department</TableHead>
                <TableHead>Status</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {isLoading ? (
                Array.from({ length: 6 }).map((_, i) => (
                  <TableRow key={`skeleton-${i}`}>
                    <TableCell>
                      <div className="h-4 w-20 bg-gray-200 dark:bg-white/10 rounded animate-pulse" />
                    </TableCell>
                    <TableCell>
                      <div className="h-4 w-32 bg-gray-200 dark:bg-white/10 rounded animate-pulse" />
                    </TableCell>
                    <TableCell>
                      <div className="h-4 w-24 bg-gray-200 dark:bg-white/10 rounded animate-pulse" />
                    </TableCell>
                    <TableCell>
                      <div className="h-4 w-24 bg-gray-200 dark:bg-white/10 rounded animate-pulse" />
                    </TableCell>
                    <TableCell>
                      <div className="h-6 w-20 bg-gray-200 dark:bg-white/10 rounded-full animate-pulse" />
                    </TableCell>
                    <TableCell>
                      <div className="ml-auto h-8 w-20 bg-gray-200 dark:bg-white/10 rounded animate-pulse" />
                    </TableCell>
                  </TableRow>
                ))
              ) : filteredStudents.length > 0 ? (
                filteredStudents.map((student) => (
                  <TableRow key={student.id}>
                    <TableCell className="font-medium text-primary">
                      {student.studentNumber}
                    </TableCell>
                    <TableCell className="font-medium text-gray-900 dark:text-white">
                      {student.fullName}
                    </TableCell>
                    <TableCell>{student.faculty}</TableCell>
                    <TableCell>{student.department}</TableCell>
                    <TableCell>
                      <Badge
                        variant={
                          student.status === "approved"
                            ? "success"
                            : student.status === "pending"
                              ? "warning"
                              : "danger"
                        }
                      >
                        {student.status.charAt(0).toUpperCase() +
                          student.status.slice(1)}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-right">
                      <div className="flex items-center justify-end gap-2">
                        <ViewButton
                          onClick={() => openViewModal(student)}
                          tooltip="View"
                        />
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-8 w-8 p-0 text-yellow-500 hover:text-yellow-600 hover:bg-yellow-50 dark:text-yellow-400 dark:hover:text-yellow-300 dark:hover:bg-yellow-500/10"
                          onClick={() => openEditModal(student)}
                        >
                          <Edit size={16} />
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-8 w-8 p-0 text-red-500 hover:text-red-600 hover:bg-red-50 dark:text-red-400 dark:hover:text-red-300 dark:hover:bg-red-500/10"
                          onClick={() => openDeleteModal(student)}
                        >
                          <Trash2 size={16} />
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))
              ) : (
                <TableRow>
                  <TableCell
                    colSpan={7}
                    className="text-center py-8 text-gray-500"
                  >
                    No students found in the database.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      {/* Add Modal */}
      <Modal
        isOpen={isAddModalOpen}
        onClose={closeModals}
        title="Add New Student"
      >
        <form onSubmit={handleSubmit(onAddSubmit)} className="space-y-5">
          <div className="space-y-1.5">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Full Name
            </label>
            <Input
              {...register("fullName")}
              placeholder="Enter student's full name"
              error={errors.fullName?.message}
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Faculty
            </label>
            <Controller
              control={control}
              name="faculty"
              render={({ field }) => (
                <Select
                  value={field.value}
                  onChange={(e) => {
                    field.onChange(e.target.value);
                    setValue("department", "", { shouldValidate: true });
                  }}
                  error={errors.faculty?.message}
                  options={[
                    { value: "", label: "Select Faculty" },
                    ...faculties.map((f) => ({ value: f, label: f })),
                  ]}
                />
              )}
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Department
            </label>
            <Controller
              control={control}
              name="department"
              render={({ field }) => (
                <Select
                  value={field.value}
                  onChange={field.onChange}
                  disabled={!selectedFaculty}
                  error={errors.department?.message}
                  options={[
                    { value: "", label: "Select Department" },
                    ...availableDepartments.map((d: string) => ({
                      value: d,
                      label: d,
                    })),
                  ]}
                />
              )}
            />
          </div>

          <div className="pt-4 flex justify-end gap-3">
            <Button type="button" variant="secondary" onClick={closeModals}>
              Cancel
            </Button>
            <Button type="submit" isLoading={isSubmitting || isSaving}>
              Create Student
            </Button>
          </div>
        </form>
      </Modal>

      {/* Edit Modal */}
      <Modal
        isOpen={isEditModalOpen}
        onClose={closeModals}
        title="Edit Student"
      >
        <form onSubmit={handleSubmit(onEditSubmit)} className="space-y-5">
          <div className="space-y-1.5">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Full Name
            </label>
            <Input
              {...register("fullName")}
              placeholder="Enter student's full name"
              error={errors.fullName?.message}
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Faculty
            </label>
            <Controller
              control={control}
              name="faculty"
              render={({ field }) => (
                <Select
                  value={field.value}
                  onChange={(e) => {
                    field.onChange(e.target.value);
                    setValue("department", "", { shouldValidate: true });
                  }}
                  error={errors.faculty?.message}
                  options={[
                    { value: "", label: "Select Faculty" },
                    ...faculties.map((f) => ({ value: f, label: f })),
                  ]}
                />
              )}
            />
          </div>

          <div className="space-y-1.5">
            <label className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Department
            </label>
            <Controller
              control={control}
              name="department"
              render={({ field }) => (
                <Select
                  value={field.value}
                  onChange={field.onChange}
                  disabled={!selectedFaculty}
                  error={errors.department?.message}
                  options={[
                    { value: "", label: "Select Department" },
                    ...availableDepartments.map((d: string) => ({
                      value: d,
                      label: d,
                    })),
                  ]}
                />
              )}
            />
          </div>

          <div className="pt-4 flex justify-end gap-3">
            <Button type="button" variant="secondary" onClick={closeModals}>
              Cancel
            </Button>
            <Button type="submit" isLoading={isSubmitting || isSaving}>
              Save Changes
            </Button>
          </div>
        </form>
      </Modal>

      <ViewModal
        isOpen={isViewModalOpen}
        onClose={closeModals}
        title="Student Details"
        data={
          selectedStudent
            ? [
                { label: "ST-NO", value: selectedStudent.studentNumber },
                { label: "ST-Name", value: selectedStudent.fullName },
                { label: "Faculty", value: selectedStudent.faculty },
                { label: "Department", value: selectedStudent.department },
                {
                  label: "Images Captured",
                  value:
                    selectedStudent.imagesCaptured !== undefined
                      ? `${selectedStudent.imagesCaptured} Images`
                      : "None",
                },
                {
                  label: "Status",
                  value: (
                    <Badge
                      variant={
                        selectedStudent.status === "approved"
                          ? "success"
                          : selectedStudent.status === "pending"
                            ? "warning"
                            : "danger"
                      }
                    >
                      {selectedStudent.status.charAt(0).toUpperCase() +
                        selectedStudent.status.slice(1)}
                    </Badge>
                  ),
                },
              ]
            : null
        }
      />

      {/* Delete Modal */}
      <Modal
        isOpen={isDeleteModalOpen}
        onClose={closeModals}
        title="Confirm Deletion"
      >
        <div className="space-y-6">
          <div className="flex items-center gap-4 p-4 rounded-xl bg-red-50 dark:bg-red-500/10 border border-red-100 dark:border-red-500/20">
            <div className="w-10 h-10 rounded-full bg-red-100 dark:bg-red-500/20 flex items-center justify-center shrink-0">
              <AlertTriangle
                className="text-red-600 dark:text-red-400"
                size={20}
              />
            </div>
            <div>
              <h4 className="font-semibold text-red-900 dark:text-red-200 text-sm">
                Warning
              </h4>
              <p className="text-sm text-red-700 dark:text-red-300 mt-0.5">
                Are you sure you want to delete{" "}
                <span className="font-bold">{selectedStudent?.fullName}</span>?
                This action cannot be undone.
              </p>
            </div>
          </div>

          <div className="pt-2 flex justify-end gap-3">
            <Button variant="secondary" onClick={closeModals}>
              Cancel
            </Button>
            <Button
              className="bg-red-500 hover:bg-red-600 text-white border-transparent"
              disabled={isSaving}
              onClick={handleDeleteConfirm}
            >
              Delete Student
            </Button>
          </div>
        </div>
      </Modal>
    </div>
  );
}
